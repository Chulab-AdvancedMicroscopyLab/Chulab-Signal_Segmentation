"""
profile_GUSL.py — per-stage timing and memory profiler for GUSLModel.

Profiles each pipeline stage per level:
  prepare   — FeatGen pad + gradient map precomputation (once per z-chunk)
  feat      — FeatGen transform_precomputed (numba + Saab, per pixel batch)
  rft+lnt   — GPU pipeline: X upload → RFT column select → LNT matmuls → download
  xgb       — XGBoost predict (per pixel batch)

Requires a trained model file. Input is synthetic random data — no real dataset needed.

Usage:
  python profile_GUSL.py --config configs/config_GUSL.json
  python profile_GUSL.py --model_path output/Lectin/GUSL_v1/gusl_model.joblib
  python profile_GUSL.py --model_path ... --pixel_batch 1000000
  python profile_GUSL.py --model_path ... --level 2 --n_frames 16 --height 3627 --width 2782
  python profile_GUSL.py --model_path ... --n_warmup 1 --n_runs 5
"""

import argparse
import gc
import json
import logging
import os
import sys
import time

import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.WARNING)   # suppress model internals during profiling

from models.GUSL import GUSLModel


# ── formatting helpers ────────────────────────────────────────────────────────

def _fmt_time(s):
    if s < 1e-3: return f"{s*1e6:.1f} µs"
    if s < 1:    return f"{s*1e3:.2f} ms"
    return f"{s:.3f} s"

def _fmt_flops(f):
    if f <= 0:   return "—"
    if f >= 1e12: return f"{f/1e12:.2f} TFLOPs"
    if f >= 1e9:  return f"{f/1e9:.2f} GFLOPs"
    if f >= 1e6:  return f"{f/1e6:.2f} MFLOPs"
    return f"{f:.0f} FLOPs"

def _fmt_mem(b):
    if b >= 1 << 30: return f"{b / (1<<30):.2f} GB"
    if b >= 1 << 20: return f"{b / (1<<20):.1f} MB"
    return f"{b / (1<<10):.1f} KB"

def _sync(device_str):
    if device_str.startswith("cuda"):
        import torch
        torch.cuda.synchronize()

def _gpu_mem():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved()
    except Exception:
        pass
    return 0, 0

def _reset_gpu_mem():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ── stage profilers ───────────────────────────────────────────────────────────

def time_stage(fn, n_warmup, n_runs, sync_device=None):
    """Run fn() n_warmup+n_runs times; return (mean_s, std_s, results_list)."""
    results = []
    for i in range(n_warmup + n_runs):
        if sync_device:
            _sync(sync_device)
        t0 = time.perf_counter()
        r = fn()
        if sync_device:
            _sync(sync_device)
        elapsed = time.perf_counter() - t0
        if i >= n_warmup:
            results.append(elapsed)
    times = np.array(results)
    return float(times.mean()), float(times.std())


def profile_prepare(fg, imgs, n_warmup, n_runs):
    fn = lambda: fg.prepare(imgs)
    return time_stage(fn, n_warmup, n_runs)


def profile_feat(fg, fg_r, ctx_img, ctx_res, keep_lin, n_warmup, n_runs):
    if fg_r is not None and ctx_res is not None:
        def fn():
            X_img = fg.transform_precomputed(ctx_img, keep_lin)
            X_res = fg_r.transform_precomputed(ctx_res, keep_lin)
            return np.concatenate([X_img, X_res], axis=1).astype(np.float32, copy=False)
    else:
        def fn():
            return fg.transform_precomputed(ctx_img, keep_lin)
    return time_stage(fn, n_warmup, n_runs)


def profile_rft_lnt_gpu(X, rft, lnt, device_str, n_warmup, n_runs):
    import torch
    rft_idx = torch.from_numpy(rft.selected_features).to(device_str)
    lnt_kernels = [torch.from_numpy(np.ascontiguousarray(k)).to(device_str) for k in lnt.svd]

    def fn():
        X_t = torch.from_numpy(np.ascontiguousarray(X)).to(device_str)
        X_rft_t = X_t[:, rft_idx].contiguous()
        del X_t
        parts = [X_rft_t]
        for k_t in lnt_kernels:
            parts.append(X_rft_t @ k_t)
        X_final_t = torch.cat(parts, dim=1)
        del X_rft_t, parts
        result = X_final_t.cpu().numpy().astype(np.float32, copy=False)
        del X_final_t
        return result

    mean, std = time_stage(fn, n_warmup, n_runs, sync_device=device_str)
    # Clean cached GPU tensors
    del rft_idx, lnt_kernels
    return mean, std


def profile_rft_lnt_cpu(X, rft, lnt, n_warmup, n_runs):
    def fn():
        X_rft = rft.transform(X)
        X_lnt = lnt.transform(X_rft)
        return np.concatenate([X_rft, X_lnt], axis=1).astype(np.float32, copy=False)
    return time_stage(fn, n_warmup, n_runs)


def profile_xgb(X_final, xgb_model, n_warmup, n_runs):
    dmat = xgb.DMatrix(X_final)
    fn = lambda: xgb_model.get_booster().predict(dmat)
    mean, std = time_stage(fn, n_warmup, n_runs)
    del dmat
    return mean, std


# ── per-level profiling ───────────────────────────────────────────────────────

def profile_level(model, level, n_frames, H_orig, W_orig, pixel_batch, device_str,
                  n_warmup, n_runs):
    import cv2

    print(f"\n  {'─'*64}")
    print(f"  Level {level}")
    print(f"  {'─'*64}")

    lvl_idx = model.deepest_level - level
    fg      = model.featgen_img[lvl_idx]
    fg_r    = model.featgen_res[lvl_idx]
    rft     = model.rft_models[lvl_idx]
    lnt     = model.lnt_models[lvl_idx]
    xgb_model = model.xgb_models[lvl_idx]

    if any(m is None for m in [fg, rft, lnt, xgb_model]):
        print(f"  [SKIP] Level {level} not trained.")
        return

    scale = 2 ** (level - 1)
    Hl = max(1, H_orig // scale)
    Wl = max(1, W_orig // scale)
    n_pixels = n_frames * Hl * Wl
    n_batches = (n_pixels + pixel_batch - 1) // pixel_batch
    print(f"  Volume   : ({n_frames}, {Hl}, {Wl})  pixels={n_pixels:,}  batches={n_batches}")

    # Synthetic input
    imgs = np.random.rand(n_frames, Hl, Wl).astype(np.float32)
    if level < model.deepest_level:
        res_imgs = np.random.rand(n_frames, Hl, Wl).astype(np.float32)
    else:
        res_imgs = None

    # ── prepare ──
    print(f"\n  [prepare]  n_warmup={n_warmup}  n_runs={n_runs}")
    _reset_gpu_mem()
    t_prep_mean, t_prep_std = profile_prepare(fg, imgs, n_warmup, n_runs)
    ctx_img = fg.prepare(imgs)
    ctx_res = fg_r.prepare(res_imgs) if (fg_r is not None and res_imgs is not None) else None
    peak_alloc, peak_reserved = _gpu_mem()
    print(f"  prepare   : {_fmt_time(t_prep_mean)} ± {_fmt_time(t_prep_std)}"
          + (f"  [×2 for residual FeatGen at this level]" if fg_r is not None else ""))
    _print_mem_ctx(ctx_img, ctx_res)

    # ── feat ──
    print(f"\n  [feat]  pixel_batch={pixel_batch:,}  n_warmup={n_warmup}  n_runs={n_runs}")
    keep_lin = np.arange(min(pixel_batch, n_pixels), dtype=np.int64)
    _reset_gpu_mem()
    t_feat_mean, t_feat_std = profile_feat(fg, fg_r, ctx_img, ctx_res, keep_lin, n_warmup, n_runs)
    X = (fg.transform_precomputed(ctx_img, keep_lin) if fg_r is None
         else np.concatenate([fg.transform_precomputed(ctx_img, keep_lin),
                               fg_r.transform_precomputed(ctx_res, keep_lin)], axis=1).astype(np.float32, copy=False))
    peak_alloc, _ = _gpu_mem()
    print(f"  feat      : {_fmt_time(t_feat_mean)} ± {_fmt_time(t_feat_std)}"
          f"   X.shape={X.shape}  {_fmt_mem(X.nbytes)}"
          + (f"  GPU peak {_fmt_mem(peak_alloc)}" if peak_alloc > 0 else ""))
    print(f"  throughput: {len(keep_lin)/t_feat_mean:,.0f} pixels/s")

    # ── rft+lnt ──
    print(f"\n  [rft+lnt pipeline]")
    _reset_gpu_mem()
    if device_str.startswith("cuda"):
        t_rl_mean, t_rl_std = profile_rft_lnt_gpu(X, rft, lnt, device_str, n_warmup, n_runs)
        X_final = _build_x_final_gpu(X, rft, lnt, device_str)
    else:
        t_rl_mean, t_rl_std = profile_rft_lnt_cpu(X, rft, lnt, n_warmup, n_runs)
        X_rft = rft.transform(X)
        X_lnt = lnt.transform(X_rft)
        X_final = np.concatenate([X_rft, X_lnt], axis=1).astype(np.float32, copy=False)
    peak_alloc, _ = _gpu_mem()
    print(f"  rft+lnt   : {_fmt_time(t_rl_mean)} ± {_fmt_time(t_rl_std)}"
          f"   X_final.shape={X_final.shape}  {_fmt_mem(X_final.nbytes)}"
          + (f"  GPU peak {_fmt_mem(peak_alloc)}" if peak_alloc > 0 else ""))
    print(f"  throughput: {len(keep_lin)/t_rl_mean:,.0f} pixels/s")

    # ── xgb ──
    print(f"\n  [xgb]")
    _reset_gpu_mem()
    t_xgb_mean, t_xgb_std = profile_xgb(X_final, xgb_model, n_warmup, n_runs)
    peak_alloc, _ = _gpu_mem()
    print(f"  xgb       : {_fmt_time(t_xgb_mean)} ± {_fmt_time(t_xgb_std)}"
          + (f"  GPU peak {_fmt_mem(peak_alloc)}" if peak_alloc > 0 else ""))
    print(f"  throughput: {len(keep_lin)/t_xgb_mean:,.0f} pixels/s")

    # ── FLOPs ──
    saab_flops, lnt_flops, xgb_cmp = compute_flops(
        fg, fg_r, lnt, xgb_model, len(keep_lin), model.max_depth
    )
    total_flops = saab_flops + lnt_flops
    print(f"\n  ── FLOPs per batch (batch={len(keep_lin):,} pixels) ──")
    print(f"  Note: FLOPs = 2×MACs (matches profile_model.py convention).")
    print(f"        Saab+LNT FLOPs are directly comparable to UNet Conv/Linear FLOPs.")
    print(f"        XGBoost comparisons are a different op type — listed separately.")
    w2 = 14
    print(f"  {'Stage':<{w2}} {'FLOPs':>16}  {'FLOPs/pixel':>12}  {'% FLOPs':>8}")
    print(f"  {'-'*w2} {'-'*16}  {'-'*12}  {'-'*8}")
    for label, flops in [("Saab", saab_flops), ("LNT", lnt_flops)]:
        pct = 100 * flops / total_flops if total_flops > 0 else 0
        print(f"  {label:<{w2}} {_fmt_flops(flops):>16}  "
              f"{_fmt_flops(flops/len(keep_lin)):>12}  {pct:>7.1f}%")
    print(f"  {'-'*w2} {'-'*16}  {'-'*12}  {'-'*8}")
    print(f"  {'Total FLOPs':<{w2}} {_fmt_flops(total_flops):>16}  "
          f"{_fmt_flops(total_flops/len(keep_lin)):>12}  {'100.0%':>8}")
    print(f"  {'XGBoost (cmp)':<{w2}} {_fmt_flops(xgb_cmp):>16}  "
          f"{_fmt_flops(xgb_cmp/len(keep_lin)):>12}  {'—':>8}  ← comparisons, not FLOPs")
    grand_total = total_flops + xgb_cmp
    print(f"  {'─'*w2} {'─'*16}  {'─'*12}  {'─'*8}")
    print(f"  {'Grand Total':<{w2}} {_fmt_flops(grand_total):>16}  "
          f"{_fmt_flops(grand_total/len(keep_lin)):>12}  {'—':>8}  ← FLOPs + comparisons")

    # ── timing summary ──
    t_per_batch = t_feat_mean + t_rl_mean + t_xgb_mean
    t_total_batches = t_per_batch * n_batches
    t_total_chunks = t_prep_mean + t_total_batches
    print(f"\n  ── per-batch timing summary ──")
    total = t_feat_mean + t_rl_mean + t_xgb_mean
    for label, t in [("feat", t_feat_mean), ("rft+lnt", t_rl_mean), ("xgb", t_xgb_mean)]:
        bar = "█" * int(20 * t / total) if total > 0 else ""
        print(f"  {label:<10} {_fmt_time(t):>10}  {100*t/total:>5.1f}%  {bar}")
    print(f"  {'total':<10} {_fmt_time(total):>10}")
    print(f"\n  ── z-chunk projection ({n_batches} batches) ──")
    print(f"  prepare (once) : {_fmt_time(t_prep_mean)}"
          + (f"  [×2 w/ residual]" if fg_r is not None else ""))
    print(f"  {n_batches} batches × {_fmt_time(t_per_batch):>8} = {_fmt_time(t_total_batches)}")
    print(f"  z-chunk total  : {_fmt_time(t_total_chunks)}")

    del ctx_img, ctx_res, X, X_final
    gc.collect()

    # Return per-original-pixel FLOPs: each level's contribution weighted by
    # its pixel count so the caller can sum across levels and divide by n_orig_pixels.
    batch_size = len(keep_lin)
    return {
        "saab_flops_total":  saab_flops  / batch_size * n_pixels,
        "lnt_flops_total":   lnt_flops   / batch_size * n_pixels,
        "xgb_cmp_total":     xgb_cmp     / batch_size * n_pixels,
        "n_pixels":          n_pixels,
    }


def _build_x_final_gpu(X, rft, lnt, device_str):
    import torch
    X_t = torch.from_numpy(np.ascontiguousarray(X)).to(device_str)
    idx_t = torch.from_numpy(rft.selected_features).to(device_str)
    X_rft_t = X_t[:, idx_t].contiguous()
    del X_t, idx_t
    parts = [X_rft_t]
    for k in lnt.svd:
        k_t = torch.from_numpy(np.ascontiguousarray(k)).to(device_str)
        parts.append(X_rft_t @ k_t)
    X_final_t = torch.cat(parts, dim=1)
    del X_rft_t, parts
    result = X_final_t.cpu().numpy().astype(np.float32, copy=False)
    del X_final_t
    return result


def compute_flops(fg, fg_r, lnt, xgb_model, batch_size, max_depth):
    """
    Returns (saab_flops, lnt_flops, xgb_comparisons).

    FLOPs = multiply-accumulate counted as 2 ops (matches profile_model.py convention).
    saab_flops / lnt_flops are directly comparable to UNet Conv/Linear FLOPs.

    xgb_comparisons — tree-node comparisons (NOT FLOPs); listed separately since
    they are a different op type and cannot be summed with saab/lnt FLOPs.
    """
    def _saab_flops(fg_obj):
        if fg_obj is None:
            return 0
        kernels = getattr(fg_obj.saab, "Kernels", None)
        if kernels is None:
            return 0
        out_per_patch = int(kernels.shape[0])
        patch_dim     = int(fg_obj.kd * fg_obj.k * fg_obj.k)
        n_neigh       = int(len(fg_obj._neigh_off_d_1d) * len(fg_obj._neigh_off_xy_1d) ** 2)
        # matmul: (batch × n_neigh, patch_dim) @ (patch_dim, out_per_patch) → 2×MACs = FLOPs
        return 2 * batch_size * n_neigh * patch_dim * out_per_patch

    saab_flops = _saab_flops(fg) + _saab_flops(fg_r)

    lnt_flops = 0
    if lnt is not None and hasattr(lnt, "svd"):
        for K in lnt.svd:
            # matmul: (batch, K.shape[0]) @ (K.shape[0], K.shape[1]) → 2×MACs = FLOPs
            lnt_flops += 2 * batch_size * int(K.shape[0]) * int(K.shape[1])

    xgb_comparisons = 0
    if xgb_model is not None:
        try:
            n_trees = xgb_model.get_booster().num_boosted_rounds()
        except Exception:
            n_trees = int(getattr(xgb_model, "n_estimators", 0) or 0)
        # Upper-bound: full binary tree of depth max_depth has 2^max_depth - 1 internal nodes
        nodes_per_tree = max(1, (2 ** max_depth) - 1)
        xgb_comparisons = batch_size * n_trees * nodes_per_tree

    return saab_flops, lnt_flops, xgb_comparisons


def _print_mem_ctx(ctx_img, ctx_res):
    def _ctx_bytes(ctx):
        if ctx is None:
            return 0
        return sum(v.nbytes for v in ctx.values() if isinstance(v, np.ndarray))
    img_b = _ctx_bytes(ctx_img)
    res_b = _ctx_bytes(ctx_res)
    total = img_b + res_b
    s = f"  ctx_img   : {_fmt_mem(img_b)}"
    if res_b > 0:
        s += f"  ctx_res: {_fmt_mem(res_b)}"
    s += f"  (total {_fmt_mem(total)})"
    print(s)


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GUSLModel per-stage profiler")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--config",     type=str, help="Path to config_GUSL.json (reads model_path and volume shape)")
    g.add_argument("--model_path", type=str, help="Direct path to gusl_model.joblib")
    p.add_argument("--level",      type=int, default=None,
                   help="Profile only this level (default: all levels in model)")
    p.add_argument("--n_frames",   type=int, default=32,
                   help="Number of Z-frames in synthetic volume (default: 16)")
    p.add_argument("--height",     type=int, default=1000,
                   help="Y size of synthetic volume at level 1 (default: 3627)")
    p.add_argument("--width",      type=int, default=1000,
                   help="X size of synthetic volume at level 1 (default: 2782)")
    p.add_argument("--pixel_batch",type=int, default=1_000_000,
                   help="Pixels per inference batch (default: 1_000_000)")
    p.add_argument("--device",     type=str, default=None,
                   help="Override device (default: from config or cuda:0)")
    p.add_argument("--n_warmup",   type=int, default=2)
    p.add_argument("--n_runs",     type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve model path and device
    model_path = None
    device_str = args.device or "cuda:0"
    H_orig, W_orig = args.height, args.width

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        model_path = cfg["inference"]["model_path"]
        device_str = args.device or cfg["inference"].get("device", "cuda:0")
        pixel_batch = args.pixel_batch or cfg["inference"].get("inference_pixel_batch", 500_000)
        H_orig = args.height
        W_orig = args.width
    elif args.model_path:
        model_path = args.model_path
        pixel_batch = args.pixel_batch
    else:
        print("Error: provide --config or --model_path")
        sys.exit(1)

    pixel_batch = args.pixel_batch

    # Header
    try:
        import torch
        gpu_name = torch.cuda.get_device_name(device_str) if device_str.startswith("cuda") else ""
    except Exception:
        gpu_name = ""

    print(f"\n{'━'*68}")
    print(f"  GUSL Model Profiler")
    print(f"{'━'*68}")
    print(f"  Model path  : {model_path}")
    print(f"  Device      : {device_str}" + (f"  ({gpu_name})" if gpu_name else ""))
    print(f"  Volume      : ({args.n_frames}, {H_orig}, {W_orig})  [level 1 / finest]")
    print(f"  Pixel batch : {pixel_batch:,}")
    print(f"  Timing      : {args.n_warmup} warmup + {args.n_runs} timed runs per stage")

    print(f"\n  Loading model...", end="", flush=True)
    t0 = time.perf_counter()
    model: GUSLModel = GUSLModel.load(model_path)
    model.set_device(device_str)
    print(f" done ({_fmt_time(time.perf_counter()-t0)})")

    print(f"  deepest_level={model.deepest_level}  base_size={model.base_size}")

    levels = [args.level] if args.level else list(range(model.deepest_level, 0, -1))
    n_orig_pixels = args.n_frames * H_orig * W_orig

    level_results = []
    for level in levels:
        result = profile_level(
            model=model,
            level=level,
            n_frames=args.n_frames,
            H_orig=H_orig,
            W_orig=W_orig,
            pixel_batch=pixel_batch,
            device_str=device_str,
            n_warmup=args.n_warmup,
            n_runs=args.n_runs,
        )
        if result is not None:
            level_results.append(result)

    if len(level_results) > 1:
        saab_per_px  = sum(r["saab_flops_total"] for r in level_results) / n_orig_pixels
        lnt_per_px   = sum(r["lnt_flops_total"]  for r in level_results) / n_orig_pixels
        xgb_per_px   = sum(r["xgb_cmp_total"]    for r in level_results) / n_orig_pixels
        total_per_px = saab_per_px + lnt_per_px
        grand_per_px = total_per_px + xgb_per_px

        print(f"\n{'━'*68}")
        print(f"  Combined FLOPs/pixel (all {len(level_results)} levels, {n_orig_pixels:,} original pixels)")
        print(f"{'━'*68}")
        w2 = 14
        print(f"  {'Stage':<{w2}} {'FLOPs/pixel':>14}")
        print(f"  {'-'*w2} {'-'*14}")
        print(f"  {'Saab':<{w2}} {_fmt_flops(saab_per_px):>14}")
        print(f"  {'LNT':<{w2}} {_fmt_flops(lnt_per_px):>14}")
        print(f"  {'-'*w2} {'-'*14}")
        print(f"  {'Total FLOPs':<{w2}} {_fmt_flops(total_per_px):>14}")
        print(f"  {'XGBoost (cmp)':<{w2}} {_fmt_flops(xgb_per_px):>14}  ← comparisons, not FLOPs")
        print(f"  {'─'*w2} {'─'*14}")
        print(f"  {'Grand Total':<{w2}} {_fmt_flops(grand_per_px):>14}  ← FLOPs + comparisons")

    print(f"\n{'━'*68}")
    print("  Done.")
    print(f"{'━'*68}\n")


if __name__ == "__main__":
    main()
