"""
profile_model.py — FLOPs, timing, and memory profiler for Chulab-Signal_Segmentation models.

Profiles all four models (UNet, AttentionUNet, SwinUNETR, VNet) or a single selected model
using random synthetic tensors. No real dataset required.

Usage:
  python profile_model.py                                    # all models, 3D defaults
  python profile_model.py --model unet                       # single model
  python profile_model.py --gpu 1                            # specific GPU
  python profile_model.py --spatial_dims 2 --patch 1 64 64  # 2D mode
  python profile_model.py --batch_size 4 --n_runs 20        # larger batch / more runs
  python profile_model.py --device cpu                       # CPU profiling
"""

import argparse
import time
import gc
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning, module="cuda")

import torch
import torch.nn as nn

from models.factory import build_model_from_config


# ── formatting helpers ────────────────────────────────────────────────────────

def _fmt_time(s):
    if s < 1e-3:
        return f"{s*1e6:.1f} µs"
    if s < 1:
        return f"{s*1e3:.2f} ms"
    return f"{s:.3f} s"

def _fmt_mem(b):
    if b >= 1 << 30:
        return f"{b / (1<<30):.2f} GB"
    if b >= 1 << 20:
        return f"{b / (1<<20):.1f} MB"
    return f"{b / (1<<10):.1f} KB"

def _fmt_flops(f):
    if f <= 0:
        return "—"
    if f >= 1e12:
        return f"{f/1e12:.2f} TFLOPs"
    if f >= 1e9:
        return f"{f/1e9:.2f} GFLOPs"
    if f >= 1e6:
        return f"{f/1e6:.2f} MFLOPs"
    return f"{f:.0f} FLOPs"

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def _param_count(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ── FLOP formulas per module type ─────────────────────────────────────────────

def _flops_conv(m, inp, out):
    k = m.kernel_size if isinstance(m.kernel_size, (list, tuple)) else [m.kernel_size]
    kernel_ops = (m.in_channels // m.groups) * int(torch.prod(torch.tensor(k, dtype=torch.float)).item())
    out_elems  = out.numel() // out.shape[1]
    return 2 * kernel_ops * out.shape[1] * out_elems

def _flops_convtranspose(m, inp, out):
    k = m.kernel_size if isinstance(m.kernel_size, (list, tuple)) else [m.kernel_size]
    kernel_ops = (m.in_channels // m.groups) * int(torch.prod(torch.tensor(k, dtype=torch.float)).item())
    out_elems  = out.numel() // out.shape[1]
    return 2 * kernel_ops * out.shape[1] * out_elems

def _flops_linear(m, inp, out):
    return 2 * m.in_features * out.numel()

def _flops_norm(m, inp, out):
    return 4 * inp[0].numel()      # normalize (2) + affine (2)

def _flops_activation(m, inp, out):
    return inp[0].numel()          # ~1 op per element

def _flops_gelu(m, inp, out):
    return 8 * inp[0].numel()      # tanh approx: ~8 ops

def _flops_attention(m, inp, out):
    x = inp[0]
    if x.dim() == 3:
        if m.batch_first:
            b, seq, d = x.shape
        else:
            seq, b, d = x.shape
    else:
        seq, b, d = x.shape[0], 1, x.shape[-1]
    return b * (4 * seq * seq * d + 8 * seq * d * d)

# Map module class → (flop_fn or None)
# None means we measure time but cannot count FLOPs for that type.
_MODULE_REGISTRY = {
    # Convolutions
    nn.Conv1d:              _flops_conv,
    nn.Conv2d:              _flops_conv,
    nn.Conv3d:              _flops_conv,
    nn.ConvTranspose1d:     _flops_convtranspose,
    nn.ConvTranspose2d:     _flops_convtranspose,
    nn.ConvTranspose3d:     _flops_convtranspose,
    # Linear / Attention
    nn.Linear:              _flops_linear,
    nn.MultiheadAttention:  _flops_attention,
    # Normalisation
    nn.BatchNorm1d:         _flops_norm,
    nn.BatchNorm2d:         _flops_norm,
    nn.BatchNorm3d:         _flops_norm,
    nn.InstanceNorm1d:      _flops_norm,
    nn.InstanceNorm2d:      _flops_norm,
    nn.InstanceNorm3d:      _flops_norm,
    nn.LayerNorm:           _flops_norm,
    nn.GroupNorm:           _flops_norm,
    # Activations
    nn.ReLU:                _flops_activation,
    nn.ReLU6:               _flops_activation,
    nn.LeakyReLU:           _flops_activation,
    nn.PReLU:               _flops_activation,
    nn.ELU:                 _flops_activation,
    nn.GELU:                _flops_gelu,
    nn.SiLU:                _flops_activation,
    nn.Sigmoid:             _flops_activation,
    nn.Tanh:                _flops_activation,
    nn.Softmax:             _flops_activation,
    # Dropout (0 FLOPs, but shows up in timing)
    nn.Dropout:             None,
    nn.Dropout2d:           None,
    nn.Dropout3d:           None,
}


# ── unified module profiler (timing + FLOPs via CUDA events) ─────────────────

def profile_modules(model, x, device):
    """
    Hook every module in _MODULE_REGISTRY, run one forward pass, and return
    per-type stats: {class_name: {count, time_s, flops}}.

    Timing uses torch.cuda.Event (GPU) or time.perf_counter (CPU) so only the
    self-time of each op is measured without double-counting parent containers.
    """
    stats   = defaultdict(lambda: {"count": 0, "time_s": 0.0, "flops": 0})
    handles = []
    pending = []   # (cls_name, start, end, flops) collected during forward

    use_cuda = device.type == "cuda"

    def make_hooks(cls_name, flop_fn):
        if use_cuda:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev   = torch.cuda.Event(enable_timing=True)

            def pre(m, inp):
                start_ev.record(torch.cuda.current_stream(device))

            def post(m, inp, out):
                end_ev.record(torch.cuda.current_stream(device))
                f = flop_fn(m, inp, out) if flop_fn is not None else 0
                pending.append((cls_name, start_ev, end_ev, f))
        else:
            buf = {}

            def pre(m, inp):
                buf["t"] = time.perf_counter()

            def post(m, inp, out):
                elapsed = time.perf_counter() - buf["t"]
                f = flop_fn(m, inp, out) if flop_fn is not None else 0
                pending.append((cls_name, elapsed, None, f))

        return pre, post

    for mod in model.modules():
        cls = type(mod)
        if cls not in _MODULE_REGISTRY:
            continue
        flop_fn  = _MODULE_REGISTRY[cls]
        pre, post = make_hooks(cls.__name__, flop_fn)
        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))

    model.eval()
    with torch.no_grad():
        _ = model(x)
    _sync(device)

    for h in handles:
        h.remove()

    # Collect results after sync (CUDA events only readable after sync)
    for entry in pending:
        cls_name, start, end, flops = entry
        if use_cuda:
            elapsed_s = start.elapsed_time(end) * 1e-3   # ms → s
        else:
            elapsed_s = start   # already seconds
        stats[cls_name]["count"]  += 1
        stats[cls_name]["time_s"] += elapsed_s
        stats[cls_name]["flops"]  += flops

    return dict(stats)


# ── overall forward / forward+backward timing ────────────────────────────────

def measure_forward(model, x, device, n_warmup, n_runs):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x); _sync(device)
        times = []
        for _ in range(n_runs):
            _sync(device)
            t0 = time.perf_counter()
            _ = model(x)
            _sync(device)
            times.append(time.perf_counter() - t0)
    t = torch.tensor(times)
    return float(t.mean()), float(t.std())


def measure_forward_backward(model, x, device, n_warmup, n_runs):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    target = torch.zeros_like(model(x).detach())

    for _ in range(n_warmup):
        model.zero_grad(set_to_none=True)
        loss = criterion(model(x), target); loss.backward()
        _sync(device)

    times = []
    for _ in range(n_runs):
        model.zero_grad(set_to_none=True)
        _sync(device)
        t0 = time.perf_counter()
        loss = criterion(model(x), target); loss.backward()
        _sync(device)
        times.append(time.perf_counter() - t0)

    t = torch.tensor(times)
    return float(t.mean()), float(t.std())


# ── model config factory ──────────────────────────────────────────────────────

def make_config(model_type, spatial_dims, in_channels, out_channels, patch):
    base = {
        "train": {"model_type": model_type},
        "model": {
            model_type: {
                "spatial_dims": spatial_dims,
                "in_channels":  in_channels,
                "out_channels": out_channels,
            }
        },
    }
    if model_type == "swin_unetr":
        base["model"][model_type]["feature_size"] = 48
    return base


# ── per-model profiling ───────────────────────────────────────────────────────

def profile_one(model_type, args, device):
    print(f"\n{'═'*68}")
    print(f"  Model: {model_type.upper()}")
    print(f"{'═'*68}")

    cfg = make_config(model_type, args.spatial_dims, args.in_channels, args.out_channels, args.patch)
    try:
        model = build_model_from_config(cfg).to(device)
    except Exception as e:
        print(f"  [SKIP] Could not build {model_type}: {e}")
        return

    total_params, trainable_params = _param_count(model)
    print(f"  Parameters : {total_params:,} total  ({trainable_params:,} trainable)")

    # SwinUNETR needs spatial dims divisible by 32
    spatial = list(args.patch if args.spatial_dims == 3 else args.patch[-2:])
    if model_type == "swin_unetr":
        spatial = [((s + 31) // 32) * 32 for s in spatial]
    shape = (args.batch_size, args.in_channels, *spatial)
    x = torch.randn(*shape, device=device)
    print(f"  Input shape: {list(shape)}  dtype=float32  device={device}")

    # ── per-module breakdown (timing + FLOPs) ──
    print(f"\n  Per-module breakdown (forward pass)...")
    try:
        mod_stats = profile_modules(model, x, device)

        total_flops = sum(v["flops"]  for v in mod_stats.values())
        total_time  = sum(v["time_s"] for v in mod_stats.values())

        print(f"  Total FLOPs : {_fmt_flops(total_flops)}"
              + (f"  ({_fmt_flops(total_flops / args.batch_size)} / sample)" if args.batch_size > 1 else ""))

        # Sort by time descending
        rows = sorted(mod_stats.items(), key=lambda kv: -kv[1]["time_s"])

        w = 24
        print(f"\n  {'Module':<{w}} {'Count':>5}  {'Time':>10}  {'% time':>7}  {'FLOPs':>14}  {'% FLOPs':>8}")
        print(f"  {'-'*w} {'-'*5}  {'-'*10}  {'-'*7}  {'-'*14}  {'-'*8}")
        for cls_name, v in rows:
            t_pct = 100 * v["time_s"] / total_time  if total_time  > 0 else 0
            f_pct = 100 * v["flops"]  / total_flops if total_flops > 0 else 0
            f_str = _fmt_flops(v["flops"]) if v["flops"] > 0 else "—"
            fp_str = f"{f_pct:>7.1f}%"      if v["flops"] > 0 else "       —"
            print(f"  {cls_name:<{w}} {v['count']:>5}  "
                  f"{_fmt_time(v['time_s']):>10}  {t_pct:>6.1f}%  "
                  f"{f_str:>14}  {fp_str}")
        print(f"  {'─'*w} {'─'*5}  {'─'*10}  {'─'*7}  {'─'*14}  {'─'*8}")
        print(f"  {'TOTAL':<{w}} {'':>5}  {_fmt_time(total_time):>10}  {'100.0%':>7}  "
              f"{_fmt_flops(total_flops):>14}  {'100.0%':>8}")

    except Exception as e:
        print(f"  [WARN] Module profiling failed: {e}")
        raise

    # ── overall forward timing ──
    print(f"\n  Forward pass  (warmup={args.n_warmup}, runs={args.n_runs})...")
    try:
        fwd_mean, fwd_std = measure_forward(model, x, device, args.n_warmup, args.n_runs)
        print(f"  Forward  : {_fmt_time(fwd_mean)} ± {_fmt_time(fwd_std)}"
              f"   ({args.batch_size / fwd_mean:.1f} samples/sec)")
    except Exception as e:
        print(f"  [WARN] Forward timing failed: {e}")
        fwd_mean = None

    # ── forward + backward ──
    print(f"\n  Forward + backward timing...")
    try:
        bwd_mean, bwd_std = measure_forward_backward(model, x, device, args.n_warmup, args.n_runs)
        print(f"  Fwd+Bwd  : {_fmt_time(bwd_mean)} ± {_fmt_time(bwd_std)}")
        if fwd_mean:
            print(f"  Backward overhead: {_fmt_time(bwd_mean - fwd_mean)}")
    except Exception as e:
        print(f"  [WARN] Backward timing failed: {e}")

    # ── GPU memory ──
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(x)
        _sync(device)
        peak     = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        print(f"\n  GPU memory (forward, peak allocated): {_fmt_mem(peak)}")
        print(f"  GPU memory (reserved by allocator)  : {_fmt_mem(reserved)}")

    del model, x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Chulab model profiler (synthetic data)")
    p.add_argument("--model", type=str, default="all",
                   choices=["all", "unet", "attention_unet", "swin_unetr", "vnet"])
    p.add_argument("--spatial_dims", type=int, default=3, choices=[2, 3])
    p.add_argument("--patch",        type=int, nargs="+", default=[16, 64, 64])
    p.add_argument("--in_channels",  type=int, default=1)
    p.add_argument("--out_channels", type=int, default=1)
    p.add_argument("--batch_size",   type=int, default=2)
    p.add_argument("--n_warmup",     type=int, default=3)
    p.add_argument("--n_runs",       type=int, default=10)
    p.add_argument("--device",       type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gpu",          type=int, default=None,
                   help="GPU index (e.g. 0, 1). Overrides --device.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else args.device)

    models_to_run = (
        ["unet", "attention_unet", "swin_unetr", "vnet"]
        if args.model == "all" else [args.model]
    )

    spatial_str = "×".join(str(s) for s in args.patch)
    print(f"\n{'━'*68}")
    print(f"  Chulab-Signal_Segmentation Model Profiler")
    print(f"{'━'*68}")
    print(f"  Device      : {device}"
          + (f"  ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"  Spatial dims: {args.spatial_dims}D  |  patch: {spatial_str}")
    print(f"  Batch size  : {args.batch_size}  |  in_ch: {args.in_channels}  out_ch: {args.out_channels}")
    print(f"  Timing runs : {args.n_warmup} warmup + {args.n_runs} timed")
    print(f"  Models      : {', '.join(models_to_run)}")

    for model_type in models_to_run:
        profile_one(model_type, args, device)

    print(f"\n{'━'*68}")
    print("  Done.")
    print(f"{'━'*68}\n")


if __name__ == "__main__":
    main()
