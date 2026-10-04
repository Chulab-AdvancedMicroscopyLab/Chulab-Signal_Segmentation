#!/usr/bin/env python3
"""
Measure the energy a command uses (e.g. train.py / inference.py).

    python measure_energy.py --gpus 0 --log energy.jsonl -- python inference.py --config configs/config_cfos.json

GPU: NVML cumulative energy counter per selected GPU; idle power (measured before the run)
     x duration is subtracted, so other idle draw does not count. Other jobs on the same GPU do.
CPU: Intel RAPL package counters if readable (usually needs root); otherwise an ESTIMATE from the
     command's CPU time x watts per core.
"""
import argparse
import glob
import json
import resource
import subprocess
import sys
import time

import pynvml


def gpu_energy_j(handles):
    return [pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0 for h in handles]  # mJ -> J


def rapl_domains():
    """Package-level RAPL counters, or [] if not readable."""
    doms = []
    for d in sorted(glob.glob("/sys/class/powercap/intel-rapl:[0-9]")):
        try:
            doms.append((d, int(open(f"{d}/max_energy_range_uj").read())))
            open(f"{d}/energy_uj").read()
        except OSError:
            return []
    return doms


def rapl_uj(doms):
    return [int(open(f"{d}/energy_uj").read()) for d, _ in doms]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", default="0", help="comma-separated GPU indices to meter")
    p.add_argument("--idle", type=float, default=10.0, help="seconds of idle baseline before the run")
    p.add_argument("--watts-per-core", type=float, default=185.0 / 16,
                   help="CPU estimate when RAPL is unreadable (default: Xeon Gold 6326, 185 W TDP / 16 cores)")
    p.add_argument("--grid", type=float, default=0.494, help="kg CO2e per kWh (default: Taiwan grid)")
    p.add_argument("--label", default="", help="free-text label stored in the log")
    p.add_argument("--log", default=None, help="append a JSON line with the results")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="command to run, after --")
    a = p.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        p.error("no command given (put it after --)")

    pynvml.nvmlInit()
    gpu_ids = [int(g) for g in a.gpus.split(",") if g != ""]
    handles = [pynvml.nvmlDeviceGetHandleByIndex(g) for g in gpu_ids]
    doms = rapl_domains()

    # idle baseline
    e0, t0 = gpu_energy_j(handles), time.time()
    time.sleep(a.idle)
    idle_w = [(e - s) / (time.time() - t0) for s, e in zip(e0, gpu_energy_j(handles))]

    # run
    g0, c0 = gpu_energy_j(handles), (rapl_uj(doms) if doms else None)
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t_start = time.time()
    rc = subprocess.run(cmd).returncode
    wall = time.time() - t_start
    g1 = gpu_energy_j(handles)
    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)

    gpu_raw = [e - s for s, e in zip(g0, g1)]
    gpu_net = [max(r - w * wall, 0.0) for r, w in zip(gpu_raw, idle_w)]
    cpu_s = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    if doms:
        c1 = rapl_uj(doms)
        cpu_j = sum(((e - s) % rng) / 1e6 for (s, e, (_, rng)) in zip(c0, c1, doms))
        cpu_method = "rapl"
    else:
        cpu_j = cpu_s * a.watts_per_core
        cpu_method = f"estimate ({a.watts_per_core:.1f} W/core x CPU time)"

    total_j = sum(gpu_net) + cpu_j
    res = {
        "label": a.label, "cmd": " ".join(cmd), "returncode": rc, "wall_s": round(wall, 1),
        "gpus": gpu_ids, "gpu_idle_w": [round(w, 1) for w in idle_w],
        "gpu_raw_j": [round(x) for x in gpu_raw], "gpu_net_j": [round(x) for x in gpu_net],
        "cpu_time_s": round(cpu_s, 1), "cpu_j": round(cpu_j), "cpu_method": cpu_method,
        "total_j": round(total_j), "total_kwh": total_j / 3.6e6, "co2_g": total_j / 3.6e6 * a.grid * 1000,
    }
    print(f"\n[energy] {a.label or cmd[0]}: wall {wall:.0f}s | GPU net {sum(gpu_net)/1e3:.2f} kJ "
          f"(idle {sum(idle_w):.0f} W subtracted) | CPU {cpu_j/1e3:.2f} kJ [{cpu_method}] | "
          f"total {total_j/3.6e6*1e3:.3f} Wh, {res['co2_g']:.2f} g CO2e", file=sys.stderr)
    if a.log:
        with open(a.log, "a") as f:
            f.write(json.dumps(res) + "\n")
    sys.exit(rc)


if __name__ == "__main__":
    main()
