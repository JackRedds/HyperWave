"""Run ``wavelet_reconstruction.py`` at a set of injected parameter values.

Each ``--sweep KEY=VALUES`` gives the values of one injected parameter; the run
grid is their Cartesian product (or, with ``--zip``, the values taken pairwise).
``VALUES`` is either a comma-separated list or a range:

    luminosity_distance=200,400,800      explicit list
    luminosity_distance=lin:200:800:4    4 values, linearly spaced
    hrss=log:1e-22:1e-21:5               5 values, log spaced

The key ``snr`` is special: it sets the injected network optimal SNR
(``--inj-snr``, rescaling each family's amplitude parameter), so the same SNR
sweep works for every waveform. ``--waveform`` takes several families (or
``all``) and repeats the whole grid for each of them.

``--set KEY=VALUE`` holds a parameter fixed (away from the script default) for
every run; prefix it with a family (``--set sg:quality=10``) to apply it to that
family only. Every other option is passed straight through to
``wavelet_reconstruction.py`` (``--device``, ``--proposal``, ``--nsteps``, ...).

Run ``i`` writes ``<outdir>/<waveform>/run_XXXX.npz`` (+ ``run_XXXX.log``, and
plots with ``--recon-plots``); ``i`` (used by ``--indices``) counts across all
waveforms, ``XXXX`` within one. Finished runs are skipped on re-runs
(``--no-resume`` to redo them). After the runs, ``<outdir>/sweep_summary.csv``
tabulates waveform and swept values against the injected SNR, overlap and
recovered SNR.

Run::

    # list the grid without running anything
    python scripts/wavelet_param_sweep.py --waveform bbh \\
        --sweep luminosity_distance=300,600,1200 --sweep mass_1=30,50 --dry-run
    # run it, passing sampler options through
    python scripts/wavelet_param_sweep.py --waveform bbh \\
        --sweep luminosity_distance=300,600,1200 --sweep mass_1=30,50 \\
        --outdir results/bbh_sweep --device gpu --proposal flowfisher --nsteps 8000
    # every waveform at the same set of network SNRs (5 families x 4 SNRs = 20 runs)
    python scripts/wavelet_param_sweep.py --waveform all --sweep snr=8,15,30,60 \\
        --outdir results/snr_sweep --device gpu --proposal flowfisher
    # one run per SLURM array task (--array=0-5), then collect the summary
    python scripts/wavelet_param_sweep.py ... --indices $SLURM_ARRAY_TASK_ID
    python scripts/wavelet_param_sweep.py ... --summarize-only
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import sys

import numpy as np

RECON_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "wavelet_reconstruction.py")
# kept in sync with wavelet_reconstruction.WAVEFORMS (not imported: it pulls in
# eryn/torch, and --dry-run should stay instant)
FAMILIES = ["bbh", "sg", "cs", "gaussian", "wnb"]
SNR_KEY = "snr"


def parse_values(spec):
    """``"a,b,c"`` | ``"lin:lo:hi:n"`` | ``"log:lo:hi:n"`` -> list of floats."""
    if spec.startswith(("lin:", "log:")):
        kind, lo, hi, n = spec.split(":")
        space = np.linspace if kind == "lin" else np.geomspace
        return [float(v) for v in space(float(lo), float(hi), int(n))]
    return [float(v) for v in spec.split(",") if v.strip()]


def parse_sweeps(items):
    sweeps = {}
    for item in items:
        key, sep, spec = item.partition("=")
        if not sep or not spec:
            raise SystemExit(f"--sweep expects KEY=VALUES, got {item!r}")
        sweeps[key.strip()] = parse_values(spec)
    return sweeps


def build_grid(sweeps, zipped):
    """List of ``{key: value}`` points (one empty point when nothing is swept)."""
    keys = list(sweeps)
    if not keys:
        return [{}]
    if zipped:
        lengths = {len(v) for v in sweeps.values()}
        if len(lengths) != 1:
            raise SystemExit(f"--zip needs equal-length value lists, got "
                             f"{ {k: len(v) for k, v in sweeps.items()} }")
        combos = zip(*sweeps.values())
    else:
        combos = itertools.product(*sweeps.values())
    return [dict(zip(keys, c)) for c in combos]


def parse_waveforms(names):
    out = []
    for name in names:
        for w in (FAMILIES if name == "all" else name.split(",")):
            if w not in FAMILIES:
                raise SystemExit(f"unknown waveform {w!r}; choose from {FAMILIES} or 'all'")
            if w not in out:
                out.append(w)
    return out


def set_for(waveform, items):
    """``--set`` entries for ``waveform``: unprefixed ones plus ``waveform:KEY=VALUE``."""
    out = []
    for item in items:
        fam, sep, kv = item.partition(":")
        if not sep or "=" in fam:          # no family prefix
            out.append(item)
        elif fam not in FAMILIES:
            raise SystemExit(f"--set {item!r}: unknown family {fam!r}")
        elif fam == waveform:
            out.append(kv)
    return out


def run_paths(outdir, waveform, j):
    base = os.path.join(outdir, waveform, f"run_{j:04d}")
    return base + ".npz", base + ".log", base


def summarize(outdir, runs):
    rows = []
    for i, (waveform, j, point) in enumerate(runs):
        npz, _, _ = run_paths(outdir, waveform, j)
        if not os.path.exists(npz):
            continue
        r = np.load(npz, allow_pickle=False)
        ov, rs = r["overlap"], r["rec_snr"]
        rows.append(dict(
            index=i, waveform=waveform, run=j, **point,
            inj_snr=float(r["inj_snr"]),
            overlap_median=float(np.median(ov)),
            overlap_5=float(np.percentile(ov, 5)),
            overlap_95=float(np.percentile(ov, 95)),
            rec_snr_median=float(np.median(rs)),
            nleaves_median=float(np.median(r["nleaves"])),
            converged=bool(r["converged"]),
            sampled_steps=int(r["sampled_steps"]),
            sample_minutes=float(r["sample_seconds"]) / 60.0,
        ))
    if not rows:
        print("[summary] no finished runs yet")
        return
    path = os.path.join(outdir, "sweep_summary.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    keys = list(runs[0][2])
    print(f"\n[summary] {len(rows)}/{len(runs)} runs finished -> {path}")
    print(f"{'waveform':>8s}  " + "  ".join(f"{k:>20s}" for k in keys)
          + f"  {'inj_SNR':>8s}  {'overlap [5,50,95]':>22s}  {'rec_SNR':>8s}  conv")
    for r in rows:
        print(f"{r['waveform']:>8s}  " + "  ".join(f"{r[k]:>20.6g}" for k in keys)
              + f"  {r['inj_snr']:8.1f}  [{r['overlap_5']:.3f}, {r['overlap_median']:.3f}, "
                f"{r['overlap_95']:.3f}]  {r['rec_snr_median']:8.1f}  {'y' if r['converged'] else 'n'}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False,
        epilog="Unrecognised options are forwarded to wavelet_reconstruction.py.")
    p.add_argument("--waveform", nargs="+", required=True,
                   help=f"injected families to run ({', '.join(FAMILIES)}, or 'all'); "
                        "the grid is repeated for each")
    p.add_argument("--sweep", action="append", default=[], metavar="KEY=VALUES",
                   help="parameter to vary and its values (repeatable); KEY 'snr' "
                        "sweeps the injected network SNR for any waveform")
    p.add_argument("--set", action="append", default=[], metavar="[FAMILY:]KEY=VALUE",
                   help="parameter held at VALUE for every run, or only for FAMILY's "
                        "runs when prefixed (repeatable)")
    p.add_argument("--zip", action="store_true",
                   help="pair the --sweep lists element-wise instead of taking the product")
    p.add_argument("--outdir", default="results/param_sweep")
    p.add_argument("--indices", type=int, nargs="+", default=None,
                   help="only run these grid indices (e.g. $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--recon-plots", action="store_true",
                   help="save per-run reconstruction plots (run_XXXX_<IFO>_fd.png, _td.png)")
    p.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                   help="re-run grid points whose run_XXXX.npz already exists")
    p.add_argument("--keep-going", action="store_true",
                   help="continue with the next run when one fails")
    p.add_argument("--dry-run", action="store_true",
                   help="print the grid and commands without running")
    p.add_argument("--summarize-only", action="store_true",
                   help="only (re)build sweep_summary.csv from finished runs")
    args, passthrough = p.parse_known_args()

    for flag in ("--waveform", "--inj-param", "--inj-snr", "--outfile", "--recon-plot"):
        if any(a == flag or a.startswith(flag + "=") for a in passthrough):
            p.error(f"{flag} is set by the sweep; don't pass it through")

    waveforms = parse_waveforms(args.waveform)
    sweeps = parse_sweeps(args.sweep)
    grid = build_grid(sweeps, args.zip)
    # global run list: the whole grid for each waveform in turn
    runs = [(w, j, point) for w in waveforms for j, point in enumerate(grid)]
    for w in waveforms:
        os.makedirs(os.path.join(args.outdir, w), exist_ok=True)

    if args.summarize_only:
        summarize(args.outdir, runs)
        return

    indices = args.indices if args.indices is not None else range(len(runs))
    bad = [i for i in indices if not 0 <= i < len(runs)]
    if bad:
        p.error(f"indices {bad} outside the run list (0..{len(runs) - 1})")

    swept = (f"{'zipped' if args.zip else 'product'} of {', '.join(sweeps)}"
             if sweeps else "default injection")
    print(f"[sweep] {', '.join(waveforms)} x {len(grid)} grid points ({swept}) = "
          f"{len(runs)} runs; running {len(indices)} -> {args.outdir}")
    failed = []
    for i in indices:
        waveform, j, point = runs[i]
        npz, log, base = run_paths(args.outdir, waveform, j)
        label = ", ".join([waveform] + [f"{k}={v:.6g}" for k, v in point.items()])
        if args.resume and os.path.exists(npz):
            print(f"[{i:04d}] {label}: done already, skipping")
            continue

        cmd = [sys.executable, RECON_SCRIPT, "--waveform", waveform]
        for kv in set_for(waveform, args.set):
            cmd += ["--inj-param", kv]
        for k, v in point.items():
            if k == SNR_KEY:
                cmd += ["--inj-snr", repr(v)]
            else:
                cmd += ["--inj-param", f"{k}={v!r}"]
        cmd += ["--outfile", npz]
        if args.recon_plots:
            cmd += ["--recon-plot", base + ".png"]
        cmd += passthrough

        print(f"[{i:04d}] {label}")
        if args.dry_run:
            print("       " + " ".join(cmd))
            continue
        # tee the child's output to run_XXXX.log and the terminal
        with open(log, "w") as fh:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                fh.write(line)
            code = proc.wait()
        if code != 0:
            print(f"[{i:04d}] FAILED (exit {code}), see {log}")
            failed.append(i)
            # exit code 2 = argument error (e.g. unknown parameter): this waveform's
            # other runs would fail too, but another family's may still be fine
            if not args.keep_going:
                break

    if not args.dry_run:
        summarize(args.outdir, runs)
    if failed:
        raise SystemExit(f"failed runs: {failed}")


if __name__ == "__main__":
    main()
