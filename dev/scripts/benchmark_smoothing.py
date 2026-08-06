"""Micro-benchmark for the DeepSphere Gaussian-beam smoothing front-end across spmm backends.

Analogous to ``benchmark_spmm.py`` (which benchmarks the graph-conv L @ x) but for
``deepsphere.healpy_layers.HealpySmoothing``. Smoothing applies its sparse Gaussian kernel
per channel, with per-channel repetitions, and is TODAY pinned to eager fp32 and carved OUT of
the XLA-jitted network body -- *solely* because ``tf.sparse.sparse_dense_matmul`` is not
XLA-compatible. The "gather" backend reformulates the kernel application as pure dense ops
(padded neighbour gather + weighted reduce), which removes that limitation, so the entire
smoothing ``call`` can be XLA-compiled the way the transformer body is.

This script measures, on a realistic DES-Y3-sized footprint with the production per-channel
``theta_fwhm`` list, whether compiling the smoothing is actually worth it:

  (a) coo         -- current path (tf.sparse.sparse_dense_matmul), GRAPH mode (tf.function).
                     This is the baseline the network runs today.
  (b) csr         -- cuSPARSE csrmm, graph mode. Same fp32 math, no 2**31 split.
  (c) gather      -- gather + weighted reduce, graph mode (no jit).
  (d) gather+xla  -- the same gather call under @tf.function(jit_compile=True): the "compile the
                     smoothing too" case that only becomes possible with a dense-op backend.

For reference it also *attempts* coo+xla / csr+xla to demonstrate that they fail to compile
(the reason smoothing is carved out in the first place).

It reuses the real HealpySmoothing layer, so the kernel construction (BallTree neighbour search),
per-channel repetitions and mask multiply are the true production code paths. White noise is
disabled: it is a backend-independent elementwise add of tf.random.normal, and leaving it on
would make each call's output differ by RNG and destroy the max-abs-diff correctness gate.

For each backend it reports steady-state ms per smoothing call, peak GPU memory (the "does it fit"
criterion), and the max-abs difference vs the coo baseline (the correctness gate).

Run on a GPU compute node, e.g.:
    srun -A a0158 --environment=tensorflow --partition=normal --pty bash
    python dev/scripts/benchmark_smoothing.py --nside 512 --probe lensing --batch 20

Stage --out and --data_path on /iopsstor scratch (login-node paths are invisible to the node).
"""

import argparse
import time

import numpy as np
import healpy as hp
import tensorflow as tf

from deepsphere import healpy_layers as hp_nn

# Production per-channel beams (configs/scales/8wl,32gc.yaml), arcmin. base = the finest kernel;
# larger scales are reached by repeated application (per_channel_repetitions inside the layer).
FWHM = {
    "lensing": dict(fwhm=[29.5, 20.1, 15.0, 12.6], fwhm_base=None),
    "clustering": dict(fwhm=[130.2, 88.9, 68.1, 57.0], fwhm_base=40.4),
}


def des_y3_footprint(nside, fsky):
    """A contiguous DES-Y3-sized footprint: nside pixels inside a polar cap of sky fraction fsky.

    The exact footprint shape is immaterial to the kernel timing (the beam support is set by the
    pixel scale, not the outline); what matters is the pixel count (~fsky * npix) and that the
    BallTree / kernel is built on real angular positions with a real boundary, as in production.
    """
    theta_cap = np.arccos(1.0 - 2.0 * fsky)  # cap half-angle giving 2*pi*(1-cos) = 4*pi*fsky
    vec = hp.ang2vec(0.0, 0.0)  # cap centred on the north pole
    indices = hp.query_disc(nside, vec, radius=theta_cap, nest=True)
    return np.sort(indices).astype(np.int64)


def build_layer(nside, indices, probe, n_sigma_support, backend, data_path):
    beams = FWHM[probe]
    layer = hp_nn.HealpySmoothing(
        nside=nside,
        indices=indices,
        nest=True,
        mask=np.ones((len(indices), 1), dtype=np.float32),  # exercise the mask multiply
        fwhm=beams["fwhm"],
        fwhm_base=beams["fwhm_base"],
        n_sigma_support=n_sigma_support,
        arcmin=True,
        white_noise_sigma=None,  # off: backend-independent RNG add, would break the maxdiff gate
        data_path=data_path,  # cache the kernel so every backend reuses one BallTree build
        max_batch_size=None,  # csr/gather ignore splits; coo infers from the (static) batch below
        spmm_backend=backend,
    )
    return layer


def time_call(run, x, n_warmup, n_iter):
    for _ in range(n_warmup):
        out = run(x)
    _ = out.numpy()  # force device sync after warmup
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = run(x)
    _ = out.numpy()  # single sync after the batch of iters
    dt = (time.perf_counter() - t0) / n_iter
    return dt * 1e3, out.numpy()


def peak_mb():
    try:
        return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e6
    except Exception:
        return float("nan")


def reset_peak():
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nside", type=int, default=512, help="512 = full-res lensing single-kernel path")
    ap.add_argument("--probe", choices=list(FWHM), default="lensing")
    ap.add_argument("--fsky", type=float, default=0.10, help="DES Y3 ~4143 deg^2 -> f_sky ~= 0.10")
    ap.add_argument("--batch", type=int, default=20, help="RHS width per channel = local_batch_size")
    ap.add_argument("--n_sigma_support", type=float, default=3.0)
    ap.add_argument("--n_warmup", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=30)
    ap.add_argument("--data_path", type=str, default=None, help="kernel cache dir (stage on /iopsstor)")
    ap.add_argument("--out", type=str, default=None, help="optional CSV path (stage on /iopsstor)")
    args = ap.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    print(f"# TF {tf.__version__} | GPUs: {[g.name for g in gpus] or 'NONE (CPU-only)'}")

    indices = des_y3_footprint(args.nside, args.fsky)
    n_ch = len(FWHM[args.probe]["fwhm"])
    rng = np.random.default_rng(0)
    x = tf.constant(rng.standard_normal((args.batch, len(indices), n_ch)).astype(np.float32))
    print(
        f"# probe={args.probe} nside={args.nside} n_pix(footprint)={len(indices)} "
        f"(f_sky~{len(indices)/hp.nside2npix(args.nside):.3f}) channels={n_ch} batch={args.batch}"
    )
    print(f"# warmup={args.n_warmup} iter={args.n_iter}\n")

    # (backend, jit) pairs. coo/csr are graph-mode (their sparse ops are not XLA-compatible);
    # gather is run both graph-mode and jit-compiled. coo+xla / csr+xla are attempted only to show
    # they fail -- that failure is exactly why smoothing is eager today.
    plan = [("coo", False), ("csr", False), ("gather", False), ("gather", True), ("coo", True), ("csr", True)]

    header = f"{'backend':>12} {'ms/call':>10} {'peak_MB':>10} {'maxdiff_vs_coo':>16}"
    print(header)
    print("-" * len(header))

    ref = None
    rows = []
    for backend, jit in plan:
        label = f"{backend}+xla" if jit else backend
        try:
            layer = build_layer(args.nside, indices, args.probe, args.n_sigma_support, backend, args.data_path)
            layer(x)  # build (kernel + operator) eagerly so jit wraps an already-built layer
            run = tf.function(lambda t: layer(t), jit_compile=jit)
            reset_peak()
            ms, out = time_call(run, x, args.n_warmup, args.n_iter)
        except Exception as exc:
            print(f"{label:>12} {'FAILED':>10} {'':>10}   {type(exc).__name__}: {str(exc).splitlines()[0][:80]}")
            rows.append((label, float("nan"), float("nan"), float("nan")))
            continue
        if label == "coo":
            ref = out
        maxdiff = float(np.max(np.abs(out - ref))) if ref is not None else float("nan")
        pk = peak_mb()
        print(f"{label:>12} {ms:>10.3f} {pk:>10.1f} {maxdiff:>16.3e}")
        rows.append((label, ms, pk, maxdiff))

    if args.out:
        import csv

        with open(args.out, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["backend", "ms_per_call", "peak_MB", "maxdiff_vs_coo"])
            wr.writerows(rows)
        print(f"\n# wrote {args.out}")


if __name__ == "__main__":
    main()
