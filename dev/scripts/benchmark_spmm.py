"""Micro-benchmark for the DeepSphere sparse graph-conv core (L @ x), across spmm backends.

Every Chebyshev graph conv applies the (rescaled) graph Laplacian K-1 times. Today that goes
through ``deepsphere.utils.split_sparse_dense_matmul`` -> TF's generic ``SparseTensorDenseMatMul``
CUDA kernel (NOT cuSPARSE), with the ``n_splits`` int32-overflow workaround. This script times the
three pluggable backends ``deepsphere.utils.make_spmm_operator`` actually ships, so the de-risk gate
measures the *production* operators rather than a re-implementation:

  coo         -- current path: split_sparse_dense_matmul (the baseline the network runs today).
  csr-eager   -- cuSPARSE csrmm (tf.raw_ops.SparseMatrixMatMul), called EAGERLY.
  csr-graph   -- the SAME csr operator, but called inside @tf.function -- this is how training runs
                 it, and the mode where a captured CSRSparseMatrix variant can lose its shape info.
                 We test both because eager success does NOT imply graph success for csr.
  gather-graph-- gather + weighted reduce, inside @tf.function (no jit).
  gather-xla  -- the same, under @tf.function(jit_compile=True): the "compile the body" case.

Realism: the L @ x RHS width is W = fin * batch (a conv reshapes (batch, M, Fin) so L multiplies an
(M, batch*Fin) matrix), and the coo split count is computed with the exact production formula from
healpy_networks.py. The gather backend materialises a dense (N, D, W) tensor (D = max node degree);
when N*D*W exceeds 2**31 its elementwise kernel overflows int32 indexing and faults the GPU context,
so we GUARD it: if N*D*W > 2**31 the gather rows are reported SKIPPED (with the overflow ratio)
instead of crashing. This is itself a finding -- gather as written needs RHS chunking to serve
conv-width RHS, and is only viable as-is for the narrow-RHS smoothing path (see benchmark_smoothing.py).

For each (nside, backend) it reports steady-state ms per full Chebyshev-K L-application and the
max-abs difference vs the coo baseline (the correctness gate for adopting a backend).

Run on a GPU compute node, e.g.:
    srun -A a0158 --environment=tensorflow --partition=normal --pty bash
    python dev/scripts/benchmark_spmm.py --nsides 128 256 512 --batch 20 --fin 32 --K 5

Stage any --out csv on /iopsstor scratch (login-node paths are invisible to the compute node).
"""

import argparse
import time

import numpy as np
import healpy as hp
import tensorflow as tf
from scipy import sparse
from scipy.sparse.linalg import eigsh
from pygsp.graphs import SphereHealpix

from deepsphere import utils

INT32_LIMIT = 2**31  # tf.sparse_dense_matmul output limit AND the gather (N,D,W) element-index limit


def des_y3_footprint(nside, fsky):
    """A contiguous DES-Y3-sized footprint: pixels inside a polar cap of sky fraction fsky.

    Training builds the graph on the masked footprint (HealpyGCNN passes indexes=indices to
    SphereHealpix), not the full sphere -- so the realistic node count is ~fsky*npix, ~10x smaller
    than the full 12*nside**2. Matching that keeps both the L@x timing and the (fast) graph build
    representative of the network."""
    theta_cap = np.arccos(1.0 - 2.0 * fsky)  # cap half-angle giving 2*pi*(1-cos) = 4*pi*fsky
    indices = hp.query_disc(nside, hp.ang2vec(0.0, 0.0), radius=theta_cap, nest=True)
    return np.sort(indices).astype(np.int64)


def build_rescaled_L(nside, k, indices, dtype=np.float32):
    """Build the rescaled graph Laplacian on the masked footprint, as deepsphere does in training
    (SphereHealpix(indexes=indices) -> conv layer eigsh rescale)."""
    sphere = SphereHealpix(subdivisions=nside, indexes=indices, nest=True, k=k, lap_type="normalized")
    L = sparse.csr_matrix(sphere.L)
    lmax = 1.02 * eigsh(L, k=1, which="LM", return_eigenvectors=False)[0]
    L = utils.rescale_L(L, lmax=lmax, scale=0.75)  # eigenvalues -> [-0.75, 0.75]
    return L.tocoo().astype(dtype)


def make_sparse_tensor(L_coo):
    """(rescaled, row-major reordered) Laplacian as a tf.SparseTensor -- what make_spmm_operator wants."""
    indices = np.column_stack((L_coo.row, L_coo.col))
    return tf.sparse.reorder(tf.SparseTensor(indices, L_coo.data, L_coo.shape))


def production_n_splits(width, nnz):
    """The exact split count the network computes (healpy_networks.py:134-142): grow until the RHS
    width divides evenly AND width*nnz/2**31 is covered."""
    n = 1
    while not ((width % n == 0) and (n >= width * nnz / INT32_LIMIT)):
        n += 1
    return n


def max_degree(L_coo):
    """Max nonzeros per row = D, the neighbour axis the gather backend materialises."""
    return int(np.bincount(L_coo.row, minlength=L_coo.shape[0]).max())


def chebyshev_stack(op, x0, K):
    """T0=x, T1=L~x, Tk=2 L~ T{k-1} - T{k-2}; `op` applies the rescaled L via a chosen backend."""
    stack = [x0]
    x1 = op(x0) if K > 1 else x0
    if K > 1:
        stack.append(x1)
    xm2, xm1 = x0, x1
    for _ in range(2, K):
        x2 = 2.0 * op(xm1) - xm2
        stack.append(x2)
        xm2, xm1 = xm1, x2
    return tf.stack(stack, axis=0)


def make_runner(op, K, mode):
    """Wrap the Chebyshev stack for a given execution mode: eager / graph / xla."""
    fn = lambda x: chebyshev_stack(op, x, K)  # noqa: E731
    if mode == "eager":
        return fn
    return tf.function(fn, jit_compile=(mode == "xla"))


def time_runner(run, x0, n_warmup, n_iter):
    for _ in range(n_warmup):
        out = run(x0)
    _ = out.numpy()  # force device sync after warmup
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = run(x0)
    _ = out.numpy()  # single sync after the batch of iters
    dt = (time.perf_counter() - t0) / n_iter
    return dt * 1e3, out.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nsides", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--k", type=int, default=60, help="PyGSP neighbour count (matches configs)")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--fin", type=int, default=32, help="input feature dim (dense width = fin*batch)")
    ap.add_argument("--K", type=int, default=5, help="Chebyshev polynomial order (poly_degree)")
    ap.add_argument("--n_warmup", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=50)
    ap.add_argument("--fsky", type=float, default=0.10,
                    help="masked footprint sky fraction (training uses a mask; DES Y3 ~= 0.10). "
                         "Set 1.0 for the full sphere.")
    ap.add_argument("--out", type=str, default=None, help="optional CSV path (stage on /iopsstor)")
    args = ap.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    print(f"# TF {tf.__version__} | GPUs: {[g.name for g in gpus] or 'NONE (CPU-only)'}")
    print(f"# k={args.k} batch={args.batch} fin={args.fin} K={args.K} "
          f"warmup={args.n_warmup} iter={args.n_iter}", flush=True)

    # (label, backend, mode). coo-graph is the reference. csr is tested eager AND graph on purpose.
    plan = [("coo", "coo", "graph"),
            ("csr-eager", "csr", "eager"),
            ("csr-graph", "csr", "graph"),
            ("gather-graph", "gather", "graph"),
            ("gather-xla", "gather", "xla")]

    header = f"{'nside':>6} {'npix':>10} {'nnz':>12} {'backend':>13} {'ms/apply':>10} {'maxdiff_vs_coo':>16}"
    rows = []
    for nside in args.nsides:
        indices = None if args.fsky >= 1.0 else des_y3_footprint(nside, args.fsky)
        L = build_rescaled_L(nside, args.k, indices)
        sp = make_sparse_tensor(L)
        npix = L.shape[0]
        W = args.fin * args.batch
        D = max_degree(L)
        n_splits = production_n_splits(W, L.nnz)
        gather_elems = npix * D * W  # size of the (N, D, W) tensor the gather backend materialises
        x0 = tf.constant(np.random.default_rng(0).standard_normal((npix, W)).astype(np.float32))
        print(f"\n# nside={nside} fsky={args.fsky} npix={npix} nnz={L.nnz} maxdeg(D)={D} W(fin*batch)={W} "
              f"coo_n_splits={n_splits} gather(N*D*W)={gather_elems:.3e} "
              f"({gather_elems / INT32_LIMIT:.1f}x 2^31)")
        print(header)
        print("-" * len(header), flush=True)

        ref = None
        for label, backend, mode in plan:
            # Guard the gather backend: an (N,D,W) beyond 2**31 overflows int32 indexing in the
            # elementwise multiply and faults the whole CUDA context, so skip cleanly instead.
            if backend == "gather" and gather_elems > INT32_LIMIT:
                print(f"{nside:>6} {npix:>10} {L.nnz:>12} {label:>13} {'SKIPPED':>10}   "
                      f"N*D*W={gather_elems:.3e} > 2^31 (needs RHS chunking)", flush=True)
                rows.append((nside, npix, L.nnz, label, float("nan"), float("nan")))
                continue
            try:
                op = utils.make_spmm_operator(sp, backend, n_splits)
                run = make_runner(op, args.K, mode)
                ms, out = time_runner(run, x0, args.n_warmup, args.n_iter)
            except Exception as exc:
                msg = str(exc).splitlines()[0][:90]
                print(f"{nside:>6} {npix:>10} {L.nnz:>12} {label:>13} {'FAILED':>10}   "
                      f"{type(exc).__name__}: {msg}", flush=True)
                rows.append((nside, npix, L.nnz, label, float("nan"), float("nan")))
                continue
            if label == "coo":
                ref = out
            maxdiff = float(np.max(np.abs(out - ref))) if ref is not None else float("nan")
            print(f"{nside:>6} {npix:>10} {L.nnz:>12} {label:>13} {ms:>10.3f} {maxdiff:>16.3e}", flush=True)
            rows.append((nside, npix, L.nnz, label, ms, maxdiff))

    if args.out:
        import csv

        with open(args.out, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["nside", "npix", "nnz", "backend", "ms_per_apply", "maxdiff_vs_coo"])
            wr.writerows(rows)
        print(f"\n# wrote {args.out}")


if __name__ == "__main__":
    main()
