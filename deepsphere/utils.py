"""Utilities module."""

import numpy as np
from scipy import sparse
import healpy as hp
import tensorflow as tf


def extend_indices(indices, nside_in, nside_out, nest=True):
    """
    Minimally extends a set of indices such that it can be reduced to nside_out in a healpy fashion, always four pixels
    reduce naturally to a higher order pixel. Note that this function supports the ring ordering, however, since almost
    no other function does so, nest ordering is strongly recommended.
    :param indices: 1d array of integer pixel ids
    :param nside_in: nside of the input
    :param nside_out: nside of the output
    :param nest: indices are ordered in the "NEST" ordering scheme
    :return: returns a set of indices in the same ordering as the input.
    """
    # figire out the ordering
    if nest:
        ordering = "NEST"
    else:
        ordering = "RING"

    # get the map to reduce
    m_in = np.zeros(hp.nside2npix(nside_in))
    m_in[indices] = 1.0

    # reduce
    m_in = hp.ud_grade(map_in=m_in, nside_out=nside_out, order_in=ordering, order_out=ordering)

    # expand
    m_in = hp.ud_grade(map_in=m_in, nside_out=nside_in, order_in=ordering, order_out=ordering)

    # get the new indices
    return np.arange(hp.nside2npix(nside_in))[m_in > 1e-12]


def rescale_L(L, lmax=2, scale=1):
    """Rescale the Laplacian eigenvalues in [-scale,scale]."""
    M, M = L.shape
    I = sparse.identity(M, format="csr", dtype=L.dtype)
    L *= 2 * scale / lmax
    L -= I
    return L


@tf.function
def split_sparse_dense_matmul(sparse_tensor, dense_tensor, n_splits=1):
    """
    Splits axis 1 of the dense_tensor such that tensorflow can handle the size of the computation.
    :param sparse_tensor: Input sparse tensor of rank 2.
    :param dense_tensor: Input dense tensor of rank 2.
    :param n_splits: Integer number of splits applied to axis 1 of dense_tensor.

    For reference, the error message to be avoided is:

    'Cannot use GPU when output.shape[1] * nnz(a) > 2^31 [Op:SparseTensorDenseMatMul]

    Call arguments received by layer "chebyshev" (type Chebyshev):
    • input_tensor=tf.Tensor(shape=(208, 7264, 128), dtype=float32)
    • training=False'
    """
    if n_splits > 1:
        print(
            f"Tracing... Due to tensor size, tf.sparse.sparse_dense_matmul is executed over {n_splits} splits."
            f" Beware of the resulting performance penalty."
        )
        dense_splits = tf.split(dense_tensor, n_splits, axis=1)
        result = []
        for dense_split in dense_splits:
            result.append(tf.sparse.sparse_dense_matmul(sparse_tensor, dense_split))
        result = tf.concat(result, axis=1)
    else:
        result = tf.sparse.sparse_dense_matmul(sparse_tensor, dense_tensor)

    return result


# --------------------------------------------------------------------------------------------------
# Pluggable sparse-matmul backends for the graph convolutions (L @ x, applied K-1 times per conv).
#
# The graph Laplacian is fixed for the life of a layer, so we precompute a backend-specific operator
# once (in the layer's build) and reuse it. `make_spmm_operator` returns a small callable
# `op(dense) -> dense` implementing the same L @ x, so the conv layers stay backend-agnostic:
#
#   "coo"    -- current behaviour (default): tf.sparse.sparse_dense_matmul via split_sparse_dense_matmul,
#               which lowers to TF's generic SparseTensorDenseMatMul CUDA kernel. Byte-identical to the
#               pre-existing path, including the n_splits int32-overflow workaround.
#   "csr"    -- cuSPARSE csrmm via tf.raw_ops.SparseMatrixMatMul. The CSRSparseMatrix is (re)built inside
#               the call from stored dense COO-component constants (NOT cached as a variant, which loses
#               its shape info when captured into a tf.function trace). Same fp32 math, no 2**31
#               output-size limit -> the n_splits fragmentation is not needed. cuSPARSE csrmm is fp32/fp64
#               only, so the operand is cast to float32 for the multiply and back to the caller dtype.
#   "gather" -- L @ x rewritten as a static neighbour gather + weighted reduce (pure dense ops), so the
#               conv body is XLA-compilable. Padded (N, D) neighbour index / weight tables are built once;
#               padded slots point at node 0 with weight 0, contributing nothing.
# --------------------------------------------------------------------------------------------------
_SPMM_BACKENDS = ("coo", "csr", "gather")


class _CooOperator:
    """Default backend: the pre-existing tf.sparse.sparse_dense_matmul path."""

    def __init__(self, sparse_tensor, n_matmul_splits=1):
        self.sparse_tensor = sparse_tensor
        self.n_matmul_splits = n_matmul_splits

    def __call__(self, dense_tensor):
        return split_sparse_dense_matmul(self.sparse_tensor, dense_tensor, self.n_matmul_splits)


class _CsrOperator:
    """cuSPARSE csrmm backend.

    Stores the operator as dense COO-component *constants* (int64 row-major indices, fp32 values,
    int64 shape) and (re)builds the ``CSRSparseMatrix`` *inside* ``__call__``, so the conversion op
    lives in the same graph as the matmul. Building the CSR once in ``__init__`` and caching the
    opaque variant instead works eagerly but fails the moment ``__call__`` is traced into a
    ``tf.function`` -- a captured variant constant carries no shape metadata ("Unable to access shape
    and type info from variant input 0 for SparseMatrixMatMul") -- and that trace is every training
    step. Dense constants, unlike the variant, keep their shape info when captured, and the
    conversion is a cheap O(nnz) row-pointer build next to the csrmm itself. cuSPARSE csrmm is
    fp32/fp64 only, so values are stored fp32 and the operand is cast to fp32 for the multiply and
    back to the caller dtype (a no-op on the fp32 path).

    :param coo: (indices, values, n_rows) host-numpy COO. The graph Laplacian / smoothing kernel is
        square, so the dense shape is (n_rows, n_rows).
    """

    def __init__(self, coo):
        indices, values, n_rows = coo
        indices = np.asarray(indices)
        order = np.lexsort((indices[:, 1], indices[:, 0]))  # CSR requires row-major-sorted coords
        self._indices = tf.constant(indices[order].astype(np.int64))
        self._values = tf.constant(np.asarray(values)[order].astype(np.float32))
        self._dense_shape = tf.constant(np.asarray([n_rows, n_rows], dtype=np.int64))

    def __call__(self, dense_tensor):
        csr = tf.raw_ops.SparseTensorToCSRSparseMatrix(
            indices=self._indices, values=self._values, dense_shape=self._dense_shape
        )
        y = tf.raw_ops.SparseMatrixMatMul(a=csr, b=tf.cast(dense_tensor, tf.float32))
        return tf.cast(y, dense_tensor.dtype)


class _GatherOperator:
    """XLA-friendly backend: L @ x as a padded neighbour gather + weighted reduce.

    :param coo: (indices, values, n_rows) as host numpy arrays -- the neighbour structure of the
        operator. Passed explicitly (rather than read off the SparseTensor) because a conv layer's
        build() may run inside a tf.function trace, where the SparseTensor is symbolic and cannot be
        converted to numpy; the layer captures this eager COO at construction time. The order of the
        entries does not matter -- rows are grouped here.
    :param out_dtype: dtype for the weight table (the layer's compute dtype).
    """

    def __init__(self, coo, out_dtype):
        indices, values, n_rows = coo
        rows = np.asarray(indices)[:, 0]
        cols = np.asarray(indices)[:, 1]
        values = np.asarray(values)
        order = np.argsort(rows, kind="stable")  # group by row so within-row slots are contiguous
        rows, cols, values = rows[order], cols[order], values[order]
        counts = np.bincount(rows, minlength=n_rows)
        max_deg = int(counts.max()) if len(counts) else 0
        within = np.arange(len(rows)) - np.repeat(np.cumsum(counts) - counts, counts)
        idx = np.zeros((n_rows, max_deg), dtype=np.int32)  # padded slots -> node 0 ...
        w = np.zeros((n_rows, max_deg), dtype=np.float64)  # ... with weight 0
        idx[rows, within] = cols
        w[rows, within] = values
        self._idx = tf.constant(idx)
        self._w = tf.constant(w[..., None], dtype=out_dtype)  # (N, D, 1)

    def __call__(self, dense_tensor):
        # gather neighbour rows -> (N, D, W), weight and sum over the neighbour axis -> (N, W)
        return tf.reduce_sum(tf.gather(dense_tensor, self._idx, axis=0) * self._w, axis=1)


def sparse_tensor_to_coo_numpy(sparse_tensor):
    """Return (indices, values, n_rows) as host numpy from an *eager* SparseTensor (for "csr"/"gather")."""
    return (
        np.asarray(sparse_tensor.indices),
        np.asarray(sparse_tensor.values),
        int(np.asarray(sparse_tensor.dense_shape)[0]),
    )


def make_spmm_operator(sparse_tensor, backend="coo", n_matmul_splits=1, host_coo=None):
    """Precompute a reusable ``op(dense) -> dense`` implementing L @ x for the given backend.

    :param sparse_tensor: the (rescaled, row-major reordered) graph Laplacian as a tf.SparseTensor.
    :param backend: one of "coo" (default, unchanged behaviour), "csr" (cuSPARSE), "gather" (XLA-friendly).
    :param n_matmul_splits: split count for the "coo" backend only (ignored by "csr"/"gather").
    :param host_coo: (indices, values, n_rows) host-numpy COO for the "csr" and "gather" backends.
        Required when sparse_tensor may be symbolic at call time (conv layers, whose build() runs in a
        trace and casts sparse_L); when omitted it is read off the (eager) sparse_tensor, which is the
        case for the smoothing layer whose kernel is built eagerly.
    """
    if backend == "coo":
        return _CooOperator(sparse_tensor, n_matmul_splits)
    elif backend == "csr":
        if host_coo is None:
            host_coo = sparse_tensor_to_coo_numpy(sparse_tensor)
        return _CsrOperator(host_coo)
    elif backend == "gather":
        if host_coo is None:
            host_coo = sparse_tensor_to_coo_numpy(sparse_tensor)
        return _GatherOperator(host_coo, out_dtype=sparse_tensor.values.dtype)
    raise ValueError(f"Unknown spmm_backend <{backend}>; expected one of {_SPMM_BACKENDS}.")


class GaussianNoiseLayer(tf.keras.layers.Layer):
    """
    A layer that adds Gaussian noise to the input, where the standard deviation of the Gaussian can be set channel-wise
    """

    def __init__(self, stddev, **kwargs):
        super(GaussianNoiseLayer, self).__init__(**kwargs)
        self.stddev = tf.convert_to_tensor(stddev, dtype=tf.float32)

    def build(self, input_shape):
        if len(self.stddev.shape) == 0:
            self.stddev = tf.ones((input_shape[-1],)) * self.stddev
        elif self.stddev.shape[0] != input_shape[-1]:
            raise ValueError("Length of stddev does not match the number of input channels")

    def call(self, inputs):
        noise = tf.random.normal(
            shape=tf.shape(inputs),
            mean=0.0,
            stddev=tf.cast(self.stddev, inputs.dtype),
            dtype=inputs.dtype,
        )

        return inputs + noise
