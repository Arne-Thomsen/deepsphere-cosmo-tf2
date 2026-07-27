import pytest
import numpy as np
import tensorflow as tf
import healpy as hp

from deepsphere import gnn_layers


def test_Chebyshev():
    # create the layer
    tf.random.set_seed(11)
    L = tf.random.normal(shape=(3, 3), seed=11)
    # make sym
    L = tf.matmul(L, tf.transpose(L))
    x = tf.random.normal(shape=(5, 3, 7), seed=12)
    Fout = 3
    K = 4

    # create the layer
    stddev = 1 / np.sqrt(7 * (K + 0.5) / 2)
    initializer = tf.initializers.RandomNormal(stddev=stddev, seed=13)
    cheb = gnn_layers.Chebyshev(L=L.numpy(), Fout=Fout, K=K, initializer=initializer)
    new = cheb(x)

    # same with new activation
    cheb = gnn_layers.Chebyshev(L=L.numpy(), Fout=Fout, K=K, initializer=initializer, activation="linear")
    new = cheb(x)

    # now we test bias and batch norm
    cheb = gnn_layers.Chebyshev(L=L.numpy(), Fout=Fout, K=K, initializer=initializer, activation="linear",
                                use_bias=True, use_bn=True)
    new = cheb(x)


def test_Monimials():
    # create the layer
    tf.random.set_seed(11)
    L = tf.random.normal(shape=(3, 3), seed=11)
    # make sym
    L = tf.matmul(L, tf.transpose(L))
    x = tf.random.normal(shape=(5, 3, 7), seed=12)
    Fout = 3
    K = 4

    # create the layer
    stddev = 0.1
    initializer = tf.initializers.RandomNormal(stddev=stddev, seed=13)
    mon = gnn_layers.Monomial(L=L.numpy(), Fout=Fout, K=K, initializer=initializer,
                              activation=tf.keras.activations.linear)
    new = mon(x)

    # same with new actiation
    mon = gnn_layers.Monomial(L=L.numpy(), Fout=Fout, K=K, initializer=initializer,
                              activation="elu")
    new_1 = mon(x)

    # same with new actiation
    mon = gnn_layers.Monomial(L=L.numpy(), Fout=Fout, K=K, initializer=initializer,
                              activation="elu", use_bn=True, use_bias=True)
    new_1 = mon(x)


def test_Bernstein():
    # create the layer
    tf.random.set_seed(11)
    L = tf.random.normal(shape=(3, 3), seed=11)
    # make sym
    L = tf.matmul(L, tf.transpose(L))
    x = tf.random.normal(shape=(5, 3, 7), seed=12)
    Fout = 3
    K = 4

    # create the layer
    stddev = 1 / np.sqrt(7 * (K + 0.5) / 2)
    initializer = tf.initializers.RandomNormal(stddev=stddev, seed=13)
    bern = gnn_layers.Bernstein(L=L.numpy(), Fout=Fout, K=K, initializer=initializer)
    new = bern(x)

    # same with new activation
    bern = gnn_layers.Bernstein(L=L.numpy(), Fout=Fout, K=K, initializer=initializer, activation="linear")
    new = bern(x)

    # now we test bias and batch norm
    bern = gnn_layers.Bernstein(L=L.numpy(), Fout=Fout, K=K, initializer=initializer, activation="linear",
                                use_bias=True, use_bn=True)
    new = bern(x)


def test_Bernstein_basis_algebra():
    """Pin the Bernstein basis math independently of the layer's evaluation loop. The layer builds
    b_i = comb(K,i)/2^K * (2I - L~)^(K-i) * L~^i * x via a cached L-power ladder; here we form each
    operator explicitly with np.linalg.matrix_power (a different computation path) and check the full
    layer output matches. Guards the O(K) rewrite against any change to the basis it produces."""
    from scipy.special import comb

    # run in float64 so this is a true algebra check (float32 would accumulate ~1e-3 over K powers,
    # masking real errors); the float32 backends are already covered by test_spmm_backend_equivalence.
    prev_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("float64")
    try:
        tf.random.set_seed(11)
        L = tf.random.normal(shape=(6, 6), seed=11, dtype=tf.float64)
        L = tf.matmul(L, tf.transpose(L))  # symmetric
        x = tf.random.normal(shape=(4, 6, 3), seed=12, dtype=tf.float64)  # (N, M, Fin)
        N, M, Fin = 4, 6, 3
        Fout, K = 5, 4

        initializer = tf.initializers.RandomNormal(stddev=0.1, seed=13)
        bern = gnn_layers.Bernstein(L=L.numpy(), Fout=Fout, K=K, initializer=initializer)
        out = bern(x).numpy()  # triggers build

        # the exact rescaled Laplacian the layer used (avoids re-deriving rescale_L / lmax)
        Ltilde = tf.sparse.to_dense(bern.sparse_L).numpy()  # (M, M)
        kernel = bern.kernel.numpy()  # ((K+1)*Fin, Fout)
        P = 2.0 * np.eye(M) - Ltilde

        # x0 in the layer's node-major layout: (M, Fin*N)
        x0 = np.transpose(x.numpy(), (1, 2, 0)).reshape(M, Fin * N)

        basis = []
        for i in range(K + 1):
            op = comb(K, i) / (2**K) * (np.linalg.matrix_power(P, K - i) @ np.linalg.matrix_power(Ltilde, i))
            basis.append(op @ x0)  # (M, Fin*N)
        b = np.stack(basis, axis=0)  # (K+1, M, Fin*N)
        b = b.reshape(K + 1, M, Fin, N).transpose(3, 1, 2, 0).reshape(-1, Fin * (K + 1))  # (N*M, Fin*(K+1))
        ref = (b @ kernel).reshape(N, M, Fout)

        np.testing.assert_allclose(out, ref, rtol=1e-8, atol=1e-8)
    finally:
        tf.keras.mixed_precision.set_global_policy(prev_policy)


def test_GCNN_ResidualLayer():
    # we get a random map to pool
    n_pix = hp.nside2npix(4)
    np.random.seed(11)
    m_in = np.random.normal(size=[3, n_pix, 7])

    # check exception
    with pytest.raises(IOError):
        _ = gnn_layers.GCNN_ResidualLayer("juhu", dict())

    # layer definition
    layer_type = "CHEBY"
    layer_kwargs = {"L": np.eye(n_pix, dtype=np.float64),
                    "K": 5,
                    "activation": tf.keras.activations.relu,
                    "regularizer": tf.keras.regularizers.l1}

    res_layer = gnn_layers.GCNN_ResidualLayer(layer_type=layer_type, layer_kwargs=layer_kwargs,
                                              activation=tf.keras.activations.relu)
    out = res_layer(m_in)

    assert out.numpy().shape == (3, n_pix, 7)

    # same with batch norms
    res_layer = gnn_layers.GCNN_ResidualLayer(layer_type=layer_type, layer_kwargs=layer_kwargs,
                                              activation=tf.keras.activations.relu, use_bn=True)
    out = res_layer(m_in)

    assert out.numpy().shape == (3, n_pix, 7)

    res_layer = gnn_layers.GCNN_ResidualLayer(layer_type=layer_type, layer_kwargs=layer_kwargs,
                                              activation=tf.keras.activations.relu, use_bn=True,
                                              norm_type="layer_norm", bn_kwargs={"axis": (1,2)})
    out = res_layer(m_in)

    assert out.numpy().shape == (3, n_pix, 7)

    with pytest.raises(ValueError):
        res_layer = gnn_layers.GCNN_ResidualLayer(layer_type=layer_type, layer_kwargs=layer_kwargs,
                                                  activation=tf.keras.activations.relu, use_bn=True,
                                                  norm_type="moving_norm")


@pytest.mark.parametrize("layer_cls", [gnn_layers.Chebyshev, gnn_layers.Monomial, gnn_layers.Bernstein])
@pytest.mark.parametrize("backend", ["csr", "gather"])
def test_spmm_backend_equivalence(layer_cls, backend):
    """The "csr" (cuSPARSE) and "gather" (XLA-friendly) sparse-matmul backends must reproduce the
    default "coo" path to float32 tolerance -- this is the correctness gate that makes the backend
    an opt-in performance switch rather than a model change. Uses a real (sparse) HEALPix graph
    Laplacian and copies the coo layer's weights so only the L @ x kernel differs."""
    from pygsp.graphs import SphereHealpix

    sphere = SphereHealpix(subdivisions=4, nest=True, k=20, lap_type="normalized")
    L = np.asarray(sphere.L.todense(), dtype=np.float64)
    npix = L.shape[0]
    x = tf.random.normal(shape=(3, npix, 7), seed=12)

    def build(be):
        init = tf.initializers.RandomNormal(stddev=0.1, seed=13)
        layer = layer_cls(L=L, Fout=5, K=4, initializer=init, use_bias=True, spmm_backend=be)
        layer(x)  # trigger build so the weights exist
        return layer

    coo = build("coo")
    other = build(backend)
    # force identical kernel/bias so any output difference is purely the sparse-matmul backend
    other.set_weights(coo.get_weights())

    ref = coo(x).numpy()
    # eager
    np.testing.assert_allclose(other(x).numpy(), ref, rtol=2e-4, atol=2e-4)
    # graph mode -- the way training runs the layer, and the only mode that exposes csr's captured
    # CSRSparseMatrix-variant shape-inference bug (eager alone would pass right through it)
    other_graph = tf.function(lambda t: other(t))
    np.testing.assert_allclose(other_graph(x).numpy(), ref, rtol=2e-4, atol=2e-4)
