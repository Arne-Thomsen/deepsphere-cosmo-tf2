import warnings

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
    cheb(x)

    # same with new activation
    cheb = gnn_layers.Chebyshev(L=L.numpy(), Fout=Fout, K=K, initializer=initializer, activation="linear")
    cheb(x)

    # now we test bias and batch norm
    cheb = gnn_layers.Chebyshev(
        L=L.numpy(), Fout=Fout, K=K, initializer=initializer, activation="linear", use_bias=True, use_bn=True
    )
    cheb(x)


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
    mon = gnn_layers.Monomial(
        L=L.numpy(), Fout=Fout, K=K, initializer=initializer, activation=tf.keras.activations.linear
    )
    mon(x)

    # same with new actiation
    mon = gnn_layers.Monomial(L=L.numpy(), Fout=Fout, K=K, initializer=initializer, activation="elu")
    mon(x)

    # same with new actiation
    mon = gnn_layers.Monomial(
        L=L.numpy(), Fout=Fout, K=K, initializer=initializer, activation="elu", use_bn=True, use_bias=True
    )
    mon(x)


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
    bern(x)

    # same with new activation
    bern = gnn_layers.Bernstein(L=L.numpy(), Fout=Fout, K=K, initializer=initializer, activation="linear")
    bern(x)

    # now we test bias and batch norm
    bern = gnn_layers.Bernstein(
        L=L.numpy(), Fout=Fout, K=K, initializer=initializer, activation="linear", use_bias=True, use_bn=True
    )
    bern(x)


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
    layer_kwargs = {
        "L": np.eye(n_pix, dtype=np.float64),
        "K": 5,
        "activation": tf.keras.activations.relu,
        "regularizer": tf.keras.regularizers.l1,
    }

    res_layer = gnn_layers.GCNN_ResidualLayer(
        layer_type=layer_type, layer_kwargs=layer_kwargs, activation=tf.keras.activations.relu
    )
    out = res_layer(m_in)

    assert out.numpy().shape == (3, n_pix, 7)

    # same with batch norms
    res_layer = gnn_layers.GCNN_ResidualLayer(
        layer_type=layer_type, layer_kwargs=layer_kwargs, activation=tf.keras.activations.relu, use_bn=True
    )
    out = res_layer(m_in)

    assert out.numpy().shape == (3, n_pix, 7)

    res_layer = gnn_layers.GCNN_ResidualLayer(
        layer_type=layer_type,
        layer_kwargs=layer_kwargs,
        activation=tf.keras.activations.relu,
        use_bn=True,
        norm_type="layer_norm",
        bn_kwargs={"axis": (1, 2)},
    )
    out = res_layer(m_in)

    assert out.numpy().shape == (3, n_pix, 7)

    with pytest.raises(ValueError):
        res_layer = gnn_layers.GCNN_ResidualLayer(
            layer_type=layer_type,
            layer_kwargs=layer_kwargs,
            activation=tf.keras.activations.relu,
            use_bn=True,
            norm_type="moving_norm",
        )


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


def test_Chebyshev_depthwise():
    # depthwise conv: one spectral filter per channel, no channel mixing, so Fout must equal Fin
    tf.random.set_seed(11)
    L = tf.random.normal(shape=(3, 3), seed=11)
    L = tf.matmul(L, tf.transpose(L))  # make symmetric
    Fin, K = 7, 4
    x = tf.random.normal(shape=(5, 3, Fin), seed=12)

    cheb = gnn_layers.Chebyshev(L=L.numpy(), K=K, depth_wise=True, use_bias=True)
    out = cheb(x)
    # channel-preserving and the kernel is [Fin, K] (not [K*Fin, Fout])
    assert out.shape == (5, 3, Fin)
    assert tuple(cheb.kernel.shape) == (Fin, K)

    # Fout is allowed only if it equals Fin; anything else must raise the depthwise assertion
    gnn_layers.Chebyshev(L=L.numpy(), K=K, Fout=Fin, depth_wise=True)(x)
    with pytest.raises(AssertionError):
        gnn_layers.Chebyshev(L=L.numpy(), K=K, Fout=Fin + 1, depth_wise=True)(x)


def test_DropPath():
    tf.random.set_seed(11)
    x = tf.random.normal(shape=(64, 3, 7), seed=12)

    # rate 0 is the identity in both modes
    dp0 = gnn_layers.DropPath(rate=0.0)
    np.testing.assert_allclose(dp0(x, training=True).numpy(), x.numpy())
    np.testing.assert_allclose(dp0(x, training=False).numpy(), x.numpy())

    # rate > 0 is the identity at inference, and drops whole samples (rescaled) while training
    dp = gnn_layers.DropPath(rate=0.5)
    np.testing.assert_allclose(dp(x, training=False).numpy(), x.numpy())
    out = dp(x, training=True).numpy()
    # each sample is either fully zeroed or uniformly rescaled by 1/keep_prob
    per_sample_zeroed = np.all(out == 0.0, axis=(1, 2))
    assert per_sample_zeroed.any() and not per_sample_zeroed.all()
    kept = out[~per_sample_zeroed]
    np.testing.assert_allclose(kept, x.numpy()[~per_sample_zeroed] / 0.5, rtol=1e-5)

    with pytest.raises(ValueError):
        gnn_layers.DropPath(rate=1.0)


def test_GCNN_ResidualLayer_drop_path():
    # stochastic depth is NOT a ConvNeXt-only feature: DropPath is a per-sample Bernoulli mask with no
    # trainable variables, so it applies to the classic two-conv residual block just as well. This is the
    # form y3-deep-lss uses (activation=None -> the pure `x + branch` skip), where the mask is EXACT:
    # a dropped sample must come out of the block bit-identical to its input.
    tf.random.set_seed(11)
    L = tf.random.normal(shape=(3, 3), seed=11)
    L = tf.matmul(L, tf.transpose(L))  # make symmetric
    # rescale to unit spectral radius. The sibling tests leave L unnormalized, which is harmless for a
    # SINGLE conv but not here: this block chains TWO Chebyshev convs, and on an L with eigenvalues ~10
    # the K=4 recursion amplifies to ~1e4 before cancellation, leaving ~1e-3 of float32 noise between two
    # separately-built instances. Normalizing keeps the comparison below sharp instead of forcing a
    # tolerance so loose it could no longer see the effect under test.
    L = L / tf.reduce_max(tf.abs(tf.linalg.eigvalsh(L)))
    C, K = 8, 4
    x = tf.random.normal(shape=(64, 3, C), seed=12)
    layer_kwargs = {"L": L.numpy(), "K": K, "use_bias": True}

    # rate 0 must reproduce the un-dropped block exactly, in both modes
    block0 = gnn_layers.GCNN_ResidualLayer("CHEBY", dict(layer_kwargs), drop_path_rate=0.0)
    ref = block0(x, training=True).numpy()
    np.testing.assert_allclose(block0(x, training=False).numpy(), ref, rtol=1e-6, atol=1e-6)

    block = gnn_layers.GCNN_ResidualLayer("CHEBY", dict(layer_kwargs), drop_path_rate=0.5)
    block(x)  # build
    block.set_weights(block0.get_weights())  # identical weights -> any difference is DropPath alone
    # assert the copy landed, EXACTLY, so the output comparisons below cannot be excused by a bad copy
    # (they run at a float tolerance; this one does not need to)
    for w_new, w_ref in zip(block.get_weights(), block0.get_weights()):
        assert np.array_equal(w_new, w_ref), "set_weights did not reproduce the rate-0 block's weights"

    # at inference DropPath is the identity, so the block matches the rate-0 block. This is the one
    # CROSS-instance comparison and it carries a float tolerance: two separately-built layers do not agree
    # bitwise (the repo's spmm-backend test allows 2e-4 for a single conv; this block chains two, and the
    # measured spread is ~7e-4 on a handful of elements). 5e-3 still sits ~100x below the ~2x/0x deviation
    # a broken inference path would produce, so the assertion keeps its teeth.
    out_inf = block(x, training=False).numpy()
    np.testing.assert_allclose(out_inf, ref, rtol=5e-3, atol=5e-3)

    # everything below compares the SAME instance against itself, so it is not exposed to that spread.
    # While training, a dropped sample falls back to the identity path -- and this one is asserted
    # EXACTLY, because no arithmetic happens on it: the branch is multiplied by a hard zero and what
    # comes out is the input tensor itself. This is the test's real content.
    out = block(x, training=True).numpy()
    dropped = np.all(out == x.numpy(), axis=(1, 2))
    assert dropped.any() and not dropped.all(), "expected some but not all samples to drop the branch"
    # and a kept sample is the identity plus the branch rescaled by 1/keep_prob
    kept_branch = (out[~dropped] - x.numpy()[~dropped]) / 2.0
    np.testing.assert_allclose(kept_branch, (out_inf - x.numpy())[~dropped], rtol=5e-3, atol=5e-3)


def test_GCNN_ResidualLayer_drop_path_post_activation_warns():
    # with a post-activation skip the branch is masked INSIDE the activation, so a dropped path gives
    # activation(alpha*input) rather than the identity -- approximate, and it must say so out loud.
    L = tf.random.normal(shape=(3, 3), seed=11)
    L = tf.matmul(L, tf.transpose(L))
    layer_kwargs = {"L": L.numpy(), "K": 4, "use_bias": True}

    with pytest.warns(RuntimeWarning, match="approximate"):
        gnn_layers.GCNN_ResidualLayer("CHEBY", dict(layer_kwargs), activation="relu", drop_path_rate=0.1)

    # the exact forms must NOT warn: activation=None, act_before=True, and rate 0 with any activation
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gnn_layers.GCNN_ResidualLayer("CHEBY", dict(layer_kwargs), drop_path_rate=0.1)
        gnn_layers.GCNN_ResidualLayer(
            "CHEBY", dict(layer_kwargs), activation="relu", act_before=True, drop_path_rate=0.1
        )
        gnn_layers.GCNN_ResidualLayer("CHEBY", dict(layer_kwargs), activation="relu", drop_path_rate=0.0)


def test_GCNN_ConvNeXtLayer():
    tf.random.set_seed(11)
    L = tf.random.normal(shape=(3, 3), seed=11)
    L = tf.matmul(L, tf.transpose(L))  # make symmetric
    C, K = 8, 5
    x = tf.random.normal(shape=(5, 3, C), seed=12)

    block = gnn_layers.GCNN_ConvNeXtLayer(L=L.numpy(), K=K, mlp_ratio=4.0, layer_scale_init=1e-6)
    out = block(x)
    # channel-preserving
    assert out.shape == (5, 3, C)
    # LayerScale gamma is per-channel and starts tiny -> the block is a near-identity at init
    assert tuple(block.gamma.shape) == (C,)
    np.testing.assert_allclose(out.numpy(), x.numpy(), atol=1e-3)
    # the depthwise spatial conv has a [C, K] kernel (no C^2 channel-mixing weights)
    assert tuple(block.dwconv.kernel.shape) == (C, K)

    # layer_scale_init=None disables LayerScale (no gamma) and the block still runs channel-preserving
    block_no_ls = gnn_layers.GCNN_ConvNeXtLayer(L=L.numpy(), K=K, layer_scale_init=None)
    assert block_no_ls(x).shape == (5, 3, C)
    assert block_no_ls.gamma is None
