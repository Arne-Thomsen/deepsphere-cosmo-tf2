import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import tensorflow as tf
from tensorflow.keras import Model

from . import utils
from scipy.special import comb


class Chebyshev(Model):
    """
    A graph convolutional layer using the Chebyshev approximation
    """

    def __init__(
        self,
        L,
        K,
        Fout=None,
        initializer=None,
        activation=None,
        use_bias=False,
        use_bn=False,
        n_matmul_splits=1,
        depth_wise=False,
        spmm_backend="coo",
        **kwargs,
    ):
        """
        Initializes the graph convolutional layer, assuming the input has dimension (B, M, F)
        :param L: The graph Laplacian (MxM), as numpy array
        :param K: Order of the polynomial to use
        :param Fout: Number of features (channels) of the output, default to number of input channels
        :param initializer: initializer to use for weight initialisation
        :param activation: the activation function to use after the layer, defaults to linear
        :param use_bias: Use learnable bias weights
        :param use_bn: Apply batch norm before adding the bias
        :param n_matmul_splits: Number of splits to apply to axis 1 of the dense tensor in the
            tf.sparse.sparse_dense_matmul operations to avoid the operation's size limitation
        :param depth_wise: Whether to use depthwise convolutions, that is convolutions that don't mix the features (so
            F_in = F_out)
        :param spmm_backend: sparse-matmul backend for L @ x, one of "coo" (default, unchanged),
            "csr" (cuSPARSE), or "gather" (XLA-friendly gather+reduce). See utils.make_spmm_operator.
        :param kwargs: additional keyword arguments passed on to add_weight
        """

        # This is necessary for every Layer
        super(Chebyshev, self).__init__()

        # save necessary params
        self.L = L
        self.K = K
        self.Fout = Fout
        self.use_bias = use_bias
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization(
                axis=-1, momentum=0.9, epsilon=1e-5, center=False, scale=False
            )
        self.initializer = initializer
        if activation is None or callable(activation):
            self.activation = activation
        elif hasattr(tf.keras.activations, activation):
            self.activation = getattr(tf.keras.activations, activation)
        else:
            raise ValueError(f"Could not find activation <{activation}> in tf.keras.activations...")
        self.n_matmul_splits = n_matmul_splits
        self.depth_wise = depth_wise
        self.spmm_backend = spmm_backend
        self.kwargs = kwargs

        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = sparse.csr_matrix(L)
        lmax = 1.02 * eigsh(L, k=1, which="LM", return_eigenvectors=False)[0]
        L = utils.rescale_L(L, lmax=lmax, scale=0.75)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # capture the eager COO for the "csr"/"gather" backends, since build() may run inside a
        # tf.function trace where self.sparse_L is symbolic (it is re-cast there) and not
        # numpy-convertible; both backends need the host-side neighbour structure (see make_spmm_operator)
        self._host_coo = (indices, L.data, L.shape[0]) if self.spmm_backend in ("csr", "gather") else None
        L = tf.SparseTensor(indices, L.data, L.shape)
        self.sparse_L = tf.sparse.reorder(L)

    def build(self, input_shape):
        """
        Build the weights of the layer
        :param input_shape: shape of the input, batch dim has to be defined
        :return: the kernel variable to train
        """

        # get the input shape
        Fin = int(input_shape[-1])
        if self.depth_wise:
            assert (
                self.Fout is None or self.Fout == Fin
            ), "For depthwise convolutions, Fout has to be None or equal to Fin"

        # get Fout if necessary
        if self.Fout is None:
            Fout = Fin
        else:
            Fout = self.Fout

        if self.initializer is None:
            # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
            stddev = 1 / np.sqrt(Fin * (self.K + 0.5) / 2)
            initializer = tf.initializers.TruncatedNormal(stddev=stddev)
        else:
            initializer = self.initializer

        if not self.depth_wise:
            # mix the input features
            self.kernel = self.add_weight("kernel", shape=[self.K * Fin, Fout], initializer=initializer, **self.kwargs)
        else:
            # don't mix the input features, so fewer weights
            self.kernel = self.add_weight("kernel", shape=[Fin, self.K], initializer=initializer, **self.kwargs)

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[1, 1, Fout])

        # we cast the sparse L to the current backend type
        self.sparse_L = tf.cast(self.sparse_L, tf.keras.mixed_precision.global_policy().compute_dtype)

        # precompute the L @ x operator once for the chosen sparse-matmul backend
        self._L_op = utils.make_spmm_operator(
            self.sparse_L, self.spmm_backend, self.n_matmul_splits, host_coo=self._host_coo
        )

    def call(self, input_tensor, training=False):
        """
        Calls the layer on an input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param training: whether we are training or not
        :return: the output of the layer
        """

        # shapes, this fun is necessary since sparse_matmul_dense in TF only supports
        # the multiplication of 2d matrices, therefore one has to do some weird reshaping
        # this is not strictly necessary but leads to a huge performance gain...
        # See: https://arxiv.org/pdf/1903.11409.pdf
        N, M, Fin = input_tensor.get_shape()
        M, Fin = int(M), int(Fin)

        # get Fout if necessary
        if self.Fout is None:
            Fout = Fin
        else:
            Fout = self.Fout

        # Transform to Chebyshev basis
        x0 = tf.transpose(input_tensor, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, -1])  # M x Fin*N

        # list for stacking
        stack = [x0]

        if self.K > 1:
            x1 = self._L_op(x0)
            stack.append(x1)
        for k in range(2, self.K):
            x2 = 2 * self._L_op(x1) - x0  # M x Fin*N
            stack.append(x2)
            x0, x1 = x1, x2
        x = tf.stack(stack, axis=0)  # K x M x Fin*N
        x = tf.reshape(x, [self.K, M, Fin, -1])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K

        if not self.depth_wise:
            x = tf.reshape(x, [-1, Fin * self.K])  # N*M x Fin*K
            # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
            x = tf.matmul(x, self.kernel)  # N*M x Fout
            x = tf.reshape(x, [-1, M, Fout])  # N x M x Fout
        else:
            # one single filter (not bank) per input feature
            x = tf.einsum("ijkl,kl->ijk", x, self.kernel)  # N x M x Fin

        if self.use_bn:
            x = self.bn(x, training=training)

        if self.use_bias:
            x = tf.add(x, self.bias)

        if self.activation is not None:
            x = self.activation(x)

        return x


class Monomial(Model):
    """
    A graph convolutional layer using Monomials
    """

    def __init__(
        self,
        L,
        K,
        Fout=None,
        initializer=None,
        activation=None,
        use_bias=False,
        use_bn=False,
        n_matmul_splits=1,
        spmm_backend="coo",
        **kwargs,
    ):
        """
        Initializes the graph convolutional layer, assuming the input has dimension (B, M, F)
        :param L: The graph Laplacian (MxM), as numpy array
        :param K: Order of the polynomial to use
        :param Fout: Number of features (channels) of the output, default to number of input channels
        :param initializer: initializer to use for weight initialisation
        :param activation: the activation function to use after the layer, defaults to linear
        :param use_bias: Use learnable bias weights
        :param use_bn: Apply batch norm before adding the bias
        :param n_matmul_splits: Number of splits to apply to axis 1 of the dense tensor in the
            tf.sparse.sparse_dense_matmul operations to avoid the operation's size limitation
        :param spmm_backend: sparse-matmul backend for L @ x, one of "coo" (default, unchanged),
            "csr" (cuSPARSE), or "gather" (XLA-friendly gather+reduce). See utils.make_spmm_operator.
        :param kwargs: additional keyword arguments passed on to add_weight
        """

        # This is necessary for every Layer
        super(Monomial, self).__init__()

        # save necessary params
        self.L = L
        self.K = K
        self.Fout = Fout
        self.use_bias = use_bias
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization(
                axis=-1, momentum=0.9, epsilon=1e-5, center=False, scale=False
            )
        self.initializer = initializer
        if activation is None or callable(activation):
            self.activation = activation
        elif hasattr(tf.keras.activations, activation):
            self.activation = getattr(tf.keras.activations, activation)
        else:
            raise ValueError(f"Could not find activation <{activation}> in tf.keras.activations...")
        self.n_matmul_splits = n_matmul_splits
        self.spmm_backend = spmm_backend
        self.kwargs = kwargs

        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = sparse.csr_matrix(L)
        lmax = 1.02 * eigsh(L, k=1, which="LM", return_eigenvectors=False)[0]
        L = utils.rescale_L(L, lmax=lmax)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # capture the eager COO for the "csr"/"gather" backends, since build() may run inside a
        # tf.function trace where self.sparse_L is symbolic (it is re-cast there) and not
        # numpy-convertible; both backends need the host-side neighbour structure (see make_spmm_operator)
        self._host_coo = (indices, L.data, L.shape[0]) if self.spmm_backend in ("csr", "gather") else None
        L = tf.SparseTensor(indices, L.data, L.shape)
        self.sparse_L = tf.sparse.reorder(L)

    def build(self, input_shape):
        """
        Build the weights of the layer
        :param input_shape: shape of the input, batch dim has to be defined
        """

        # get the input shape
        Fin = int(input_shape[-1])

        # get Fout if necessary
        if self.Fout is None:
            Fout = Fin
        else:
            Fout = self.Fout

        if self.initializer is None:
            # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
            initializer = tf.initializers.TruncatedNormal(stddev=0.1)
            self.kernel = self.add_weight("kernel", shape=[self.K * Fin, Fout], initializer=initializer, **self.kwargs)
        else:
            self.kernel = self.add_weight(
                "kernel", shape=[self.K * Fin, Fout], initializer=self.initializer, **self.kwargs
            )

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[1, 1, Fout])

        # we cast the sparse L to the current backend type
        self.sparse_L = tf.cast(self.sparse_L, tf.keras.mixed_precision.global_policy().compute_dtype)

        # precompute the L @ x operator once for the chosen sparse-matmul backend
        self._L_op = utils.make_spmm_operator(
            self.sparse_L, self.spmm_backend, self.n_matmul_splits, host_coo=self._host_coo
        )

    def call(self, input_tensor, training=False):
        """
        Calls the layer on an input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param training: whether we are training or not
        :return: the output of the layer
        """

        # shapes, this fun is necessary since sparse_matmul_dense in TF only supports
        # the multiplication of 2d matrices, therefore one has to do some weird reshaping
        # this is not strictly necessary but leads to a huge performance gain...
        # See: https://arxiv.org/pdf/1903.11409.pdf
        N, M, Fin = input_tensor.get_shape()
        M, Fin = int(M), int(Fin)

        # get Fout if necessary
        if self.Fout is None:
            Fout = Fin
        else:
            Fout = self.Fout

        # Transform to monomial basis.
        x0 = tf.transpose(input_tensor, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, -1])  # M x Fin*N

        # list for stacking
        stack = [x0]

        for k in range(1, self.K):
            x1 = self._L_op(x0)  # M x Fin*N
            stack.append(x1)
            x0 = x1

        x = tf.stack(stack, axis=0)
        x = tf.reshape(x, [self.K, M, Fin, -1])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [-1, Fin * self.K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
        x = tf.matmul(x, self.kernel)  # N*M x Fout
        x = tf.reshape(x, [-1, M, Fout])  # N x M x Fout

        if self.use_bn:
            x = self.bn(x, training=training)

        if self.use_bias:
            x = tf.add(x, self.bias)

        if self.activation is not None:
            x = self.activation(x)

        return x


class GCNN_ResidualLayer(Model):
    """
    A generic residual layer of the form
    in -> layer -> layer -> out + alpha*in
    with optional batchnorm in the end
    """

    def __init__(
        self,
        layer_type,
        layer_kwargs,
        activation=None,
        act_before=False,
        use_bn=False,
        norm_type="batch_norm",
        bn_kwargs=None,
        alpha=1.0,
        drop_path_rate=0.0,
    ):
        """
        Initializes the residual layer with the given argument
        :param layer_type: The layer type, "CHEBY", "MONO" or "BERN" for Chebyshev, Monomial or
                           Bernstein graph convolutions
        :param layer_kwargs: A dictionary with the inputs for the layer
        :param activation: activation function to use for the res layer
        :param act_before: use activation before skip connection
        :param use_bn: use batchnorm inbetween the layers
        :param norm_type: type of batch norm, either batch_norm for normal batch norm or layer_norm for
                          tf.keras.layers.LayerNormalization
        :param bn_kwargs: An optional dictionary containing further keyword arguments for the normalization layer
        :param alpha: Coupling strength of the input -> layer(input) + alpha*input
        :param drop_path_rate: stochastic-depth rate on the residual branch (0.0 = off). Same DropPath
                               layer as GCNN_ConvNeXtLayer -- it is a per-sample Bernoulli mask with no
                               trainable variables, so nothing about it is specific to that block and
                               turning it on here does NOT change the checkpoint lineage.
                               NOTE the mask is EXACT only when the branch is added outside the
                               activation: for activation=None (the pure `x + branch` form, which is what
                               the y3-deep-lss resnet body uses) and for act_before=True with an
                               activation satisfying f(0)=0 (relu/gelu/tanh). In the post-activation form
                               `act(branch + alpha*in)` a dropped branch yields `act(alpha*in)` rather
                               than the identity, so stochastic depth is approximate there; a warning is
                               raised rather than an error, since the approximation is mild for a
                               near-linear activation.
        """
        # This is necessary for every Layer
        super(GCNN_ResidualLayer, self).__init__()

        # save variables
        self.layer_type = layer_type
        self.layer_kwargs = layer_kwargs
        if activation is None or callable(activation):
            self.activation = activation
        elif hasattr(tf.keras.activations, activation):
            self.activation = getattr(tf.keras.activations, activation)
        else:
            raise ValueError(f"Could not find activation <{activation}> in tf.keras.activations...")
        self.act_before = act_before
        self.use_bn = use_bn
        self.norm_type = norm_type
        # set the default axis if necessary
        if bn_kwargs is None:
            self.bn_kwargs = {"axis": -1}
        else:
            self.bn_kwargs = bn_kwargs
            if "axis" not in self.bn_kwargs and norm_type != "moving_norm":
                self.bn_kwargs.update({"axis": -1})

        if self.layer_type == "CHEBY":
            self.layer1 = Chebyshev(**self.layer_kwargs)
            self.layer2 = Chebyshev(**self.layer_kwargs)
        elif self.layer_type == "MONO":
            self.layer1 = Monomial(**self.layer_kwargs)
            self.layer2 = Monomial(**self.layer_kwargs)
        elif self.layer_type == "BERN":
            self.layer1 = Bernstein(**self.layer_kwargs)
            self.layer2 = Bernstein(**self.layer_kwargs)
        else:
            raise IOError(f"Layertype not understood: {self.layer_type}")

        if use_bn:
            if norm_type == "layer_norm":
                self.bn1 = tf.keras.layers.LayerNormalization(**self.bn_kwargs)
                self.bn2 = tf.keras.layers.LayerNormalization(**self.bn_kwargs)
            elif norm_type == "batch_norm":
                self.bn1 = tf.keras.layers.BatchNormalization(**self.bn_kwargs)
                self.bn2 = tf.keras.layers.BatchNormalization(**self.bn_kwargs)
            else:
                raise ValueError(f"norm_type <{norm_type}> not understood!")

        self.alpha = alpha

        self.drop_path = DropPath(drop_path_rate)
        if drop_path_rate > 0.0 and self.activation is not None and not self.act_before:
            warnings.warn(
                "GCNN_ResidualLayer: drop_path_rate > 0 with a post-activation skip "
                "(activation set and act_before=False). The branch is masked inside the activation, so a "
                "dropped path gives activation(alpha*input) rather than the identity -- stochastic depth "
                "is approximate here. Use activation=None or act_before=True for the exact form.",
                RuntimeWarning,
            )

    def call(self, input_tensor, training=False):
        """
        Calls the layer on an input tensorf
        :param input_tensor: The input of the layer
        :param training: whether we are training or not
        :return: the output of the layer
        """
        x = self.layer1(input_tensor)

        # bn
        if self.use_bn:
            x = self.bn1(x, training=training)

        # 2nd layer
        x = self.layer2(x)

        # bn
        if self.use_bn:
            x = self.bn2(x, training=training)

        # stochastic depth on the residual BRANCH only (identity path is never dropped)
        x = self.drop_path(x, training=training)

        # deal with the activation
        if self.activation is None:
            return x + input_tensor

        if self.act_before:
            return self.activation(x) + self.alpha * input_tensor
        else:
            return self.activation(x + self.alpha * input_tensor)


class DropPath(tf.keras.layers.Layer):
    """Stochastic depth: drop the whole residual branch per sample during training.

    timm-style (arXiv:1603.09382). At training time each sample keeps the branch with probability
    ``1 - rate`` and the kept branches are rescaled by ``1 / (1 - rate)`` so the expectation is
    unchanged; at inference (or ``rate == 0``) it is the identity. Applied to the residual branch of
    ``GCNN_ConvNeXtLayer`` (the block's ``x + DropPath(branch)``).
    """

    def __init__(self, rate=0.0, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"DropPath rate must be in [0, 1), got {rate}")
        self.rate = rate

    def call(self, x, training=False):
        if not training or self.rate == 0.0:
            return x
        keep_prob = 1.0 - self.rate
        # one Bernoulli draw per sample, broadcast over the remaining (pixel, channel) axes
        shape = [tf.shape(x)[0]] + [1] * (len(x.shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        binary_mask = tf.floor(random_tensor)
        return (x / keep_prob) * binary_mask

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config


class GCNN_ConvNeXtLayer(Model):
    """ConvNeXt-style graph-conv block (arXiv:2201.03545), channel-preserving ``(N, M, C) -> (N, M, C)``.

    Unlike ``GCNN_ResidualLayer`` (two full ChebK convs, spatial+channel mixing entangled), this
    block *separates* spatial from channel mixing, the ConvNeXt idiom:

        h = depthwise_ChebK(x)      # spatial mix, one spectral filter per channel (no channel mixing)
        h = LayerNorm(h)
        h = Dense(mlp_ratio * C)(h) # pointwise 1x1 expand (inverted bottleneck)
        h = activation(h)           # single activation (GELU)
        h = Dense(C)(h)             # pointwise 1x1 contract
        h = LayerScale(h)           # per-channel gamma, small init -> near-identity at start
        h = DropPath(h)             # stochastic depth on the residual branch
        return x + h

    The depthwise spatial conv is ``Chebyshev(depth_wise=True)`` (Monomial/Bernstein have no
    depthwise path), so this block is Chebyshev-only. It has ONE graph conv per block vs the residual
    layer's two, i.e. ~half the sparse ``L @ x`` cost; and because the depthwise kernel is ``[C, K]``,
    the polynomial order ``K`` only adds ``C * K`` params (no ``C^2`` blowup), so large receptive
    fields (K=8-16) are cheap.
    """

    def __init__(
        self,
        L,
        K,
        mlp_ratio=4.0,
        layer_scale_init=1e-6,
        drop_path_rate=0.0,
        activation="gelu",
        norm_kwargs=None,
        use_bias=True,
        n_matmul_splits=1,
        spmm_backend="coo",
    ):
        """
        :param L: the graph Laplacian (injected by HealpyGCNN via Healpy_ConvNeXtLayer._get_layer)
        :param K: Chebyshev polynomial order of the depthwise spatial conv
        :param mlp_ratio: hidden width of the pointwise MLP as a multiple of the channel count C
        :param layer_scale_init: LayerScale init; None disables LayerScale (unit-weight branch)
        :param drop_path_rate: stochastic-depth rate on the residual branch (0.0 = off)
        :param activation: MLP activation, string (resolved via tf.keras.activations) or callable
        :param norm_kwargs: kwargs for the LayerNormalization (axis defaults to -1)
        :param use_bias: use a bias in the depthwise conv
        :param n_matmul_splits: sparse-matmul split count for the depthwise conv (from HealpyGCNN)
        :param spmm_backend: sparse-matmul backend for the depthwise conv ("coo"/"csr"/"gather")
        """
        super().__init__()

        if callable(activation):
            self.activation = activation
        elif hasattr(tf.keras.activations, activation):
            self.activation = getattr(tf.keras.activations, activation)
        else:
            raise ValueError(f"Could not find activation <{activation}> in tf.keras.activations...")

        self.mlp_ratio = mlp_ratio
        self.layer_scale_init = layer_scale_init

        norm_kwargs = {"axis": -1} if norm_kwargs is None else {"axis": -1, **norm_kwargs}

        # depthwise spatial mixing: one spectral filter per channel, Fout=None keeps the width
        self.dwconv = Chebyshev(
            L=L,
            K=K,
            Fout=None,
            depth_wise=True,
            use_bias=use_bias,
            activation=None,
            use_bn=False,
            n_matmul_splits=n_matmul_splits,
            spmm_backend=spmm_backend,
        )
        self.norm = tf.keras.layers.LayerNormalization(**norm_kwargs)
        self.drop_path = DropPath(drop_path_rate)
        # the pointwise MLP and LayerScale gamma need the channel count, built lazily
        self.pw_expand = None
        self.pw_contract = None
        self.gamma = None

    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden = int(round(self.mlp_ratio * channels))
        self.pw_expand = tf.keras.layers.Dense(hidden)
        self.pw_contract = tf.keras.layers.Dense(channels)
        if self.layer_scale_init is not None:
            self.gamma = self.add_weight(
                name="layer_scale",
                shape=(channels,),
                initializer=tf.keras.initializers.Constant(self.layer_scale_init),
                trainable=True,
            )
        super().build(input_shape)

    def call(self, input_tensor, training=False):
        h = self.dwconv(input_tensor)
        h = self.norm(h)
        h = self.pw_expand(h)
        h = self.activation(h)
        h = self.pw_contract(h)
        if self.gamma is not None:
            h = h * tf.cast(self.gamma, h.dtype)
        h = self.drop_path(h, training=training)
        return input_tensor + h


class Bernstein(Model):
    """
    A graph convolutional layer using the Bernstein approximation
    see https://arxiv.org/abs/2106.10994
    """

    def __init__(
        self,
        L,
        K,
        Fout=None,
        initializer=None,
        activation=None,
        use_bias=False,
        use_bn=False,
        n_matmul_splits=1,
        spmm_backend="coo",
        **kwargs,
    ):
        """
        Initializes the graph convolutional layer, assuming the input has dimension (B, M, F)
        :param L: The graph Laplacian (MxM), as numpy array
        :param K: Order of the polynomial to use
        :param Fout: Number of features (channels) of the output, default to number of input channels
        :param initializer: initializer to use for weight initialisation
        :param activation: the activation function to use after the layer, defaults to linear
        :param use_bias: Use learnable bias weights
        :param use_bn: Apply batch norm before adding the bias
        :param n_matmul_splits: Number of splits to apply to axis 1 of the dense tensor in the
            tf.sparse.sparse_dense_matmul operations to avoid the operation's size limitation
        :param spmm_backend: sparse-matmul backend for L @ x, one of "coo" (default, unchanged),
            "csr" (cuSPARSE), or "gather" (XLA-friendly gather+reduce). See utils.make_spmm_operator.
        :param kwargs: additional keyword arguments passed on to add_weight
        """

        # This is necessary for every Layer
        super(Bernstein, self).__init__()

        # save necessary params
        self.L = L
        self.K = K
        self.Fout = Fout
        self.use_bias = use_bias
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization(
                axis=-1, momentum=0.9, epsilon=1e-5, center=False, scale=False
            )
        self.initializer = initializer
        if activation is None or callable(activation):
            self.activation = activation
        elif hasattr(tf.keras.activations, activation):
            self.activation = getattr(tf.keras.activations, activation)
        else:
            raise ValueError(f"Could not find activation <{activation}> in tf.keras.activations...")
        self.n_matmul_splits = n_matmul_splits
        self.spmm_backend = spmm_backend
        self.kwargs = kwargs

        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = sparse.csr_matrix(L)
        lmax = 1.02 * eigsh(L, k=1, which="LM", return_eigenvectors=False)[0]
        L = utils.rescale_L(L, lmax=lmax, scale=0.75)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # capture the eager COO for the "csr"/"gather" backends, since build() may run inside a
        # tf.function trace where self.sparse_L is symbolic (it is re-cast there) and not
        # numpy-convertible; both backends need the host-side neighbour structure (see make_spmm_operator)
        self._host_coo = (indices, L.data, L.shape[0]) if self.spmm_backend in ("csr", "gather") else None
        L = tf.SparseTensor(indices, L.data, L.shape)
        self.sparse_L = tf.sparse.reorder(L)

    def build(self, input_shape):
        """
        Build the weights of the layer
        :param input_shape: shape of the input, batch dim has to be defined
        :return: the kernel variable to train
        """

        # get the input shape
        Fin = int(input_shape[-1])

        # get Fout if necessary
        if self.Fout is None:
            Fout = Fin
        else:
            Fout = self.Fout

        if self.initializer is None:
            # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
            stddev = np.sqrt(6 / (Fin + Fout))
            initializer = tf.initializers.TruncatedNormal(stddev=stddev)
            self.kernel = self.add_weight(
                "kernel", shape=[(self.K + 1) * Fin, Fout], initializer=initializer, **self.kwargs
            )
        else:
            self.kernel = self.add_weight(
                "kernel", shape=[(self.K + 1) * Fin, Fout], initializer=self.initializer, **self.kwargs
            )

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[1, 1, Fout])

        # we cast the sparse L to the current backend type
        self.sparse_L = tf.cast(self.sparse_L, tf.keras.mixed_precision.global_policy().compute_dtype)

        # precompute the L @ x operator once for the chosen sparse-matmul backend
        self._L_op = utils.make_spmm_operator(
            self.sparse_L, self.spmm_backend, self.n_matmul_splits, host_coo=self._host_coo
        )

    def call(self, input_tensor, training=False, *args, **kwargs):
        """
        Calls the layer on a input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param args: further arguments
        :param training: wheter we are training or not
        :param kwargs: further keyword arguments
        :return: the output of the layer
        """

        # shapes, this fun is necessary since sparse_matmul_dense in TF only supports
        # the multiplication of 2d matrices, therefore one has to do some weird reshaping
        # this is not strictly necessary but leads to a huge performance gain...
        # See: https://arxiv.org/pdf/1903.11409.pdf
        N, M, Fin = input_tensor.get_shape()
        M, Fin = int(M), int(Fin)

        # get Fout if necessary
        if self.Fout is None:
            Fout = Fin
        else:
            Fout = self.Fout

        # Transform to Chebyshev basis
        x0 = tf.transpose(input_tensor, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, -1])  # M x Fin*N

        # Build the K+1 Bernstein basis vectors b_i = comb(K,i)/2^K * (2I - L)^(K-i) * L^i * x0.
        # The L-power ladder Lp[i] = L^i x0 is computed ONCE (K spmms) and shared across terms
        # instead of recomputed per term; then P = (2I - L) is applied (K-i) times to Lp[i].
        # This is FLOP-optimal for the Bernstein basis: K + K(K+1)/2 spmms (20 at K=5) vs the
        # previous double loop's K(K+1) (30 at K=5). Each _L_op call still processes exactly Fin*N
        # columns, so the cuSPARSE nnz*cols kernel limit behaves as before.
        Lp = [x0]
        for _ in range(self.K):
            Lp.append(self._L_op(Lp[-1]))  # Lp[i] = L^i x0

        stack = []
        for i in range(0, self.K + 1):
            theta = comb(self.K, i) / (2**self.K)
            v = Lp[i]
            for _ in range(self.K - i):
                v = 2 * v - self._L_op(v)  # apply P = 2I - L
            stack.append(theta * v)
        x = tf.stack(stack, axis=0)
        x = tf.reshape(x, [(self.K + 1), M, Fin, -1])  # K+1 x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K+1
        x = tf.reshape(x, [-1, Fin * (self.K + 1)])  # N*M x Fin*K+1
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
        x = tf.matmul(x, self.kernel)  # N*M x Fout
        x = tf.reshape(x, [-1, M, Fout])  # N x M x Fout

        if self.use_bn:
            x = self.bn(x, training=training)

        if self.use_bias:
            x = tf.add(x, self.bias)

        if self.activation is not None:
            x = self.activation(x)

        return x
