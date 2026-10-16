#!/usr/bin/env python

"""Tests for NIR import and export."""

import nir
import numpy as np
import pytest
import snntorch as snn
from snntorch.export_nir import export_to_nir
from snntorch.import_nir import import_from_nir
import torch


# sample data for snntorch_sequential
@pytest.fixture(scope="module")
def sample_data():
    return torch.ones((4, 784))


# sample data for snntorch with conv2d_avgpool
@pytest.fixture(scope="module")
def sample_data2():
    return torch.randn(1, 1, 28, 28)


class NetWithAvgPool(torch.nn.Module):
    def __init__(self):
        super(NetWithAvgPool, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = torch.nn.Flatten()
        self.lif1 = snn.Leaky(
            beta=0.9 * torch.ones(28 * 28 * 16 // 4),
            threshold=torch.ones(28 * 28 * 16 // 4),
            init_hidden=True,
        )
        self.fc1 = torch.nn.Linear(28 * 28 * 16 // 4, 500)
        self.lif2 = snn.Leaky(
            beta=0.9 * torch.ones(500),
            threshold=torch.ones(500),
            init_hidden=True,
            output=True,
        )

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.flatten(x)
        x = self.lif1(x)
        x = self.fc1(x)
        x = self.lif2(x)
        return x


@pytest.fixture(scope="module")
def net_with_avg_pool():
    net = NetWithAvgPool()
    return net


@pytest.fixture(scope="module")
def snntorch_sequential():
    lif1 = snn.Leaky(
        beta=0.9 * torch.ones(500), threshold=torch.ones(500), init_hidden=True
    )
    lif2 = snn.Leaky(
        beta=0.9 * torch.ones(10),
        threshold=torch.ones(10),
        init_hidden=True,
        output=True,
    )

    return torch.nn.Sequential(
        torch.nn.Linear(784, 500),
        lif1,
        torch.nn.Linear(500, 10),
        lif2,
    )


@pytest.fixture(scope="module")
def snntorch_recurrent():
    v = torch.ones((500,))
    lif1 = snn.RSynaptic(
        alpha=0.5 * torch.ones(500),
        beta=0.9 * torch.ones(500),
        V=v,
        all_to_all=False,
        init_hidden=True,
    )
    lif2 = snn.Leaky(
        beta=0.9 * torch.ones(10),
        threshold=torch.ones(10),
        init_hidden=True,
        output=True,
    )

    return torch.nn.Sequential(
        torch.nn.Linear(784, 500),
        lif1,
        torch.nn.Linear(500, 10),
        lif2,
    )


@pytest.fixture(scope="module")
def snntorch_rleaky():
    v = torch.ones((500,))
    lif1 = snn.RLeaky(
        beta=0.9 * torch.ones(500),
        threshold=torch.ones(500),
        V=v,
        all_to_all=False,
        init_hidden=True,
    )
    lif2 = snn.Leaky(
        beta=0.9 * torch.ones(10),
        threshold=torch.ones(10),
        init_hidden=True,
        output=True,
    )

    return torch.nn.Sequential(
        torch.nn.Linear(784, 500),
        lif1,
        torch.nn.Linear(500, 10),
        lif2,
    )


class TestNIR:
    """Test import and export from snnTorch to NIR."""

    def test_export_sequential(self, snntorch_sequential, sample_data):
        nir_graph = export_to_nir(
            snntorch_sequential, sample_data, ignore_dims=[0]
        )
        assert nir_graph is not None
        assert set(nir_graph.nodes.keys()) == set(
            ["input", "output"] + [str(i) for i in range(4)]
        ), nir_graph.nodes.keys()
        assert set(nir_graph.edges) == set(
            [
                ("3", "output"),
                ("input", "0"),
                ("2", "3"),
                ("1", "2"),
                ("0", "1"),
            ]
        )
        assert isinstance(nir_graph.nodes["input"], nir.Input)
        assert isinstance(nir_graph.nodes["output"], nir.Output)
        assert isinstance(nir_graph.nodes["0"], nir.Affine)
        assert isinstance(nir_graph.nodes["1"], nir.LIF)
        assert isinstance(nir_graph.nodes["2"], nir.Affine)
        assert isinstance(nir_graph.nodes["3"], nir.LIF)

    def test_export_NetWithAvgPool(self, net_with_avg_pool, sample_data2):
        pytest.xfail("conv2d export currently unsupported")

    def test_export_recurrent(self, snntorch_recurrent, sample_data):
        nir_graph = export_to_nir(
            snntorch_recurrent, sample_data, ignore_dims=[0]
        )
        assert nir_graph is not None
        assert set(nir_graph.nodes.keys()) == set(
            ["input", "output", "0", "1.lif", "1.w_rec", "2", "3"]
        ), nir_graph.nodes.keys()
        assert isinstance(nir_graph.nodes["input"], nir.Input)
        assert isinstance(nir_graph.nodes["output"], nir.Output)
        assert isinstance(nir_graph.nodes["0"], nir.Affine)
        assert isinstance(nir_graph.nodes["1.lif"], nir.CubaLIF)
        assert isinstance(nir_graph.nodes["1.w_rec"], nir.Linear)
        assert isinstance(nir_graph.nodes["2"], nir.Affine)
        assert isinstance(nir_graph.nodes["3"], nir.LIF)
        assert set(nir_graph.edges) == set(
            [
                ("1.lif", "1.w_rec"),
                ("1.w_rec", "1.lif"),
                ("0", "1.lif"),
                ("3", "output"),
                ("2", "3"),
                ("input", "0"),
                ("1.lif", "2"),
            ]
        )

    def test_export_rleaky(self, snntorch_rleaky, sample_data):
        nir_graph = export_to_nir(
            snntorch_rleaky, sample_data, ignore_dims=[0]
        )
        assert nir_graph is not None
        assert set(nir_graph.nodes.keys()) == set(
            ["input", "output", "0", "1.lif", "1.w_rec", "2", "3"]
        ), nir_graph.nodes.keys()
        assert isinstance(nir_graph.nodes["input"], nir.Input)
        assert isinstance(nir_graph.nodes["output"], nir.Output)
        assert isinstance(nir_graph.nodes["0"], nir.Affine)
        assert isinstance(nir_graph.nodes["1.lif"], nir.LIF)
        assert isinstance(nir_graph.nodes["1.w_rec"], nir.Linear)
        assert isinstance(nir_graph.nodes["2"], nir.Affine)
        assert isinstance(nir_graph.nodes["3"], nir.LIF)
        assert set(nir_graph.edges) == set(
            [
                ("1.lif", "1.w_rec"),
                ("1.w_rec", "1.lif"),
                ("0", "1.lif"),
                ("3", "output"),
                ("2", "3"),
                ("input", "0"),
                ("1.lif", "2"),
            ]
        )

    def test_import_nir(self):
        graph = nir.read("tests/lif.nir")
        net = import_from_nir(graph)
        out, _ = net(torch.ones(1, 1))
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape == (1, 1), out.shape

    def test_import_if_node_is_ideal_integrator(self):
        """Regression test for issue #416.

        NIR defines the IF primitive as an ideal integrate-and-fire
        neuron, v[t+1] = v[t] + r * i[t], so it must be imported as a
        leak-free ``snn.Leaky`` with ``beta = 1``. Previously the import
        hardcoded a leak (``beta=0.9``, later ``beta=0``), which silently
        changed the neuron dynamics of imported models.
        """
        n = 3
        graph = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type=np.array([n])),
                "if": nir.IF(r=np.ones(n), v_threshold=np.ones(n)),
                "output": nir.Output(output_type=np.array([n])),
            },
            edges=[("input", "if"), ("if", "output")],
        )
        net = import_from_nir(graph)

        leaky_mods = [m for m in net.modules() if isinstance(m, snn.Leaky)]
        assert len(leaky_mods) == 1, "expected exactly one imported neuron"
        lif = leaky_mods[0]

        # An ideal integrator has no leak.
        assert torch.all(lif.beta == 1.0), lif.beta

        # Sub-threshold, the membrane must accumulate linearly:
        # v[t] = t * i for constant input i (no decay between steps).
        x = 0.0625 * torch.ones(1, n)
        lif.reset_mem()
        for step in range(1, 5):
            lif(x)
            assert torch.allclose(lif.mem, step * x), (step, lif.mem)

        # A constant sub-threshold input must eventually accumulate up to
        # the threshold and emit a spike. With a leak (beta < 1), the
        # membrane would plateau below threshold and never fire.
        lif.reset_mem()
        spk_count = torch.zeros(1, n)
        for _ in range(20):
            spk_count += lif(x)
        assert torch.all(spk_count >= 1), spk_count

    def test_import_conv_nir(self):
        pytest.xfail("conv2d import unsupported")

    def test_commute_sequential(self, snntorch_sequential, sample_data):
        x = torch.rand((4, 784))
        y_snn, state = snntorch_sequential(x)
        assert y_snn.shape == (4, 10)
        nir_graph = export_to_nir(
            snntorch_sequential, sample_data, ignore_dims=[0]
        )
        net = import_from_nir(nir_graph)
        y_nir, state = net(x)
        if isinstance(y_nir, tuple):
            y_nir = y_nir[0]
        assert y_nir.shape == (4, 10), y_nir.shape
        assert torch.allclose(y_snn, y_nir)

    def test_commute_rleaky(self, snntorch_rleaky, sample_data):
        x = torch.rand((4, 784))
        y_snn, state = snntorch_rleaky(x)
        assert y_snn.shape == (4, 10)
        nir_graph = export_to_nir(
            snntorch_rleaky, sample_data, ignore_dims=[0]
        )
        net = import_from_nir(nir_graph)
        y_nir, state = net(x)
        assert y_nir.shape == (4, 10), y_nir.shape
        assert torch.allclose(y_snn, y_nir)


def _roundtrip(graph, path):
    nir.write(str(path), graph)
    return nir.read(str(path))


def _scalar_synaptic_net(in_features, width):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, width),
        snn.Synaptic(
            alpha=0.5,
            beta=0.9,
            threshold=1.0,
            init_hidden=True,
            output=True,
        ),
    )


def _scalar_leaky_net(in_features, width):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, width),
        snn.Leaky(beta=0.9, threshold=1.0, init_hidden=True, output=True),
    )


_LIF_PARAMS = ("tau", "r", "v_leak", "v_reset", "v_threshold")
_CUBALIF_PARAMS = (
    "tau_syn",
    "tau_mem",
    "r",
    "v_leak",
    "v_reset",
    "v_threshold",
    "w_in",
)


def _assert_population_shape(node, shape):
    params = _CUBALIF_PARAMS if isinstance(node, nir.CubaLIF) else _LIF_PARAMS
    for attr in params:
        assert np.asarray(getattr(node, attr)).shape == shape, attr
    assert tuple(node.input_type["input"]) == shape
    assert tuple(node.output_type["output"]) == shape


class TestNeuronParamShapes:
    """Regression tests for issues #410 and #334."""

    def test_issue_410_scalar_synaptic_roundtrip(self, tmp_path):
        net = _scalar_synaptic_net(2450, 128)
        graph = export_to_nir(net, torch.ones(1, 2450), ignore_dims=[0])

        neuron = graph.nodes["1"]
        assert isinstance(neuron, nir.CubaLIF)
        assert np.asarray(neuron.v_threshold).shape == (128,)
        assert np.asarray(neuron.tau_syn).shape == (128,)
        assert np.asarray(neuron.tau_mem).shape == (128,)
        assert np.asarray(neuron.r).shape == (128,)
        assert np.asarray(neuron.v_leak).shape == (128,)
        assert np.asarray(neuron.v_reset).shape == (128,)
        assert np.asarray(neuron.w_in).shape == (128,)
        assert list(np.asarray(neuron.input_type["input"])) == [128]
        assert list(np.asarray(neuron.output_type["output"])) == [128]

        reloaded = _roundtrip(graph, tmp_path / "issue410.nir")
        reloaded_neuron = reloaded.nodes["1"]
        assert np.allclose(
            np.asarray(reloaded_neuron.v_threshold),
            np.asarray(neuron.v_threshold),
        )
        assert np.allclose(
            np.asarray(reloaded_neuron.tau_mem),
            np.asarray(neuron.tau_mem),
        )
        assert np.allclose(
            np.asarray(reloaded_neuron.tau_syn),
            np.asarray(neuron.tau_syn),
        )
        assert np.asarray(reloaded_neuron.v_threshold).shape == (128,)

    def test_scalar_synaptic_width_4(self, tmp_path):
        net = _scalar_synaptic_net(16, 4)
        graph = export_to_nir(net, torch.ones(1, 16), ignore_dims=[0])
        neuron = graph.nodes["1"]
        assert isinstance(neuron, nir.CubaLIF)
        assert np.asarray(neuron.v_threshold).shape == (4,)
        reloaded = _roundtrip(graph, tmp_path / "width4.nir")
        assert np.asarray(reloaded.nodes["1"].v_threshold).shape == (4,)

    def test_scalar_leaky_roundtrip(self, tmp_path):
        net = _scalar_leaky_net(2450, 128)
        graph = export_to_nir(net, torch.ones(1, 2450), ignore_dims=[0])

        neuron = graph.nodes["1"]
        assert isinstance(neuron, nir.LIF)
        assert np.asarray(neuron.v_threshold).shape == (128,)
        assert np.asarray(neuron.tau).shape == (128,)
        assert np.asarray(neuron.r).shape == (128,)
        assert np.asarray(neuron.v_leak).shape == (128,)
        assert np.asarray(neuron.v_reset).shape == (128,)
        assert list(np.asarray(neuron.input_type["input"])) == [128]
        assert list(np.asarray(neuron.output_type["output"])) == [128]

        reloaded = _roundtrip(graph, tmp_path / "scalar_leaky.nir")
        assert np.asarray(reloaded.nodes["1"].v_threshold).shape == (128,)
        assert np.allclose(
            np.asarray(reloaded.nodes["1"].v_threshold),
            np.asarray(neuron.v_threshold),
        )

    def test_vector_params_unchanged(self, tmp_path):
        n = 16
        beta = 0.9 * torch.ones(n)
        thr = torch.ones(n)
        alpha = 0.5 * torch.ones(n)
        net = torch.nn.Sequential(
            torch.nn.Linear(32, n),
            snn.Synaptic(
                alpha=alpha,
                beta=beta,
                threshold=thr,
                init_hidden=True,
                output=True,
            ),
        )
        graph = export_to_nir(net, torch.ones(1, 32), ignore_dims=[0])
        neuron = graph.nodes["1"]
        assert np.asarray(neuron.v_threshold).shape == (n,)
        assert np.allclose(np.asarray(neuron.v_threshold), 1.0)
        reloaded = _roundtrip(graph, tmp_path / "vector.nir")
        assert np.asarray(reloaded.nodes["1"].v_threshold).shape == (n,)
        assert np.allclose(np.asarray(reloaded.nodes["1"].v_threshold), 1.0)

    def test_mixed_param_shapes(self):
        """A vector beta/alpha combined with the default scalar threshold."""
        n = 8
        leaky = snn.Leaky(
            beta=0.9 * torch.ones(n), init_hidden=True, output=True
        )
        synaptic = snn.Synaptic(
            alpha=0.5 * torch.ones(n), beta=0.9, init_hidden=True, output=True
        )
        for neuron in (leaky, synaptic):
            net = torch.nn.Sequential(torch.nn.Linear(16, n), neuron)
            graph = export_to_nir(net, torch.ones(1, 16), ignore_dims=[0])
            _assert_population_shape(graph.nodes["1"], (n,))
            assert np.allclose(graph.nodes["1"].v_threshold, 1.0)

    def test_conv_scalar_params(self, tmp_path):
        """Neurons after a Conv2d layer have the (C, H, W) output shape."""
        net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
            snn.Leaky(beta=0.9, init_hidden=True),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 8 * 8, 10),
            snn.Leaky(beta=0.9, init_hidden=True, output=True),
        )
        graph = export_to_nir(net, torch.ones(1, 1, 8, 8), ignore_dims=[0])
        _assert_population_shape(graph.nodes["1"], (4, 8, 8))
        _assert_population_shape(graph.nodes["4"], (10,))
        reloaded = _roundtrip(graph, tmp_path / "conv.nir")
        _assert_population_shape(reloaded.nodes["1"], (4, 8, 8))

    def test_conv_per_channel_params(self):
        """Per-channel (C, 1, 1) parameters are broadcast along H and W."""
        beta = torch.linspace(0.5, 0.9, 4).reshape(4, 1, 1)
        net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
            snn.Leaky(beta=beta, init_hidden=True),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 8 * 8, 10),
            snn.Leaky(beta=0.9, init_hidden=True, output=True),
        )
        graph = export_to_nir(net, torch.ones(1, 1, 8, 8), ignore_dims=[0])
        neuron = graph.nodes["1"]
        _assert_population_shape(neuron, (4, 8, 8))
        dt = 1e-4
        expected_tau = np.broadcast_to(dt / (1 - beta.numpy()), (4, 8, 8))
        assert np.allclose(neuron.tau, expected_tau)

    def test_flatten_then_neuron(self):
        net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            snn.Leaky(beta=0.9, init_hidden=True),
            torch.nn.Linear(4 * 4 * 4, 10),
            snn.Leaky(beta=0.9, init_hidden=True, output=True),
        )
        graph = export_to_nir(net, torch.ones(1, 1, 8, 8), ignore_dims=[0])
        _assert_population_shape(graph.nodes["3"], (4 * 4 * 4,))
        _assert_population_shape(graph.nodes["5"], (10,))

    @pytest.mark.parametrize(
        "recurrent",
        [
            lambda: snn.RLeaky(beta=0.9, linear_features=40, init_hidden=True),
            lambda: snn.RSynaptic(
                alpha=0.5, beta=0.9, linear_features=40, init_hidden=True
            ),
        ],
        ids=["RLeaky", "RSynaptic"],
    )
    def test_recurrent_followed_by_scalar_neuron(self, recurrent):
        net = torch.nn.Sequential(
            torch.nn.Linear(12, 40),
            recurrent(),
            torch.nn.Linear(40, 7),
            snn.Leaky(beta=0.9, init_hidden=True, output=True),
        )
        graph = export_to_nir(net, torch.ones(1, 12), ignore_dims=[0])
        _assert_population_shape(graph.nodes["1.lif"], (40,))
        _assert_population_shape(graph.nodes["3"], (7,))

    def test_class_style_names(self):
        """Neurons are matched by module name, also in nested modules."""

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(16, 20)
                self.lif1 = snn.Leaky(beta=0.9)
                self.blk = torch.nn.Sequential(
                    torch.nn.Linear(20, 5),
                    snn.Leaky(beta=0.8, init_hidden=True, output=True),
                )

            def forward(self, x):
                mem1 = self.lif1.reset_mem()
                spk1, mem1 = self.lif1(self.fc1(x), mem1)
                return self.blk(spk1)

        graph = export_to_nir(Net(), torch.ones(1, 16), ignore_dims=[0])
        _assert_population_shape(graph.nodes["lif1"], (20,))
        _assert_population_shape(graph.nodes["blk.1"], (5,))

    def test_scalar_equals_vector_export(self):
        n = 8
        scalar = snn.Synaptic(
            alpha=0.7, beta=0.9, threshold=1.3, init_hidden=True, output=True
        )
        vector = snn.Synaptic(
            alpha=0.7 * torch.ones(n),
            beta=0.9 * torch.ones(n),
            threshold=1.3 * torch.ones(n),
            init_hidden=True,
            output=True,
        )
        nodes = [
            export_to_nir(
                torch.nn.Sequential(torch.nn.Linear(16, n), neuron),
                torch.ones(1, 16),
                ignore_dims=[0],
            ).nodes["1"]
            for neuron in (scalar, vector)
        ]
        for attr in _CUBALIF_PARAMS:
            assert np.array_equal(
                getattr(nodes[0], attr), getattr(nodes[1], attr)
            ), attr

    def test_hooks_removed(self):
        """Exporting leaves no hooks on the model, also when it fails."""
        net = _scalar_leaky_net(16, 8)
        export_to_nir(net, torch.ones(1, 16), ignore_dims=[0])
        with pytest.raises(Exception):
            export_to_nir(net, torch.ones(1, 3), ignore_dims=[0])
        assert all(len(m._forward_pre_hooks) == 0 for m in net.modules())


class TestAvgPool2dExport:
    """AvgPool2d is exported with (height, width) pairs."""

    @pytest.mark.parametrize(
        "pool, kernel_size, stride",
        [
            (torch.nn.AvgPool2d(2), (2, 2), (2, 2)),
            (torch.nn.AvgPool2d(2, stride=1), (2, 2), (1, 1)),
            (torch.nn.AvgPool2d((2, 3), stride=(1, 2)), (2, 3), (1, 2)),
        ],
        ids=["int", "int_stride", "tuple"],
    )
    def test_kernel_stride_padding_are_pairs(
        self, pool, kernel_size, stride, tmp_path
    ):
        net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
            pool,
            snn.Leaky(beta=0.9, init_hidden=True, output=True),
        )
        graph = export_to_nir(net, torch.ones(1, 1, 8, 8), ignore_dims=[0])
        reloaded = _roundtrip(graph, tmp_path / "avgpool.nir")
        for node in (graph.nodes["1"], reloaded.nodes["1"]):
            assert tuple(node.kernel_size) == kernel_size
            assert tuple(node.stride) == stride
            assert tuple(node.padding) == (0, 0)

    def test_padding_exported(self):
        net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
            torch.nn.AvgPool2d(2, padding=1),
            snn.Leaky(beta=0.9, init_hidden=True, output=True),
        )
        graph = export_to_nir(net, torch.ones(1, 1, 8, 8), ignore_dims=[0])
        assert tuple(graph.nodes["1"].padding) == (1, 1)
        _assert_population_shape(graph.nodes["2"], (4, 5, 5))

    def test_conv_avgpool_roundtrip(self):
        """The imported network produces the same spikes over time."""
        from snntorch import utils

        torch.manual_seed(0)
        net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
            torch.nn.AvgPool2d(2),
            # NIR neurons reset to v_reset, i.e. the "zero" reset mechanism
            snn.Leaky(
                beta=0.9,
                init_hidden=True,
                output=True,
                reset_mechanism="zero",
            ),
        )
        x = 4 * torch.rand(5, 1, 1, 8, 8)
        graph = export_to_nir(net, x[0], ignore_dims=[0])
        imported = import_from_nir(graph)

        utils.reset(net)
        state = None
        n_spikes = 0
        for x_t in x:
            spk_snn, _ = net(x_t)
            spk_nir, state = imported(x_t, state)
            if isinstance(spk_nir, tuple):
                spk_nir = spk_nir[0]
            assert torch.equal(spk_snn, spk_nir)
            n_spikes += int(spk_snn.sum())
        assert n_spikes > 0


class TestUnsupportedModules:
    """Modules without a NIR equivalent raise instead of exporting."""

    @pytest.mark.parametrize(
        "layer, name",
        [
            (lambda: torch.nn.MaxPool2d(2), "MaxPool2d"),
            (
                lambda: snn.Alpha(alpha=0.9, beta=0.8, init_hidden=True),
                "Alpha",
            ),
        ],
        ids=["MaxPool2d", "Alpha"],
    )
    def test_unsupported_module_raises(self, layer, name):
        net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
            layer(),
            snn.Leaky(beta=0.9, init_hidden=True, output=True),
        )
        with pytest.raises(NotImplementedError, match=name):
            export_to_nir(net, torch.ones(1, 1, 8, 8), ignore_dims=[0])

    def _avgpool_net(self, pool):
        return torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
            pool,
            snn.Leaky(beta=0.9, init_hidden=True, output=True),
        )

    def test_avgpool_divisor_override_raises(self):
        net = self._avgpool_net(torch.nn.AvgPool2d(2, divisor_override=1))
        with pytest.raises(NotImplementedError, match="divisor_override"):
            export_to_nir(net, torch.ones(1, 1, 8, 8), ignore_dims=[0])

    def test_avgpool_count_include_pad_with_padding_raises(self):
        pool = torch.nn.AvgPool2d(2, padding=1, count_include_pad=False)
        net = self._avgpool_net(pool)
        with pytest.raises(NotImplementedError, match="count_include_pad"):
            export_to_nir(net, torch.ones(1, 1, 8, 8), ignore_dims=[0])

    def test_avgpool_count_include_pad_without_padding_exports(self):
        pool = torch.nn.AvgPool2d(2, count_include_pad=False)
        net = self._avgpool_net(pool)
        graph = export_to_nir(net, torch.ones(1, 1, 8, 8), ignore_dims=[0])
        assert isinstance(graph.nodes["1"], nir.AvgPool2d)


_LEARN = dict(init_hidden=True, output=True)


def _with_learned(neuron, **values):
    for name, value in values.items():
        getattr(neuron, name).data.fill_(value)
    return neuron


class TestBetaAlphaEdgeValues:
    """beta and alpha are exported as clamped to [0, 1] by the neurons."""

    def test_leaky_beta_one_exports_if(self):
        from snntorch import utils

        torch.manual_seed(0)
        net = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            snn.Leaky(
                beta=1.0,
                init_hidden=True,
                output=True,
                reset_mechanism="zero",
            ),
        )
        # a positive input current integrates up to the threshold
        net[0].weight.data.fill_(0.2)
        net[0].bias.data.zero_()
        x = torch.rand(6, 1, 4)
        graph = export_to_nir(net, x[0], ignore_dims=[0])
        neuron = graph.nodes["1"]
        assert isinstance(neuron, nir.IF)
        assert np.array_equal(neuron.r, np.ones(3))
        assert tuple(neuron.input_type["input"]) == (3,)

        # NIR IF: v[t+1] = v[t] + r * i[t], reset to v_reset after a spike
        utils.reset(net)
        v = np.zeros(3, dtype=np.float32)
        n_spikes = 0
        for x_t in x:
            spk_snn, _ = net(x_t)
            v = v + neuron.r * net[0](x_t).detach().numpy()[0]
            spk_nir = (v > neuron.v_threshold).astype(np.float32)
            v = np.where(spk_nir == 1, neuron.v_reset, v)
            assert np.array_equal(spk_snn.detach().numpy()[0], spk_nir)
            n_spikes += int(spk_nir.sum())
        assert n_spikes > 0

    @pytest.mark.parametrize(
        "neuron, learned, node_type, attr, expected",
        [
            (
                lambda: snn.Leaky(beta=0.9, learn_beta=True, **_LEARN),
                {"beta": 1.2},
                nir.IF,
                "r",
                1.0,
            ),
            (
                lambda: snn.Leaky(beta=0.9, learn_beta=True, **_LEARN),
                {"beta": -0.3},
                nir.LIF,
                "tau",
                1e-4,
            ),
            (
                lambda: snn.Synaptic(
                    alpha=0.5, beta=0.9, learn_beta=True, **_LEARN
                ),
                {"beta": -0.3},
                nir.CubaLIF,
                "tau_mem",
                1e-4,
            ),
        ],
        ids=["leaky_above_one", "leaky_below_zero", "synaptic_below_zero"],
    )
    def test_learned_values_are_clamped(
        self, neuron, learned, node_type, attr, expected
    ):
        layer = _with_learned(neuron(), **learned)
        net = torch.nn.Sequential(torch.nn.Linear(4, 3), layer)
        graph = export_to_nir(net, torch.ones(1, 4), ignore_dims=[0])
        node = graph.nodes["1"]
        assert isinstance(node, node_type)
        assert np.allclose(getattr(node, attr), expected)

    def test_leaky_partial_beta_one_raises(self):
        layer = snn.Leaky(beta=torch.tensor([0.9, 1.0, 1.0]), **_LEARN)
        net = torch.nn.Sequential(torch.nn.Linear(4, 3), layer)
        with pytest.raises(NotImplementedError, match="only some"):
            export_to_nir(net, torch.ones(1, 4), ignore_dims=[0])

    @pytest.mark.parametrize(
        "neuron",
        [
            lambda: snn.Synaptic(alpha=1.0, beta=0.9, **_LEARN),
            lambda: snn.Synaptic(alpha=0.5, beta=1.0, **_LEARN),
            lambda: _with_learned(
                snn.Synaptic(alpha=0.5, beta=0.9, learn_alpha=True, **_LEARN),
                alpha=1.3,
            ),
        ],
        ids=["alpha_one", "beta_one", "learned_alpha_above_one"],
    )
    def test_synaptic_alpha_or_beta_one_raises(self, neuron):
        net = torch.nn.Sequential(torch.nn.Linear(4, 3), neuron())
        with pytest.raises(NotImplementedError, match="Synaptic"):
            export_to_nir(net, torch.ones(1, 4), ignore_dims=[0])

    @pytest.mark.parametrize(
        "neuron, name",
        [
            (
                lambda: snn.RLeaky(beta=1.0, linear_features=3, **_LEARN),
                "RLeaky",
            ),
            (
                lambda: snn.RSynaptic(
                    alpha=0.5, beta=1.0, linear_features=3, **_LEARN
                ),
                "RSynaptic",
            ),
        ],
        ids=["RLeaky", "RSynaptic"],
    )
    def test_recurrent_beta_one_raises(self, neuron, name):
        net = torch.nn.Sequential(torch.nn.Linear(4, 3), neuron())
        with pytest.raises(NotImplementedError, match=name):
            export_to_nir(net, torch.ones(1, 4), ignore_dims=[0])
