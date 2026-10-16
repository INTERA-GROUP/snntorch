#from typing import Optional
from typing import Optional, Tuple, Union
import torch
import os
import sys
import nir
import numpy as np
import nirtorch
import snntorch as snn



def _extract_snntorch_module(module: torch.nn.Module) -> Optional[nir.NIRNode]:
    """Convert a single snnTorch module to the equivalent object in the Neuromorphic
    Intermediate Representation (NIR). This function is used internally by the export_to_nir
    function to convert each submodule/layer of the network to the NIR.

    Currently supported snnTorch modules: Leaky, Linear, Synaptic, RLeaky, RSynaptic.

    Note that recurrent layers such as RLeaky and RSynaptic will be converted to a NIR graph,
    which will then be embedded as a subgraph into the main NIR graph.

    :param module: snnTorch module
    :type module: torch.nn.Module

    :return: return the NIR node
    :rtype: Optional[nir.NIRNode]
    """
    # Adding Conv2d layer
    """
    if isinstance(module, torch.nn.Conv2d):
        return nir.Conv2d(
            input_shape=None,
            weight=module.weight.detach(),
            bias=module.bias.detach(),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
    """
    #modifiying bias of the conv2d layer extraction 
    if isinstance(module, torch.nn.Conv2d):
        return nir.Conv2d(
            input_shape=None,
            weight=module.weight.detach(),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            #better handle for the bias if it's False
            bias=(
                module.bias.detach()
                if isinstance(module.bias, torch.Tensor)
                else torch.zeros((module.weight.shape[0]))
            ),
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        return nir.AvgPool2d(
            kernel_size=module.kernel_size,  # (Height, Width)
            stride=module.kernel_size
            if module.stride is None
            else module.stride,  # (Height, width)
            padding=(0, 0),  # (Height, width)
        )

    elif isinstance(module, snn.Leaky):
        dt = 1e-4

        beta = module.beta.detach().cpu().numpy()
        vthr = module.threshold.detach().cpu().numpy()
        # NIR requires all parameters of a node to have the same shape
        beta, vthr = (np.array(a) for a in np.broadcast_arrays(beta, vthr))
        tau_mem = dt / (1 - beta)
        r = tau_mem / dt
        v_leak = np.zeros_like(beta)

        return nir.LIF(
            tau=tau_mem,
            v_threshold=vthr,
            v_leak=v_leak,
            r=r,
            v_reset=np.zeros_like(vthr),
        )

    elif isinstance(module, torch.nn.Linear):
        if module.bias is None:
            return nir.Linear(weight=module.weight.data.detach().cpu().numpy())
        else:
            return nir.Affine(
                weight=module.weight.data.detach().cpu().numpy(),
                bias=module.bias.data.detach().cpu().numpy(),
            )

    elif isinstance(module, snn.Synaptic):
        dt = 1e-4

        # TODO: assert that size of the current layer is correct
        alpha = module.alpha.detach().cpu().numpy()
        beta = module.beta.detach().cpu().numpy()
        vthr = module.threshold.detach().cpu().numpy()
        # NIR requires all parameters of a node to have the same shape
        alpha, beta, vthr = (
            np.array(a) for a in np.broadcast_arrays(alpha, beta, vthr)
        )

        tau_syn = dt / (1 - alpha)
        tau_mem = dt / (1 - beta)
        r = tau_mem / dt
        v_leak = np.zeros_like(beta)
        w_in = tau_syn / dt

        return nir.CubaLIF(
            tau_syn=tau_syn,
            tau_mem=tau_mem,
            v_threshold=vthr,
            v_leak=v_leak,
            r=r,
            w_in=w_in,
            v_reset=np.zeros_like(vthr),
        )

    elif isinstance(module, snn.RLeaky):
        if module.all_to_all:
            w_rec = _extract_snntorch_module(module.recurrent)
            if hasattr(w_rec, "weight"):
                n_neurons = w_rec.weight.shape[0]
            else:
                # fallback for conv layers
                n_neurons = w_rec.weight.shape[0]
        else:
            if len(module.recurrent.V.shape) == 0:
                raise ValueError(
                    "V must be a vector, cannot infer layer size for scalar V"
                )
            n_neurons = module.recurrent.V.shape[0]
            w = np.diag(module.recurrent.V.data.detach().cpu().numpy())
            w_rec = nir.Linear(weight=w)

        dt = 1e-4

        beta = module.beta.detach().cpu().numpy()
        vthr = module.threshold.detach().cpu().numpy()
        beta = np.ones(n_neurons) * beta
        vthr = np.ones(n_neurons) * vthr

        tau_mem = dt / (1 - beta)
        r = tau_mem / dt
        v_leak = np.zeros_like(beta)

        return nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type=[n_neurons]),
                "lif": nir.LIF(
                    v_threshold=vthr,
                    tau=tau_mem,
                    r=r,
                    v_leak=v_leak,
                    v_reset=np.zeros_like(vthr),
                ),
                "w_rec": w_rec,
                "output": nir.Output(output_type=[n_neurons]),
            },
            edges=[
                ("input", "lif"),
                ("lif", "w_rec"),
                ("w_rec", "lif"),
                ("lif", "output"),
            ],
        )

    elif isinstance(module, snn.RSynaptic):
        if module.all_to_all:
            w_rec = _extract_snntorch_module(module.recurrent)
            n_neurons = w_rec.weight.shape[0]
        else:
            if len(module.recurrent.V.shape) == 0:
                # TODO: handle this better - if V is a scalar, then the weight has wrong shape
                raise ValueError(
                    "V must be a vector, cannot infer layer size for scalar V"
                )
            n_neurons = module.recurrent.V.shape[0]
            w = np.diag(module.recurrent.V.data.detach().cpu().numpy())
            w_rec = nir.Linear(weight=w)

        dt = 1e-4

        alpha = module.alpha.detach().cpu().numpy()
        beta = module.beta.detach().cpu().numpy()
        vthr = module.threshold.detach().cpu().numpy()
        alpha = np.ones(n_neurons) * alpha
        beta = np.ones(n_neurons) * beta
        vthr = np.ones(n_neurons) * vthr

        tau_syn = dt / (1 - alpha)
        tau_mem = dt / (1 - beta)
        r = tau_mem / dt
        v_leak = np.zeros_like(beta)
        w_in = tau_syn / dt

        return nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type=[n_neurons]),
                "lif": nir.CubaLIF(
                    v_threshold=vthr,
                    tau_mem=tau_mem,
                    tau_syn=tau_syn,
                    r=r,
                    v_leak=v_leak,
                    w_in=w_in,
                    v_reset=np.zeros_like(vthr),
                ),
                "w_rec": w_rec,
                "output": nir.Output(output_type=[n_neurons]),
            },
            edges=[
                ("input", "lif"),
                ("lif", "w_rec"),
                ("w_rec", "lif"),
                ("lif", "output"),
            ],
        )
    elif isinstance(module, torch.nn.Flatten):
        # Getting rid of the batch dimension for NIR
        start_dim = (
            module.start_dim - 1 if module.start_dim > 0 else module.start_dim
        )
        end_dim = module.end_dim - 1 if module.end_dim > 0 else module.end_dim
        return nir.Flatten(
            input_type=None,
            start_dim=start_dim,
            end_dim=end_dim,
        )

    else:
        print(f"[WARNING] module not implemented: {module.__class__.__name__}")
        return None


_NEURON_PARAM_ATTRS = (
    (
        nir.CubaLIF,
        (
            "tau_syn",
            "tau_mem",
            "r",
            "v_leak",
            "v_reset",
            "v_threshold",
            "w_in",
        ),
    ),
    (nir.LIF, ("tau", "r", "v_leak", "v_reset", "v_threshold")),
)


def _neuron_param_attrs(node):
    for node_type, attrs in _NEURON_PARAM_ATTRS:
        if isinstance(node, node_type):
            return attrs
    return None


def _broadcast_neuron_params(nir_graph, neuron_shapes):
    """Broadcast LIF/CubaLIF parameters to the population shape of the neuron.

    Scalar snn.Leaky/snn.Synaptic parameters become 0-d arrays on export.
    NIR derives input_type/output_type from those shapes, so infer_types()
    fails with e.g. ``[[128]] -> []`` (issues #410, #334). The population
    shape is the input shape of the neuron recorded while tracing the model,
    e.g. (N,) after a Linear layer or (C, H, W) after a Conv2d layer.
    """
    for name, node in nir_graph.nodes.items():
        attrs = _neuron_param_attrs(node)
        if attrs is None or name not in neuron_shapes:
            continue

        shape = neuron_shapes[name]
        for attr in attrs:
            value = getattr(node, attr, None)
            if value is not None:
                setattr(node, attr, np.broadcast_to(value, shape).copy())

        # __post_init__ already set types from the old parameter shapes.
        node.input_type = {"input": np.array(shape)}
        node.output_type = {"output": np.array(shape)}


def export_to_nir(
    module: torch.nn.Module,
    sample_data: torch.Tensor,
    model_name: str = "snntorch",
    model_fwd_args=[],
    ignore_dims=[],
) -> nir.NIRNode:
    """Convert an snnTorch module to the Neuromorphic Intermediate Representation (NIR).
    This function uses nirtorch to extract the computational graph of the torch module,
    and the _extract_snntorch_module method is used to convert each module in the graph
    to the corresponding NIR module.

    The NIR is a graph-based representation of a spiking neural network, which can be used to
    port the network to different neuromorphic hardware and software platforms.

    Example::

        import snntorch as snn
        import torch
        from snntorch.export_nir import export_to_nir

        lif1 = snn.Leaky(beta=0.9, init_hidden=True)
        lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

        net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(784, 500),
            lif1,
            torch.nn.Linear(500, 10),
            lif2
        )

        sample_data = torch.randn(1, 784)
        nir_graph = export_to_nir(net, sample_data)

    :param module: Network model (either wrapped in Sequential container or as a class)
    :type module: torch.nn.Module

    :param sample_data: Sample input data to the network
    :type sample_data: torch.Tensor

    :param model_name: Name of the model
    :type model_name: str, optional

    :param model_fwd_args: Arguments to pass to the forward function of the model
    :type model_fwd_args: list, optional

    :param ignore_dims: List of dimensions to ignore when extracting the NIR
    :type ignore_dims: list, optional

    :return: return the NIR graph
    :rtype: nir.NIRNode
    """
    # record the input shape of each Leaky/Synaptic neuron while nirtorch
    # traces the model; NIR node names follow the torch module names
    neuron_shapes = {}

    def record_input_shape(name):
        def hook(_module, args):
            shape = np.delete(np.array(args[0].shape), ignore_dims or [])
            neuron_shapes[name] = tuple(int(d) for d in shape)

        return hook

    hooks = [
        submodule.register_forward_pre_hook(record_input_shape(name))
        for name, submodule in module.named_modules()
        if isinstance(submodule, (snn.Leaky, snn.Synaptic))
    ]
    try:
        nir_graph = nirtorch.extract_nir_graph(
            module,
            _extract_snntorch_module,
            sample_data,
            model_name=model_name,
            ignore_submodules_of=[snn.RLeaky, snn.RSynaptic],
            model_fwd_args=model_fwd_args,
            ignore_dims=ignore_dims,
        )
    finally:
        for hook in hooks:
            hook.remove()

    # neuron parameters must match the population shape before type checks
    _broadcast_neuron_params(nir_graph, neuron_shapes)
    # ensure node input and output types are fully defined
    nir_graph.infer_types()
    return nir_graph
