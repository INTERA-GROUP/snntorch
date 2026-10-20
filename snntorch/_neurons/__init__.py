###############################################################
# When adding new neurons, update the following:              #
# i) create neuron in snntorch/_neurons/your_neuron.py        #
# ii) utils.py: reset(); _layer_check(), _layer_reset() etc.  #
# iii) neurons_dict in backprop.py,                           #
# iv) update __neuron__ & import below                        #
# v) unit tests in tests/test_snntorch.py                     #
# vi) update docs: snntorch.rst                               #
###############################################################

__neuron__ = [
    "alpha",
    "deltaleaky",
    "lapicque",
    "leaky",
    "linearleaky",
    "stateleaky",
    "leakyparallel",
    "rleaky",
    "rsynaptic",
    "synaptic",
    "customneuron",
    "synapticdelay",
    "sconv2dlstm",
    "slstm",
    "gen2",
]

from .alpha import Alpha
from .associative import AssociativeLeaky
from .customneuron import CustomNeuron, neuron_from_equations
from .deltaleaky import DeltaLeaky
from .lapicque import Lapicque
from .leaky import Leaky
from .leakyparallel import LeakyParallel
from .linearleaky import LinearLeaky
from .neurons import LIF, SpikingNeuron
from .rleaky import RLeaky
from .rsynaptic import RSynaptic
from .sconv2dlstm import SConv2dLSTM
from .slstm import SLSTM
from .stateleaky import StateLeaky
from .synaptic import Synaptic
from .synapticdelay import SynapticDelay
