from weakref import WeakKeyDictionary

import nengo
import nengo.builder
from nengo.builder.builder import Builder
from nengo.builder.connection import build_linear_system
from nengo.builder.operator import Operator
from nengo.builder.signal import Signal
from nengo import Connection, Ensemble
from nengo.utils.builder import full_transform
from nengo.utils.compat import is_number
import neuron
import numpy as np

from nengo_detailed_neurons.neurons import Bahr2, Compartmental, IntFire1
from nengo_detailed_neurons.synapses import ExpSyn


class SimNrnPointNeurons(Operator):
    def __init__(self, neurons, J, output, voltage):
        self.neurons = neurons
        self.J = J
        self.output = output
        self.voltage = voltage

        self.reads = [J]
        self.sets = [output, voltage]
        self.updates = []
        self.incs = []

        self.cells = [self.neurons.create() for i in range(self.J.shape[0])]
        neuron.init()

    def make_step(self, signals, dt, rng):
        J = signals[self.J]
        output = signals[self.output]
        voltage = signals[self.voltage]

        def step():
            self.neurons.step_math(dt, J, output, self.cells, voltage)
        return step


class NrnTransmitSpikes(Operator):
    def __init__(self, spikes, connections):
        self.spikes = spikes

        self.connections = connections

        self.reads = [spikes]
        self.updates = []
        self.sets = []
        self.incs = []

        neuron.init()

    def make_step(self, signals, dt, rng):
        spikes = signals[self.spikes]

        def step():
            for idx in np.where(spikes)[0]:
                for synaptic_con in self.connections[idx]:
                    synaptic_con.in_con.event(neuron.h.t)
        return step


ens_to_cells = WeakKeyDictionary()

@Builder.register(IntFire1)
@Builder.register(Bahr2)
def build_nrn_neuron(model, nrn, ens):
    model.sig[ens]['voltage'] = Signal(
        np.zeros(ens.ensemble.n_neurons),
        name="%s.voltage" % ens.ensemble.label)
    op = SimNrnPointNeurons(
        neurons=nrn,
        J=model.sig[ens]['in'],
        output=model.sig[ens]['out'],
        voltage=model.sig[ens]['voltage'])
    ens_to_cells[ens.ensemble] = op.cells
    model.add_op(op)

@Builder.register(Ensemble)
def build_ensemble(model, ens):
    nengo.builder.ensemble.build_ensemble(model, ens)
    if isinstance(ens.neuron_type, Compartmental):
        for c, b in zip(ens_to_cells[ens], model.params[ens].bias):
            c.bias.amp = b

@Builder.register(Connection)
def build_connection(model, conn):
    use_nrn = (
        isinstance(conn.post, nengo.Ensemble) and
        isinstance(conn.post.neuron_type, Compartmental))
    if use_nrn:
        return build_nrn_connection(model, conn)
    else:
        return nengo.builder.connection.build_connection(model, conn)

def build_nrn_connection(model, conn):
    # Create random number generator
    rng = np.random.RandomState(model.seeds[conn])

    # Check pre-conditions
    assert isinstance(conn.pre, nengo.Ensemble)
    assert not isinstance(conn.pre.neuron_type, nengo.neurons.Direct)
    # FIXME assert no rate neurons are used. How to do that?

    # Get input signal
    # FIXME this should probably be
    # model.sig[conn]['in'] = model.sig[conn.pre]["out"]
    # in both cases
    if isinstance(conn.pre, nengo.ensemble.Neurons):
        model.sig[conn]['in'] = model.sig[conn.pre.ensemble]['out']
    else:
        model.sig[conn]['in'] = model.sig[conn.pre]["out"]

    # Figure out type of connection
    if isinstance(conn.post, nengo.ensemble.Neurons):
        raise NotImplementedError()  # TODO
    elif isinstance(conn.post.neuron_type, Compartmental):
        pass
    else:
        raise AssertionError(
            "This function should only be called if post neurons are "
            "compartmental.")

    # Solve for weights
    # FIXME just assuming solver is a weight solver, may that break?
    # Default solver should probably also produce sparse solutions for
    # performance reasons
    eval_points, activities, targets = build_linear_system(
        model, conn, rng=rng)

    # account for transform
    transform = full_transform(conn)
    targets = np.dot(targets, transform.T)

    weights, solver_info = conn.solver(
        activities, targets, rng=rng,
        E=model.params[conn.post].scaled_encoders.T)

    # Synapse type
    synapse = conn.synapse
    if is_number(synapse):
        synapse = ExpSyn(synapse)

    # Connect
    # TODO: Why is this adjustment of the weights necessary?
    weights = weights / synapse.tau / 5. * .1
    connections = [[] for i in range(len(weights))]
    for i, cell in enumerate(ens_to_cells[conn.post]):
        for j, w in enumerate(weights[:, i]):
            if w >= 0.0:
                x = np.random.rand()
                connections[j].append(synapse.create(
                    cell.neuron.apical(x),
                    w * (x + 1)))
            else:
                connections[j].append(synapse.create(
                    cell.neuron.soma(0.5), w))

    # 3. Add operator creating events for synapses if pre neuron fired
    model.add_op(NrnTransmitSpikes(model.sig[conn]['in'], connections))
