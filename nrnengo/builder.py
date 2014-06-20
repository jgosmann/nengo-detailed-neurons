from weakref import WeakKeyDictionary

import nengo
from nengo.builder import build_connection, build_ensemble, \
    build_linear_system, full_transform, Builder, Operator, Signal
from nengo.objects import Connection, Ensemble
from nengo.utils.compat import is_number
import neuron
import numpy as np

from nrnengo.neurons import Bahr2, Compartmental, IntFire1
from nrnengo.synapses import ExpSyn


class SimNrnPointNeurons(Operator):
    def __init__(self, neurons, J, output, voltage):
        self.neurons = neurons
        self.J = J
        self.output = output
        self.voltage = voltage

        self.reads = [J]
        self.updates = [output, voltage]
        self.sets = []
        self.incs = []

        self.cells = [self.neurons.create() for i in range(len(self.J))]
        neuron.init()

    def make_step(self, signals, dt):
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

    def make_step(self, signals, dt):
        spikes = signals[self.spikes]

        def step():
            for idx in np.where(spikes)[0]:
                for synaptic_con in self.connections[idx]:
                    synaptic_con.in_con.event(neuron.h.t)
        return step


class NrnBuilders(object):
    def __init__(self):
        self.ens_to_cells = WeakKeyDictionary()

    def build_nrn_neuron(self, nrn, ens, model, config):
        model.sig[ens]['voltage'] = Signal(
            np.zeros(ens.n_neurons), name="%s.voltage" % ens.label)
        op = SimNrnPointNeurons(
            neurons=nrn,
            J=model.sig[ens]['neuron_in'],
            output=model.sig[ens]['neuron_out'],
            voltage=model.sig[ens]['voltage'])
        self.ens_to_cells[ens] = op.cells
        model.add_op(op)

    def build_ensemble(self, ens, model, config):
        build_ensemble(ens, model, config)
        if isinstance(ens.neuron_type, Compartmental):
            for c, b in zip(self.ens_to_cells[ens], model.params[ens].bias):
                c.bias.amp = b

    def build_connection(self, conn, model, config):
        use_nrn = (
            isinstance(conn.post, nengo.objects.Ensemble) and
            isinstance(conn.post.neuron_type, Compartmental))
        if use_nrn:
            return self.build_nrn_connection(conn, model, config)
        else:
            return build_connection(conn, model, config)

    def build_nrn_connection(self, conn, model, config):
        # Create random number generator
        rng = np.random.RandomState(model.seeds[conn])

        # Check pre-conditions
        assert isinstance(conn.pre, nengo.objects.Ensemble)
        assert not isinstance(conn.pre.neuron_type, nengo.neurons.Direct)
        # FIXME assert no rate neurons are used. How to do that?

        # Get input signal
        if isinstance(conn.pre, nengo.objects.Neurons):
            model.sig[conn]['in'] = model.sig[conn.pre.ensemble]['neuron_out']
        else:
            model.sig[conn]['in'] = model.sig[conn.pre]["out"]

        # Figure out type of connection
        if isinstance(conn.post, nengo.objects.Neurons):
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
            conn, model, rng=rng)

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
        weights = weights * 0.001 / len(weights)
        connections = [[]] * len(weights)
        for i, cell in enumerate(self.ens_to_cells[conn.post]):
            for j, w in enumerate(weights[:, i]):
                if w >= 0.0:
                    x = np.random.rand()
                    connections[j].append(synapse.create(
                        cell.neuron.soma(x),
                        w * (x + 1)))
                else:
                    connections[j].append(synapse.create(
                        cell.neuron.soma(0.5), w))

        # 3. Add operator creating events for synapses if pre neuron fired
        model.add_op(NrnTransmitSpikes(model.sig[conn]['in'], connections))

    def register(self):
        Builder.register_builder(self.build_nrn_neuron, IntFire1)
        Builder.register_builder(self.build_nrn_neuron, Bahr2)
        Builder.register_builder(self.build_ensemble, Ensemble)
        Builder.register_builder(self.build_connection, Connection)
