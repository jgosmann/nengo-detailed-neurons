import nengo
from nengo.builder import build_connection, build_linear_system, \
    full_transform, Builder, Operator, Signal
from nengo.objects import Connection
import neuron
import numpy as np

from nrnengo.neurons import Compartmental, IntFire1


class SimNrnNeurons(Operator):
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
    def __init__(self, spikes, stim, synapses):
        self.spikes = spikes
        self.stim = stim
        self.synapses = synapses

        self.reads = [spikes]
        self.updates = []
        self.sets = []
        self.incs = []

        neuron.init()

    def make_step(self, signals, dt):
        spikes = signals[self.spikes]

        def step():
            for i, (spike, s) in enumerate(zip(spikes, self.stim)):
                if spike > 0:
                    for syn in s:
                        syn.event(neuron.h.t)
        return step


class NrnBuilders(object):
    def __init__(self):
        self.ens_to_cells = {}  # FIXME use dict with weak keys

    def build_nrn_neuron(self, nrn, ens, model, config):
        model.sig[ens]['voltage'] = Signal(
            np.zeros(ens.n_neurons), name="%s.voltage" % ens.label)
        op = SimNrnNeurons(
            neurons=nrn,
            J=model.sig[ens]['neuron_in'],
            output=model.sig[ens]['neuron_out'],
            voltage=model.sig[ens]['voltage'])
        self.ens_to_cells[ens] = op.cells
        model.add_op(op)

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

        if isinstance(conn.pre, nengo.objects.Neurons):
            model.sig[conn]['in'] = model.sig[conn.pre.ensemble]['neuron_out']
        else:
            model.sig[conn]['in'] = model.sig[conn.pre]["out"]

        if isinstance(conn.post, nengo.objects.Neurons):
            raise NotImplementedError()  # TODO
        elif isinstance(conn.post.neuron_type, Compartmental):
            pass
        else:
            raise AssertionError(
                "This function should only be called if post neurons are "
                "compartmental.")

        assert isinstance(conn.pre, nengo.objects.Ensemble)
        assert not isinstance(conn.pre.neuron_type, nengo.neurons.Direct)
        # FIXME assert no rate neurons are used. How to do that?

        # FIXME just assuming solver is a weight solver, may that break?
        # Default solver should probably also produce sparse solutions for
        # performance reasons
        eval_points, activities, targets = build_linear_system(
            conn, model, rng=rng)

        # account for transform
        transform = full_transform(conn)
        targets = np.dot(targets, transform.T)
        transform = np.array(1., dtype=np.float64)

        weights, solver_info = conn.solver(
            activities, targets, rng=rng,
            E=model.params[conn.post].scaled_encoders.T)

        stim = [[]] * len(weights)
        synapses = [[]]
        for i, (cell, _) in enumerate(self.ens_to_cells[conn.post]):
            for j, w in enumerate(weights[:, i]):
                # 1. Add synapse with corresponding weight
                syn = neuron.h.ExpSyn(cell(0.5))  # TODO position on neuron
                syn.tau = conn.synapse * 1000  # FIXME assumes exponential
                                               # synapse
                # FIXME set synapse parameters
                if w > 0:
                    syn.e = 10
                else:
                    syn.e = -70
                    #syn.e = 10

                nc = neuron.h.NetCon(None, syn)
                nc.weight[0] = abs(w) / 100.0
                #nc.weight[0] = w

                # 2. Gather list of synapses corresponding to pre neurons
                stim[j].append(nc)
                #stim[j].append(syn)
                synapses.append(syn)

        # 3. Add operator creating events for synapses if pre neuron fired
        model.add_op(NrnTransmitSpikes(model.sig[conn]['in'], stim, synapses))

    def register(self):
        Builder.register_builder(self.build_nrn_neuron, IntFire1)
        Builder.register_builder(self.build_nrn_neuron, Compartmental)
        Builder.register_builder(self.build_connection, Connection)
