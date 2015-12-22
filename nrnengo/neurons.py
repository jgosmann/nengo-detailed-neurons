"""Provides some basic Neuron neuron models."""

from collections import namedtuple
from weakref import WeakKeyDictionary

# FIXME using non-public Nengo API
from nengo.neurons import LIFRate as _LIFBase, NeuronType
import neuron
import numpy as np
import nrn

from nrnengo.util import nrn_duration


class NrnNeuron(NeuronType):
    """Marks neurons for simulation in Neuron."""

    def create(self):
        """Creates the required Neuron objects to simulate the neuron and
        returns them."""
        raise NotImplementedError()

    def _setup_spike_recorder(self, cells):
        spikes = [neuron.h.Vector() for c in cells]
        for c, s in zip(cells, spikes):
            c.out_con.record(neuron.h.ref(s))
        return spikes


class Compartmental(NrnNeuron):
    pass


class IntFire1(_LIFBase, NrnNeuron):
    """IntFire1 neuron model of Neuron simulator."""

    class Cell(object):
        """Individual IntFire1 cell."""

        __slots__ = ['neuron', 'in_con', 'out_con', 'spiketime']

        def __init__(self, tau_rc, tau_ref):
            self.neuron = neuron.h.IntFire1()
            self.neuron.tau = nrn_duration(tau_rc)
            self.neuron.refrac = nrn_duration(tau_ref)
            self.in_con = neuron.h.NetCon(None, self.neuron)
            self.out_con = neuron.h.NetCon(self.neuron, None)
            self.spiketime = 0.0

        @property
        def refractory(self):
            return self.neuron.m > 1.0

    def create(self):
        return self.Cell(self.tau_rc, self.tau_ref)

    def step_math(self, dt, J, spiked, cells, voltage):
        # 1. Determine voltage changes
        dV = (dt / self.tau_rc) * J

        spiketimes = np.array(
            [c.spiketime if not c.refractory else 0.0 for c in cells])
        dV += spiketimes * J / nrn_duration(self.tau_rc)

        # 2. Apply voltage changes
        for c, w in zip(cells, dV):
            if not c.refractory:
                c.spiketime = 0.0
            c.in_con.weight[0] = w
            c.in_con.event(neuron.h.t + nrn_duration(dt) / 2.0)

        # 3. Setup recording of spikes
        spikes = self._setup_spike_recorder(cells)

        # 4. Simulate for one time step
        neuron.run(neuron.h.t + nrn_duration(dt))

        # 5. Check for spikes and record voltages
        spiked[:] = [s.size() > 0 for s in spikes]
        spiked /= dt
        voltage[:] = [np.clip(c.neuron.M(), 0, 1) for c in cells]

        # 6. Record spike times
        for idx in np.where(spiked)[0]:
            cells[idx].spiketime = neuron.h.t - spikes[idx][0]
            cells[idx].neuron.refrac = nrn_duration(
                self.tau_ref + dt) - cells[idx].spiketime


class Bahr2(Compartmental):
    probeable = ['spikes', 'voltage']
    Cell = namedtuple('Cell', ['neuron', 'bias', 'spikes', 'out_con'])
    # FIXME hard coded path
    rate_table = np.load(
        '/home/jgosmann/Documents/projects/summerschool2014/neuron-models/'
        'data/bahl2_response_curve.npz')

    def __init__(self):
        super(Bahr2, self).__init__()
        # FIXME hard coded path
        model_path = '/home/jgosmann/Documents/projects/summerschool2014/' \
            'neuron-models/models/bahr2.hoc'
        neuron.h.load_file(model_path)

    def create(self):
        cell = neuron.h.Bahr2()
        bias = neuron.h.IClamp(cell.soma(0.5))
        bias.delay = 0
        bias.dur = 1e9  # FIXME limits simulation time
        ap_counter = neuron.h.APCount(cell.soma(0.5))
        spikes = neuron.h.Vector()
        ap_counter.record(neuron.h.ref(spikes))
        return self.Cell(
            neuron=cell, bias=bias, spikes=spikes, out_con=ap_counter)

    def rates_from_current(self, J):
        return np.interp(
            J, self.rate_table['current'], self.rate_table['rate'])

    def rates(self, x, gain, bias):
        J = gain * x + bias
        return self.rates_from_current(J)

    def gain_bias(self, max_rates, intercepts):
        intercepts = np.asarray(intercepts)
        max_rates = np.minimum(max_rates, self.rate_table['rate'].max())

        min_j = self.rate_table['current'][np.argmax(
            self.rate_table['rate'] > 1)]
        max_j = self.rate_table['current'][np.argmax(
            np.atleast_2d(self.rate_table['rate']).T >= max_rates, axis=0)]

        gain = (min_j - max_j) / (intercepts - 1.0)
        bias = min_j - gain * intercepts
        return gain, bias

    def step_math(self, dt, J, spiked, cells, voltage):
        for c in cells:
            c.spikes.resize(0)

        # 1. Simulate for one time step
        neuron.run(neuron.h.t + nrn_duration(dt))

        # 2. Check for spikes
        spiked[:] = [c.spikes.size() > 0 for c in cells]
        spiked /= dt
        voltage[:] = [c.neuron.soma.v for c in cells]
