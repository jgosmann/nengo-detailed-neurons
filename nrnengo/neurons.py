"""Provides some basic Neuron neuron models."""

from collections import namedtuple

# FIXME using non-public Nengo API
from nengo.neurons import _LIFBase, NeuronType
import neuron
import numpy as np
import nrn


def _nrn_duration(t):
    """Converts a duration in native Nengo units to native Neuron units."""
    # Nengo uses seconds
    # Neuron uses milliseconds
    return 1000 * t


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


class IntFire1(_LIFBase, NrnNeuron):
    """IntFire1 neuron model of Neuron simulator."""

    class Cell(object):
        """Individual IntFire1 cell."""

        __slots__ = ['neuron', 'in_con', 'out_con', 'spiketime']

        def __init__(self, tau_rc, tau_ref):
            self.neuron = neuron.h.IntFire1()
            self.neuron.tau = _nrn_duration(tau_rc)
            self.neuron.refrac = _nrn_duration(tau_ref)
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
        dV += spiketimes * J / _nrn_duration(self.tau_rc)

        # 2. Apply voltage changes
        for c, w in zip(cells, dV):
            if not c.refractory:
                c.spiketime = 0.0
            c.in_con.weight[0] = w
            c.in_con.event(neuron.h.t + _nrn_duration(dt) / 2.0)

        # 3. Setup recording of spikes
        spikes = self._setup_spike_recorder(cells)

        # 4. Simulate for one time step
        neuron.run(neuron.h.t + _nrn_duration(dt))

        # 5. Check for spikes and record voltages
        spiked[:] = [s.size() > 0 for s in spikes]
        voltage[:] = [np.clip(c.neuron.M(), 0, 1) for c in cells]

        # 6. Record spike times
        for idx in np.where(spiked)[0]:
            cells[idx].spiketime = neuron.h.t - spikes[idx][0]
            cells[idx].neuron.refrac = _nrn_duration(
                self.tau_ref + dt) - cells[idx].spiketime


# FIXME: Deriving from _LIFBase for now to have some default implementation
# for bias, gain, and tuning curve calculation. Obviously, this does not match
# the neuron implemented here.
class Compartmental(_LIFBase, NrnNeuron):
    Cell = namedtuple('Cell', ['neuron', 'out_con'])

    def create(self):
        # TODO replace this with an actual, biological plausible neuron model
        # Even better: Allow to use different Neuron neuron models.
        cell = nrn.Section()
        cell.nseg = 1
        cell.diam = 18.8
        cell.L = 18.8
        cell.Ra = 123.0
        cell.insert('hh')
        cell.gnabar_hh = 0.25
        cell.gl_hh = 0.0001666
        cell.el_hh = -60.0

        out_con = neuron.h.APCount(cell(0.5))

        return self.Cell(neuron=cell, out_con=out_con)

    def step_math(self, dt, J, spiked, cells, voltage):
        # 1. Setup recording of spikes
        spikes = self._setup_spike_recorder(cells)

        # 2. Simulate for one time step
        neuron.run(neuron.h.t + _nrn_duration(dt))

        # 3. Check for spikes
        spiked[:] = [s.size() > 0 for s in spikes]
        voltage[:] = [c.neuron.v for c in cells]
