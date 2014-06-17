"""Provides some basic Neuron neuron models."""

from collections import namedtuple
from weakref import WeakKeyDictionary

# FIXME using non-public Nengo API
from nengo.neurons import _LIFBase, NeuronType
import neuron
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


class IntFire1(_LIFBase, NeuronType):
    Cell = namedtuple('Cell', ['neuron', 'in_con', 'out_con'])

    class Cell(object):
        __slots__ = ['neuron', 'in_con', 'out_con', 'spiketime']

        def __init__(self, tau_rc, tau_ref):
            self.neuron = neuron.h.IntFire1()
            self.neuron.tau = _nrn_duration(tau_rc)
            self.neuron.refrac = _nrn_duration(tau_ref)
            self.in_con = neuron.h.NetCon(None, self.neuron)
            self.out_con = neuron.h.NetCon(self.neuron, None)
            self.spiketime = 0.0

    def create(self):
        return self.Cell(self.tau_rc, self.tau_ref)

    def step_math(self, dt, J, spiked, cells, voltage):
        # 1. Add J to current c.i
        dV = (dt / self.tau_rc) * J
        for change, j, c in zip(dV, J, cells):
            if c.neuron.m <= 1.0 and c.spiketime > 0.0:
                change += c.spiketime * j / _nrn_duration(self.tau_rc)
                c.spiketime = 0.0
            c.in_con.weight[0] = change
            c.in_con.event(neuron.h.t + _nrn_duration(dt) / 2.0)
        # 2. Setup recording of spikes
        spikes = [neuron.h.Vector() for c in cells]
        for c, s in zip(cells, spikes):
            c.out_con.record(neuron.h.ref(s))
        # 3. Simulate for one time step
        neuron.run(neuron.h.t + _nrn_duration(dt))
        # 4. check for spikes
        spiked[:] = [s.size() > 0 for s in spikes]
        voltage[:] = [min(max(c.neuron.M(), 0), 1) for c in cells]

        for c, s in zip(cells, spikes):
            if s.size() > 0:
                c.spiketime = neuron.h.t - s[0]
                c.neuron.refrac = _nrn_duration(
                    self.tau_ref + dt) - c.spiketime


# FIXME: Deriving from _LIFBase for now to have some default implementation
# for bias, gain, and tuning curve calculation. Obviously, this does not match
# the neuron implemented here.
class Compartmental(_LIFBase, NeuronType):
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

        return (cell, out_con)

    def step_math(self, dt, J, spiked, cells, voltage):
        # 1. Setup recording of spikes
        spikes = [neuron.h.Vector() for c in cells]
        for (_, out_con), s in zip(cells, spikes):
            out_con.record(neuron.h.ref(s))
        # 2. Simulate for one time step
        neuron.run(neuron.h.t + dt * 1000)
        # 3. check for spikes
        spiked[:] = [s.size() > 0 for s in spikes]
        voltage[:] = [c.v for (c, _) in cells]
