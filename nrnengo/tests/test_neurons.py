import nengo
from nengo.decoders import LstsqL2
import numpy as np
from numpy.testing import assert_allclose
import pytest

from nrnengo.neurons import Compartmental, IntFire1


def test_int_fire1_behaves_like_nengo_lif():
    model = nengo.Network(seed=1337, label="Communications Channel")
    with model:
        sin = nengo.Node(np.sin)

        # First ensemble is needed to convert input to spikes. Otherwise the
        # Neuron neurons cannot be connected currently.
        A = nengo.Ensemble(50, dimensions=1)
        B = nengo.Ensemble(50, dimensions=1, neuron_type=IntFire1())
        C = nengo.Ensemble(50, dimensions=1)

        nengo.Connection(sin, A)
        nengo.Connection(A, B)
        nengo.Connection(A, C)

        B_probe = nengo.Probe(B, synapse=.1)
        C_probe = nengo.Probe(C, synapse=.1)

    sim = nengo.Simulator(model)
    sim.run(2 * np.pi)

    assert_allclose(sim.data[B_probe], sim.data[C_probe], 0., 0.1)


def test_compartmental():
    model = nengo.Network(seed=1337, label="Communications Channel")
    with model:
        sin = nengo.Node(np.sin)

        # First ensemble is needed to convert input to spikes. Otherwise the
        # Neuron neurons cannot be connected currently.
        A = nengo.Ensemble(50, dimensions=1)
        B = nengo.Ensemble(50, dimensions=1, neuron_type=Compartmental())

        nengo.Connection(sin, A)
        nengo.Connection(A, B, solver=LstsqL2(weights=True))

        B_probe = nengo.Probe(B, synapse=.1)
        voltage_probe = nengo.Probe(B, 'voltage')

    sim = nengo.Simulator(model)
    sim.run(1.0)

    # TODO So far this test only checks that a model with compartmental neurons
    # can be build and simulated. It should also check that it produces sane
    # results. However, the current implementation is not capable of doing so,
    # yet.


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
