import matplotlib.pyplot as plt
import seaborn as sns

prefix = 'plots/sq_'


import numpy as np
import nengo
from nengo.dists import Uniform
from nengo_detailed_neurons.neurons import Bahr2, IntFire1
from nengo_detailed_neurons.synapses import ExpSyn, FixedCurrent

# Create a 'model' object to which we can add ensembles, connections, etc.
model = nengo.Network(label="Communications Channel", seed=3145987)
with model:
    # Create an abstract input signal that oscillates as sin(t)
    sin = nengo.Node(lambda x: np.sin(x))

    # Create the neuronal ensembles
    num_pre_neurons = 200
    num_neurons = 500
    pre = nengo.Ensemble(
        num_pre_neurons, dimensions=1, max_rates=Uniform(60, 80))
    A = nengo.Ensemble(
        num_neurons, dimensions=1, neuron_type=Bahr2(),
        max_rates=Uniform(60, 80))
    B = nengo.Ensemble(
        num_neurons, dimensions=1, neuron_type=Bahr2(),
        max_rates=Uniform(60, 80))

    # Connect the input to the first neuronal ensemble
    nengo.Connection(sin, pre)

    # Connect the first neuronal ensemble to the second
    # (this is the communication channel)
    solver = nengo.solvers.LstsqL2(True)
    nengo.Connection(pre, A, solver=solver, synapse=ExpSyn(0.005))
    nengo.Connection(
        A, B, solver=solver, synapse=ExpSyn(0.005), function=lambda x: x * x)


with model:
    sin_probe = nengo.Probe(sin)
    A_probe = nengo.Probe(A, synapse=.01)  # ensemble output
    B_probe = nengo.Probe(B, synapse=.01)
    A_spikes = nengo.Probe(A.neurons, 'spikes')
    B_spikes = nengo.Probe(B.neurons, 'spikes')
    voltage = nengo.Probe(B.neurons, 'voltage')


sim = nengo.Simulator(model)
sim.run(2 * np.pi)

from nengo.utils.ensemble import tuning_curves_1d
plt.plot(*tuning_curves_1d(B, sim))
plt.xlabel("x")
plt.ylabel("Firing rate (1/s)")
plt.title("Approximate interpolated tuning curves")
plt.savefig(prefix + 'tuning_curves.pdf')


plt.figure(figsize=(9, 3))
plt.subplot(1, 3, 1)
plt.title("Input")
plt.plot(sim.trange(), sim.data[sin_probe])
plt.xlabel("t")
plt.xlim(0, 2 * np.pi)
plt.ylim(-1.2, 1.2)
plt.subplot(1, 3, 2)
plt.title("A ({} compartmental)".format(num_neurons))
plt.plot(sim.trange(), sim.data[A_probe])
plt.xlabel("t")
plt.gca().set_yticklabels([])
plt.xlim(0, 2 * np.pi)
plt.ylim(-1.2, 1.2)
plt.subplot(1, 3, 3)
plt.title("B ({} compartmental)".format(num_neurons))
plt.plot(sim.trange(), sim.data[B_probe])
plt.xlabel("t")
plt.gca().set_yticklabels([])
plt.xlim(0, 2 * np.pi)
plt.ylim(-1.2, 1.2)
plt.savefig(prefix + 'decoded.pdf')


dt = 1000.0
sns.set_style('white')
plt.figure(figsize=(18, 4))
plt.eventplot(
    [np.where(x)[0] / dt for x in sim.data[A_spikes].T[:50, :] if np.any(x)],
    colors=[(0, 0, 0, 1)], linewidth=1)
plt.title("Spike raster of A population (first 50 neurons)")
plt.xlabel("t")
plt.ylabel("Neuron index")
plt.xlim(0, 2 * np.pi)
plt.ylim(-0.5, 49.5)
plt.savefig(prefix + 'spikes_A.pdf')

dt = 1000.0
sns.set_style('white')
plt.figure(figsize=(18, 4))
spikes = [np.where(x)[0] / dt for x in sim.data[B_spikes].T if np.any(x)]
plt.eventplot(spikes, colors=[(0, 0, 0, 1)], linewidth=1)
plt.title("Spike raster of B population")
plt.xlabel("t")
plt.ylabel("Neuron index")
plt.xlim(0, 2 * np.pi)
plt.ylim(-0.5, len(spikes) - 0.5)
plt.savefig(prefix + 'spikes_B.pdf')

sns.set_style('darkgrid')
plt.title("Voltage trace (soma)")
plt.xlabel("t (s)")
plt.ylabel("Membrane voltage (mV)")
plt.plot(sim.trange(), sim.data[voltage][:, 12])
plt.xlim(0, 2 * np.pi)
plt.savefig(prefix + 'voltage.pdf')
