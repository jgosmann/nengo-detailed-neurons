"""Script for checking minimum and maximum required synaptic weights."""

import neuron
import numpy as np

neuron.h.load_file('models/bahr2.hoc')

cell = neuron.h.Bahr2()

syn = neuron.h.ExpSyn(cell.soma(0.5))
syn.e = 0
syn.tau = 5

nc = neuron.h.NetCon(None, syn)

v = neuron.h.Vector()
v.record(cell.soma(0.5)._ref_v)

apcount = neuron.h.APCount(cell.soma(0.5))
spikes = neuron.h.Vector()
apcount.record(neuron.h.ref(spikes))

freqs = np.linspace(10, 100, 9)
weights = np.linspace(0.01, 0.05, 100)
lower = []
upper = []

for f in freqs:
    outfreqs = []
    for w in weights:
        nc.weight[0] = w
        neuron.init()
        for i in range(int(f)):
            nc.event(200 + i * 1000.0 / f)
        neuron.run(1500)
        outfreqs.append(len(spikes))
    outfreqs = np.array(outfreqs)
    lower.append(weights[np.where(outfreqs > 1)[0][0]])
    upper.append(weights[np.where(outfreqs >= f)[0][0]])


import matplotlib.pyplot as plt
plt.plot(freqs, lower)
plt.plot(freqs, upper)
plt.savefig('plots/synfreq.pdf')
