"""Script for checking the linearizing heuristic along the dendrite."""

import neuron
import numpy as np

neuron.h.load_file('models/bahr2.hoc')

cell = neuron.h.Bahr2()
cell.apical.nseg = 20

base_weight = 0.02

xs = np.linspace(0.0, 1.0)
ys_baseline = []
ys_lin = []
for x in xs:
    syn = neuron.h.ExpSyn(cell.apical(x))
    syn.e = 0
    syn.tau = 5

    v = neuron.h.Vector()
    v.record(cell.soma(0.5)._ref_v)

    nc = neuron.h.NetCon(None, syn)

    nc.weight[0] = 0.02
    neuron.init()
    nc.event(200)
    neuron.run(1500)
    ys_baseline.append(np.max(v))

    nc.weight[0] = 0.02 * (x + 1.0)
    neuron.init()
    nc.event(200)
    neuron.run(1500)
    ys_lin.append(np.max(v))

import matplotlib.pyplot as plt
plt.plot(xs, ys_baseline, label="Baseline")
plt.plot(xs, ys_lin, label="Linearization heuristic")
plt.title("Voltage peak in soma after dendritic stimulation")
plt.xlabel("Position on dendrite")
plt.ylabel("Voltage")
plt.legend(loc='best')
plt.savefig('plots/dendrite.pdf')
