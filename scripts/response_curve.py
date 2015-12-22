"""Script for generating the F-I-curve for stimulation in the soma and store
the data as plot and npz file for use in the nrnengo module."""

import neuron
import numpy as np

neuron.h.load_file('models/bahr2.hoc')

neuron.h.dt = 0.005

cell = neuron.h.Bahr2()

iclamp = neuron.h.IClamp(cell.soma(0.5))
iclamp.delay = 200
iclamp.dur = 1100

v = neuron.h.Vector()
v.record(cell.soma(0.5)._ref_v)

apcount = neuron.h.APCount(cell.soma(0.5))
spikes = neuron.h.Vector()
apcount.record(neuron.h.ref(spikes))

current = np.linspace(0.0, 3.0, 100)
freq = []

for i in current:
    iclamp.amp = i
    neuron.init()
    neuron.run(1500)
    spike_array = np.array(spikes)
    freq.append(np.sum(spike_array > 300))

np.savez('data/bahl2_response_curve.npz', current=current, rate=freq)

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
plt.plot(current, freq)
plt.xlabel("Injected current (nA, at soma)")
plt.ylabel("Firing rate (1/s)")
plt.savefig('plots/response_curve.pdf')
