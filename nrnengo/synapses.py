from collections import namedtuple

from nengo.synapses import Synapse
import neuron

from nrnengo.util import nrn_duration


class NrnSynapse(Synapse):
    def create(self, sec, weight):
        raise NotImplementedError()


class ExpSyn(NrnSynapse):
    SynapticCon = namedtuple('SynapticCon', ['syn', 'in_con'])

    def __init__(self, tau, e_exc=0.0, e_inh=-70.0):
        self.tau = tau
        self.e_exc = e_exc
        self.e_inh = e_inh

    def create(self, sec, weight):
        syn = neuron.h.ExpSyn(sec)
        syn.tau = nrn_duration(self.tau)
        if weight >= 0.0:
            syn.e = self.e_exc
        else:
            syn.e = self.e_inh

        in_con = neuron.h.NetCon(None, syn)
        in_con.weight[0] = abs(weight)

        return self.SynapticCon(syn=syn, in_con=in_con)


class FixedCurrent(NrnSynapse):
    SynapticCon = namedtuple('SynapticCon', ['syn', 'in_con'])

    def __init__(self, tau):
        self.tau = tau
        neuron.h.nrn_load_dll(
            '/home/jgosmann/Documents/projects/summerschool2014/neuron-models/'
            'models/x86_64/.libs/libnrnmech.so')

    def create(self, sec, weight):
        syn = neuron.h.FixedCurrent(sec)
        syn.tau = nrn_duration(self.tau)

        in_con = neuron.h.NetCon(None, syn)
        in_con.weight[0] = weight

        return self.SynapticCon(syn=syn, in_con=in_con)
