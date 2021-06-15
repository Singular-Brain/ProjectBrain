from abc import ABC
import torch
import numpy as np
from ..networks import NeuronGroup, Connection
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class LearningRule(ABC):
    def __init__(self):
        ...

    def set_params(self, dt, group):
        ...

    def update_neuromodulators(self):
        ...

    def zero_released_neuromodulators(self):
        ...

    def __call__(self,):
        ...


class STDP(LearningRule):
    def __init__(self,
                LTP_rate = 0.188, 
                LTD_rate = -0.094, 
                tau_LTP = 0.02,   
                tau_LTD = 0.04, 
                tau_dopamine = 0.2, tau_gaba = 0.2,  
                dopamine_base = 0.002, gaba_base = 0,
                tau_eligibility = 1, min_weight_value = 1E-5, 
                inhibitory_hardbound = None, excitatory_hardbound = None,
                plastic_inhibitory = True):
        """Spike-Timing-Dependent Plasticity learning rule
            Default values for `LTP_rate`, `LTD_rate`, `tau_LTP`, and `tau_LTD` are set based on: 
            Nicolas Frémaux, Henning Sprekeler and Wulfram Gerstner, Functional Requirements for Reward-Modulated Spike-Timing-Dependent Plasticity

        Args:
            LTP_rate (float, optional): [description]. Defaults to 0.188.
            LTD_rate (float, optional): [description]. Defaults to -0.094.
            tau_LTP (float, optional): [description]. Defaults to 0.02.
            tau_LTD (float, optional): [description]. Defaults to 0.04.
            dopamine_base (float, optional): [description]. Defaults to 0.002.
            tau_eligibility (int, optional): [description]. Defaults to 1.
            tau_dopamine (float, optional): [description]. Defaults to 0.2.
            hard_bound (tuple, optional): [description]. Defaults to None.
            plastic_inhibitory (bool, optional): [description]. Defaults to True.

        Raises:
            ValueError: If 'min_weight_value' is not more than 0
        """
        self.tau_eligibility = tau_eligibility
        self.tau_dopamine = tau_dopamine
        self.dopamine = 0
        self.tau_gaba = tau_gaba
        self.gaba = 0
        self.dopamine_base = dopamine_base
        self.gaba_base = gaba_base
        self.released_dopamine = 0
        self.released_gaba = 0
        self.M = 0 #Neuromodulator
        self.LTP_rate = LTP_rate
        self.LTD_rate = LTD_rate
        self.tau_LTP = tau_LTP
        self.tau_LTD = tau_LTD
        self.plastic_inhibitory = plastic_inhibitory
        self.excitatory_hardbound = excitatory_hardbound
        self.inhibitory_hardbound = inhibitory_hardbound
        self.min_weight_value = min_weight_value
        if not self.min_weight_value > 0:
            raise ValueError("'min_weight_value' must be more than 0")

    def set_params(self, dt, group):
        self.dt = dt
        group.STDP = torch.zeros_like(group.weights, dtype = torch.float32, device=DEVICE)
        group.eligibility_trace = torch.zeros_like(group.weights, dtype = torch.float32, device = DEVICE)
        if isinstance(group, NeuronGroup):
            group.STDP_trace = torch.zeros((group.N,1), device = DEVICE)
            group.decay_rate = (group.excitatory_neurons * (np.exp(-self.dt/self.tau_LTP)) + group.inhibitory_neurons *  (np.exp(-self.dt/self.tau_LTD))).reshape((group.N, 1))

    def _update_STDP(self, group):
        # reset STDP
        group.STDP.fill_(0)
        if isinstance(group, NeuronGroup):
            ### post-pre spikes (= pre-post connections)
            group.STDP[group.spikes,:] = self.LTD_rate * group.adjacency_matrix[group.spikes,:] * group.STDP_trace.T
            ### pre-post spikes (= post-pre connections)
            group.STDP[:,group.spikes] = self.LTP_rate * group.adjacency_matrix[:,group.spikes] * group.STDP_trace 
            #TODO: handle simultaneous pre-post spikes
        elif isinstance(group, Connection):
            ### post-pre spikes (= pre-post connections)
            group.STDP[group.source.spikes,:] = self.LTD_rate * group.adjacency_matrix[group.source.spikes,:] * group.destination.STDP_trace.T
            ### pre-post spikes (= post-pre connections)
            group.STDP[:,group.destination.spikes]   = self.LTP_rate * group.adjacency_matrix[:,group.destination.spikes]   * group.source.STDP_trace 

    def update_neuromodulators(self):
        self.dopamine += (-self.dopamine/self.tau_dopamine ) * self.dt + self.released_dopamine
        self.gaba += (-self.gaba/self.tau_gaba) * self.dt + self.released_gaba

        self.M = self.dopamine + self.dopamine_base + self.gaba + self.gaba_base

    def zero_released_neuromodulators(self):
        self.released_dopamine = 0
        self.released_gaba =0


    def __call__(self, group,):
        if isinstance(group, NeuronGroup):
            group.STDP_trace *= group.decay_rate 
            if self.plastic_inhibitory:
                group.STDP_trace[group.spikes] = 0.1 #TODO: STDP_trace pulse magnitute + pulse/cumulative
            else:
                group.STDP_trace[group.spikes * group.excitatory_neurons] = 0.1 
        ### update STDP matrix
        self._update_STDP(group)
        ### Update eligibility trace
        group.eligibility_trace += (-group.eligibility_trace/self.tau_eligibility) * self.dt + group.STDP
        ### Update weights
        group.weights += self.M * group.eligibility_trace
        ### CHeck if any inhibitory/excitatory neuron's sign has been changed
        group.weights[(group.excitatory_neurons.unsqueeze(1) * group.weights) < 0] =  self.min_weight_value
        group.weights[(group.inhibitory_neurons.unsqueeze(1) * group.weights ) > 0] = -self.min_weight_value
        ### Hard bound:
        if self.excitatory_hardbound:
            group.weights[(group.excitatory_neurons.unsqueeze(1) * group.weights) < self.excitatory_hardbound[0]] = self.excitatory_hardbound[0]
            group.weights[(group.excitatory_neurons.unsqueeze(1) * group.weights) > self.excitatory_hardbound[1]] = self.excitatory_hardbound[1]
        if self.inhibitory_hardbound:    
            group.weights[(group.inhibitory_neurons.unsqueeze(1) * group.weights ) < self.inhibitory_hardbound[0]] = self.inhibitory_hardbound[0]
            group.weights[(group.inhibitory_neurons.unsqueeze(1) * group.weights ) < self.inhibitory_hardbound[1]] = self.inhibitory_hardbound[1]


    def release_dopamine(self, quantity):
        self.released_dopamine += quantity

    def release_gaba(self, quantity):
        self.released_gaba += quantity