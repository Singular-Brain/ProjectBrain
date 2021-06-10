import torch
import numpy as np
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



class STDP:
    def __init__(self,
                LTP_rate = 0.188, 
                LTD_rate = -0.094, 
                tau_LTP = 0.02,   
                tau_LTD = 0.04,   
                dopamine_base = 0.002, tau_eligibility = 1, tau_dopamine = 0.2,
                hard_bound = None, plastic_inhibitory = True):
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
            ValueError: If any of hardbound boundaries is set to 0.
        """
        self.tau_eligibility = tau_eligibility
        self.tau_dopamine = tau_dopamine
        self.dopamine = 0
        self.dopamine_base = dopamine_base
        self.LTP_rate = LTP_rate
        self.LTD_rate = LTD_rate
        self.tau_LTP = tau_LTP
        self.tau_LTD = tau_LTD
        self.plastic_inhibitory = plastic_inhibitory
        self.hard_bound = hard_bound
        if self.hard_bound and 0 in self.hard_bound:
            raise ValueError("'0' cannot be in hard bound")

    def set_params(self, dt, network,):
        self.dt = dt
        self.weights = network.weights
        self.N = network.total_neurons
        self.adjacency_matrix = network.adjacency_matrix
        self.excitatory_neurons = network.excitatory_neurons
        self.inhibitory_neurons = network.inhibitory_neurons
        self.STDP_trace = torch.zeros((self.N,1), device = DEVICE)
        self.decay_rate = (self.excitatory_neurons * (np.exp(-self.dt/self.tau_LTP)) + self.inhibitory_neurons *  (np.exp(-self.dt/self.tau_LTD))).reshape((self.N, 1))
        self.STDP = torch.zeros((self.N, self.N), device=DEVICE)
        self.eligibility_trace = torch.zeros((self.N,self.N), device = DEVICE)

    def _update_STDP(self, spikes,):
        # reset STDP
        self.STDP.fill_(0)
        ### post-pre spikes (= pre-post connections)
        self.STDP[spikes,:] = self.LTD_rate * self.adjacency_matrix[spikes,:] * self.STDP_trace.T
        ### pre-post spikes (= post-pre connections)
        self.STDP[:,spikes] = self.LTP_rate * self.adjacency_matrix[:,spikes] * self.STDP_trace 
        #TODO: handle simultaneous pre-post spikes

    def __call__(self, weights, spikes, reward):
        self.STDP_trace *= self.decay_rate 
        if self.plastic_inhibitory:
            self.STDP_trace[spikes] = 0.1
        else:
            self.STDP_trace[spikes * self.excitatory_neurons] = 0.1 #TODO: STDP_trace pulse magnitute + pulse/cumulative
        ### update STDP matrix
        self._update_STDP(spikes)
        ### Update eligibility trace
        self.eligibility_trace += (-self.eligibility_trace/self.tau_eligibility) * self.dt + self.STDP
        ### Dopamine
        self.dopamine += (-self.dopamine/self.tau_dopamine ) * self.dt + reward
        ### Update weights
        weights += (self.dopamine_base+self.dopamine) * self.eligibility_trace
        ### Hard bound:
        if self.hard_bound:
            weights[(weights * self.excitatory_neurons) < 0] = self.hard_bound(0)
            weights[weights > 1] = self.hard_bound(1)
        return weights
