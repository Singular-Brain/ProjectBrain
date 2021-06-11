import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from ..callbacks import CallbackList

# set DEVICE
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device is set to {DEVICE.upper()}')
 
#set manual seed
def manual_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
SEED = 2045
#manual_seed(SEED) # We set the seed to 2045 because the Singularity is near!
 
 
 
BIOLOGICAL_VARIABLES = {
    'base_current': 6E-11, 
    'u_thresh': -55E-3,
    'u_rest': -68E-3,
    'tau_refractory': 0.002,
    'excitatory_chance':  0.8,
    "Rm": 135E6,
    "Cm": 14.7E-12,
}
 


class NeuronGroup:
    def __init__(self, network, total_time, dt, biological_plausible = False,
                reward_function = None, learning_rule = None, callbacks = [],
                **kwargs):
        self.dt = dt
        self.network = network
        self.weights = network.weights
        self.N = network.total_neurons
        self.adjacency_matrix = network.adjacency_matrix
        self.excitatory_neurons = network.excitatory_neurons
        self.inhibitory_neurons = network.inhibitory_neurons
        self.total_time = total_time
        self.N_runs = 0
        self.reward_function = reward_function
        self.kwargs = kwargs
        self.total_timepoints = int(total_time/dt)
        ### callbacks
        if isinstance(callbacks, CallbackList):
            self.callbacks = callbacks
        else:
            self.callbacks = CallbackList(callbacks)
        for callback in callbacks:
            callback.set_NeuronGroup(self)
        ### neurons variables
        if biological_plausible:
            self.kwargs.update(BIOLOGICAL_VARIABLES)        
        self.base_current = self.kwargs.get('base_current', 6E-11)
        self.u_thresh = self.kwargs.get('u_thresh', -55E-3)
        self.u_rest = self.kwargs.get('u_rest', -68E-3)
        self.refractory_timepoints = self.kwargs.get('tau_refractory', 0.002) / self.dt
        self.excitatory_chance = self.kwargs.get('excitatory_chance',  0.8)
        self.stochastic_spikes = self.kwargs.get('stochastic_spikes', False)
        ### neurons variables
        self.refractory = torch.ones((self.N), device = DEVICE) * self.refractory_timepoints
        self.current = torch.zeros((self.N), device = DEVICE)
        self.potential = torch.ones((self.N), device = DEVICE) * self.u_rest
        self.spike_train = torch.zeros((self.N, self.total_timepoints), dtype= torch.bool)
        ### online learning
        self.learning_rule = learning_rule
        if self.learning_rule:
            self.learning_rule.set_params(self.dt, self.network)
            self.rewards = torch.zeros(self.total_timepoints, device = DEVICE)
        ### save history
        self.save_history = self.kwargs.get('save_history',  False)
        if self.save_history:
            self.current_history = torch.zeros((self.N, self.total_timepoints), dtype= torch.float32).to(DEVICE)
            self.potential_history = torch.zeros((self.N, self.total_timepoints), dtype= torch.float32).to(DEVICE)
        ### Neuron type variables:
        self.Cm = self.kwargs.get("Cm", 14.7E-12)
        Rm = self.kwargs.get("Rm", 135E6)
        self.tau_m = self.Cm * Rm 

 
        #TODO: IF Model:
        """
        Simple Perfect Integrate-and-Fire Neural Model:
        Assumes a fully-insulated memberane with resistance approaching infinity
        
        return self.potential + (self.dt * self.current)/self.Cm
        """
 
    def _update_potential(self):
        """
        Leaky Integrate-and-Fire Neural Model
        """
        return (-self.dt/self.tau_m)*(self.potential-self.u_rest) + self.dt * self.current / self.Cm
 
 
    def _get_stimuli_current(self):
        call_stimuli =  np.vectorize(lambda stim: stim(self.timepoint))
        stimuli_output = call_stimuli(self.stimuli)
        stimuli_current = (stimuli_output * self.StimuliAdjacency).sum(axis = 1)
        return torch.from_numpy(stimuli_current).to(DEVICE)
 
    def _stochastic_function(self, u,):
        return 1/self.kwargs.get("stochastic_function_tau", 1) *\
               torch.exp(self.kwargs.get("stochastic_function_b", 1) * (u - self.u_thresh))
 
    def _make_stimuli_adjacency(self, stimuli):
        self.stimuli = stimuli
        if self.stimuli is not None:
            self.stimuli = np.array(list(stimuli))
            self.StimuliAdjacency = np.zeros((self.N, len(stimuli)),dtype=np.bool)
            for i, stimulus in enumerate(self.stimuli):
                self.StimuliAdjacency[stimulus.neurons, i] = True

 
    def run(self, stimuli = None, progress_bar = False):
        self._reset()
        self._make_stimuli_adjacency(stimuli)
        self.callbacks.on_run_start(self.N_runs)
        for self.timepoint in tqdm(range(self.total_timepoints)) if progress_bar \
        else range(self.total_timepoints):
            self.callbacks.on_timepoint_start(self.timepoint)
            self.refractory +=1
            ### update potentials
            self.potential += 0.5*self._update_potential() # To Improve approximation 
            self.potential += 0.5*self._update_potential() # and numerical stability
            if self.save_history:
                self.potential_history[:,self.timepoint] = self.potential
            ### Spikes 
            if self.stochastic_spikes:
                self.spikes = self._stochastic_function(self.potential) > torch.rand(self.N, device =DEVICE)
            else:
                self.spikes = self.potential>self.u_thresh   # torch.Size([N,])
            self.potential[self.spikes] = self.u_rest
            ### Online Learning
            if self.learning_rule:
                ### Update reward
                if self.reward_function:
                    self.reward_function(self)
                ### Update weights
                self.weights = self.learning_rule(self.weights, self.spikes, self.rewards[self.timepoint])
            ### Spike train
            self.spike_train[:,self.timepoint] = self.spikes
            self.refractory *= torch.logical_not(self.spikes).to(DEVICE)
            ### Transfer currents + external sources
            new_currents = ((self.spikes.reshape(self.N,1) * self.weights).sum(axis = 0) * self.base_current).to(DEVICE)
            stim_current = self._get_stimuli_current() if self.stimuli is not None else 0
            self.current = (stim_current + new_currents) * (self.refractory >= self.refractory_timepoints) # implement current for open neurons 
            if self.save_history:
                self.current_history[:,self.timepoint] = self.current
            self.callbacks.on_timepoint_end(self.timepoint)
        self.callbacks.on_run_end(self.N_runs)
 
    def _reset(self):
        self.N_runs +=1
        self.refractory = torch.ones(self.N).to(DEVICE) * self.refractory_timepoints
        self.current = torch.zeros(self.N).to(DEVICE)
        self.potential = torch.ones(self.N).to(DEVICE) * self.u_rest
        self.spike_train = torch.zeros((self.N, self.total_timepoints),dtype=torch.bool).to(DEVICE)
        self.reward = torch.zeros(self.total_timepoints, device = DEVICE)
        if self.save_history:
            self.current_history = torch.zeros((self.N, self.total_timepoints), dtype= torch.float32).to(DEVICE)
            self.potential_history = torch.zeros((self.N, self.total_timepoints), dtype= torch.float32).to(DEVICE)

    def weight_values(self, slice = None):
        target_weights = self.weights[slice]
        return target_weights[target_weights != 0]

    @property
    def seconds(self):
        return round(self.timepoint * self.dt, 2)