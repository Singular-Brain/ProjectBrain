import numpy as np
import torch


BIOLOGICAL_VARIABLES = {
    'base_current': 1E-9,
    'u_thresh': 35E-3,
    'u_rest': -63E-3,
    'tau_refractory': 0.002,
    'excitatory_chance':  0.8,
    "Rm": 135E6,
    "Cm": 14.7E-12,
}
class Stimulus:
    def __init__(self, dt, output, neurons):
        self.output = output
        self.neurons = neurons
        self.dt = dt

    def __call__(self, timestep):
        return self.output(timestep * self.dt)


class NeuronGroup:
    def __init__(self, dt, population_size, connection_chance, total_time, stimuli = set(),
    neuron_type = "LIF", biological_plausible= False, **kwargs):
        self.dt = dt
        self.N = population_size
        self.total_timepoints = int(total_time/dt)
        self.stimuli = stimuli
        self.kwargs = kwargs
        if biological_plausible:
            self.kwargs = {key: BIOLOGICAL_VARIABLES.get(key, self.kwargs[key]) for key in self.kwargs}
        self.connection_chance = connection_chance
        self.base_current = kwargs.get('base_current', 1E-9)
        self.u_thresh = kwargs.get('u_thresh', 35E-3)
        self.u_rest = kwargs.get('u_rest', -63E-3)
        self.refractory_timepoints = kwargs.get('tau_refractory', 0.002) / self.dt
        self.excitatory_chance = kwargs.get('excitatory_chance',  0.8)
        self.refractory = torch.ones((self.N,1)).to(DEVICE) * self.refractory_timepoints
        self.current = torch.zeros((self.N,1)).to(DEVICE)
        self.potential = torch.ones((self.N,1)).to(DEVICE) * self.u_rest
        self.save_history = kwargs.get('save_history',  False)
        if self.save_history:
            self.current_history = torch.zeros((self.N, self.total_timepoints), dtype= torch.float32).to(DEVICE)
            self.potential_history = torch.zeros((self.N, self.total_timepoints), dtype= torch.float32).to(DEVICE)
        self.spike_train = torch.zeros((self.N, self.total_timepoints), dtype= torch.bool).to(DEVICE)
        weights_values = np.random.rand(self.N,self.N)
        np.fill_diagonal(weights_values, 0)
        weights_values = torch.from_numpy(weights_values)
        self.AdjacencyMatrix = (torch.rand(self.N,self.N) + self.connection_chance).type(torch.int)
        self.excitatory_neurons = (torch.rand(self.N,self.N) + self.excitatory_chance).type(torch.int) * 2 -1
        self.weights = self.AdjacencyMatrix * self.excitatory_neurons * weights_values
        self.weights = self.weights.to(DEVICE)
        self.stimuli = np.array(list(stimuli))
        self.StimuliAdjacency = np.zeros((self.N, len(stimuli)), dtype=np.bool)
        for i, stimulus in enumerate(self.stimuli):
            self.StimuliAdjacency[stimulus.neurons, i] = True
        ### Neuron type variables:
        if neuron_type == 'IF':
            self.Cm = torch.tensor(self.kwargs.get("Cm", 14.7E-12), device = DEVICE)
        elif neuron_type == 'LIF':
            self.Cm = torch.tensor(self.kwargs.get("Cm", 14.7E-12), device = DEVICE)
            Rm = self.kwargs.get("Rm", 135E6)
            tau_m = self.Cm * Rm 
            self.exp_term = (torch.exp(-self.dt/tau_m))
            self.u_base = ((1-self.exp_term) * self.u_rest)

    def IF(self):
        """
        Simple Perfect Integrate-and-Fire Neural Model:
        Assumes a fully-insulated memberane with resistance approaching infinity
        """
        new_potential = self.potential + (self.dt * self.current)/Cm
        return new_potential
        
    def LIF(self):
        """
        Leaky Integrate-and-Fire Neural Model
        """
        new_potential = self.u_base + self.exp_term * self.potential + self.current*self.dt/self.Cm
        return new_potential

    def get_stimuli_current(self):
        call_stimuli =  np.vectorize(lambda stim: stim(self.timepoint))
        stimuli_output = call_stimuli(self.stimuli)
        stimuli_current = (stimuli_output * self.StimuliAdjacency).sum(axis = 1)
        return torch.tensor(stimuli_current.reshape(self.N,1), device = DEVICE)
    
    def run(self):
        for self.timepoint in range(self.total_timepoints):
            ### LIF update
            self.refractory +=1
            self.potential = self.LIF()
            if self.save_history:
                self.potential_history[:,self.timepoint] = self.potential.ravel()
            ### Reset currents
            self.current = torch.zeros((self.N,1), device = DEVICE)
            ### Spikes 
            spikes = self.potential>self.u_thresh
            self.potential[spikes] = self.u_rest
            self.spike_train[:,self.timepoint] = spikes.ravel()
            self.refractory *= torch.logical_not(spikes)
            ### Transfer currents + external sources
            new_currents = (spikes * self.weights).sum(axis = 0).reshape(self.N,1) * self.base_current
            open_neurons = self.refractory >= self.refractory_timepoints
            self.current += (new_currents + self.get_stimuli_current()) * open_neurons
            if self.save_history:
                self.current_history[:,self.timepoint] = self.current.ravel()

    def _spike_train_repr(self, spike_train):
        string = ''
        for t in spike_train:
            string += '|' if t else ' '
        return string

    def display_spikes(self):
        spike_train_display = ' id\n' + '=' * 5 + '╔' + '═' * self.total_timepoints + '╗\n'
        for i, spike_train in enumerate(self.spike_train):
            spike_train_display += str(i) + ' ' * (5 - len(str(i))) \
            + '║' + self._spike_train_repr(spike_train) + '║\n'  
        spike_train_display +=' ' * 5 + '╚' + '═' * self.total_timepoints + '╝'
        print(spike_train_display)

class RFSTDP:
    def __init__(self, NeuronGroup,
                 interval_time = 0.001, # seconds
                 pre_post_rate = 0.001,
                 reward_pre_post_rate = 0.002,
                 post_pre_rate = -0.001,
                 reward_post_pre_rate = 0.001,
                 ):
        """
        Reward-modulated Flat STDP 
        """
        self.NeuronGroup = NeuronGroup
        self.weights = NeuronGroup.weights
        self.N = NeuronGroup.N
        self.interval_timepoints = int(interval_time / NeuronGroup.dt) #timepoints
        self.total_timepoints = NeuronGroup.total_timepoints
        self.spike_train = NeuronGroup.spike_train
        self.reward_based = True
        self.pre_post_rate = pre_post_rate
        self.post_pre_rate = post_pre_rate
        self.reward_pre_post_rate = reward_pre_post_rate
        self.reward_post_pre_rate = reward_post_pre_rate

    def __call__(self, reward):
        spike_train = self.NeuronGroup.spike_train
        padded_spike_train = torch.nn.functional.pad(spike_train,
            (self.interval_timepoints, self.interval_timepoints, 0, 0),
             mode='constant', value=0)
        for i in range(self.total_timepoints + self.interval_timepoints):
            section = padded_spike_train[:,i:i+self.interval_timepoints]
            span = section.sum(axis = 1).type(torch.bool)
            first = padded_spike_train[:,i]
            last = padded_spike_train[:,i+self.interval_timepoints]
            if reward:
                self.weights += self.reward_pre_post_rate * (first * span.reshape(1, self.N) * self.weights)
                self.weights += self.reward_post_pre_rate * (span  * last.reshape(1, self.N) * self.weights)
            if not reward:
                self.weights += self.pre_post_rate * (first * span.reshape(1, self.N) * self.weights)
                self.weights += self.post_pre_rate * (span  * last.reshape(1, self.N) * self.weights)


