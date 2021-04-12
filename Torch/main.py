import numpy as np
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Stimulus:
    def __init__(self, dt, output, neurons):
        self.output = output
        self.neurons = neurons
        self.dt = dt

    def __call__(self, timestep):
        return self.output(timestep * self.dt)


class NeuronGroup:
    def __init__(self, dt, population_size, connection_chance, total_time, stimuli = set(),
    neuron_type = "LIF", **kwargs):
        self.dt = dt
        self.N = population_size
        self.total_timepoints = int(total_time/dt)
        self.stimuli = stimuli
        self.kwargs = kwargs
        self.connection_chance = connection_chance
        self.base_current = kwargs.get('base_current', 1E-9)
        self.u_thresh = kwargs.get('u_thresh', 35E-3)
        self.u_rest = kwargs.get('u_rest', -63E-3)
        self.refractory_timepoints = kwargs.get('tau_refractory', 0.002) / self.dt
        self.excitatory_chance = kwargs.get('excitatory_chance',  0.8)
        self.refractory = torch.ones((self.N,1)).to(DEVICE) * self.refractory_timepoints
        self.current = torch.zeros((self.N,1)).to(DEVICE)
        self.potential = torch.ones((self.N,1)).to(DEVICE) * self.u_rest
        self.spike_train = torch.zeros((self.N, self.total_timepoints), dtype= torch.bool).to(DEVICE)
        self.weights = np.random.rand(self.N,self.N)
        np.fill_diagonal(self.weights, 0)
        self.weights = torch.from_numpy(self.weights)
        self.connections = (torch.rand(*self.weights.shape) + self.connection_chance).type(torch.int)
        self.excitatory_neurons = (torch.rand(*self.weights.shape) + self.excitatory_chance).type(torch.int) * 2 -1
        self.AdjacencyMatrix = self.connections * self.excitatory_neurons * self.weights
        self.AdjacencyMatrix = self.AdjacencyMatrix.to(DEVICE)
        self.stimuli = np.array(list(stimuli))
        self.StimuliAdjacency = np.zeros((self.N, len(stimuli)), dtype=np.bool)
        for i, stimulus in enumerate(self.stimuli):
            self.StimuliAdjacency[stimulus.neurons, i] = True

    def LIF(self):
        """
        Leaky Integrate-and-Fire Neural Model
        """
        Rm = self.kwargs.get("Rm", 135E6)
        Cm = self.kwargs.get("Cm", 14.7E-12)
        tau_m = Rm*Cm
        exp_term = torch.exp(torch.tensor(-self.dt/tau_m))
        u_base = (1-exp_term) * self.u_rest
        new_potential = u_base + exp_term * self.potential + self.current*self.dt/Cm
        return new_potential.to(DEVICE)

    def get_stimuli_current(self):
        call_stimuli =  np.vectorize(lambda stim: stim(self.timepoint))
        stimuli_output = call_stimuli(self.stimuli)
        stimuli_current = (stimuli_output * self.StimuliAdjacency).sum(axis = 1)
        return torch.tensor(stimuli_current.reshape(self.N,1)).to(DEVICE)
    
    def run(self):
        for self.timepoint in range(self.total_timepoints):
            ### LIF update
            self.refractory +=1
            self.potential = self.LIF()
            ### Reset currents
            self.current = torch.zeros((self.N,1)).to(DEVICE)
            ### Spikes 
            spikes = self.potential>self.u_thresh
            self.spike_train[:,self.timepoint] = spikes.ravel()
            self.refractory *= torch.logical_not(spikes).to(DEVICE)
            ### Transfer currents + external sources
            new_currents = (spikes * self.AdjacencyMatrix).sum(axis = 0).reshape(self.N,1) * self.base_current
            open_neurons = self.refractory >= self.refractory_timepoints
            self.current += new_currents * open_neurons
            self.current += self.get_stimuli_current()

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
