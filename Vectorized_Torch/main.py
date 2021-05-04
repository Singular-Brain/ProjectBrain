import numpy as np
import torch
from AdjacencyMatrix import *

if (torch.cuda.is_available()):
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
print(f'Device is set to {DEVICE}')

#set manual seed
def manual_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    from torch.backends import cudnn
    cudnn.deterministic = True #type: ignore
    cudnn.benchmark = False # type: ignore
SEED = 2045
#manual_seed(SEED)
# We set the seed to 2045 because the Singularity is near!



BIOLOGICAL_VARIABLES = {
    'base_current': 1E-9, #based on Neural Dynamics p8,single EPSP has amplitude in range of 1mV. 
    'u_thresh': -48E-3, # The critical value for spike initiation is about 20 to 30mV above the resting potential.(P8)
    'u_rest': -68E-3,
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
    def __init__(self, network, total_time, dt, stimuli = set(),
    neuron_type = "LIF", biological_plausible = False,
     **kwargs):
        self.dt = dt
        self.weights = torch.from_numpy(network).to(DEVICE)
        self.neuron_type = neuron_type
        self.AdjacencyMatrix = network.astype(bool)
        self.N = len(network)
        self.total_time = total_time
        self.kwargs = kwargs
        self.total_timepoints = int(total_time/dt)
        if biological_plausible:
            self.kwargs.update(BIOLOGICAL_VARIABLES)
        self.base_current = self.kwargs.get('base_current', 1E-9)
        self.u_thresh = self.kwargs.get('u_thresh', 35E-3)
        self.u_rest = self.kwargs.get('u_rest', -63E-3)
        self.refractory_timepoints = self.kwargs.get('tau_refractory', 0.002) / self.dt
        self.excitatory_chance = self.kwargs.get('excitatory_chance',  0.8)
        self.refractory = (torch.ones((self.N,1)) * self.refractory_timepoints).to(DEVICE)
        self.current = torch.zeros((self.N,1)).to(DEVICE)
        self.potential = torch.ones((self.N,1)).to(DEVICE) * self.u_rest
        self.save_history = self.kwargs.get('save_history',  False)
        if self.save_history:
            self.current_history = torch.zeros((self.N, self.total_timepoints), dtype= torch.float32).to(DEVICE)
            self.potential_history = torch.zeros((self.N, self.total_timepoints), dtype= torch.float32).to(DEVICE)
        self.spike_train = torch.zeros((self.N, self.total_timepoints), dtype= torch.bool)
        self.stimuli = np.array(list(stimuli))
        self.StimuliAdjacency = np.zeros((self.N, len(stimuli)),dtype=np.bool)
        for i, stimulus in enumerate(self.stimuli):
            self.StimuliAdjacency[stimulus.neurons, i] = True
        ### Neuron type variables:
        if neuron_type == 'IF':
            self.Cm = self.kwargs.get("Cm", 14.7E-12)
        elif neuron_type == 'LIF':
            self.Cm = self.kwargs.get("Cm", 14.7E-12)
            Rm = self.kwargs.get("Rm", 135E6)
            tau_m = self.Cm * Rm 
            self.exp_term = np.exp(-self.dt/tau_m)
            self.u_base = (1-self.exp_term) * self.u_rest
        elif neuron_type == 'IZH':
            self.recovery = (torch.ones(self.N,1)*self.u_rest*0.2).to(DEVICE)

    def IF(self):
        """
        Simple Perfect Integrate-and-Fire Neural Model:
        Assumes a fully-insulated memberane with resistance approaching infinity
        """
        return self.potential + (self.dt * self.current)/self.Cm

    def LIF(self):
        """
        Leaky Integrate-and-Fire Neural Model
        """
        return self.u_base + self.exp_term * self.potential + self.current*self.dt/self.Cm 
    
    def IZH(self,a=0.02, b=0.2, c =-65, d=2,
                    c1=0.04, c2=5, c3=140, c4=1, c5=1):
        """
        Izhikevich Neural Model
        """
        self.potential = self.potential + c1*(self.potential**2)\
            + c2*self.potential + c3-c4*self.recovery + c5*self.current
        self.recovery += a*(b*self.potential-self.recovery)
        return self.potential


    def get_stimuli_current(self):
        call_stimuli =  np.vectorize(lambda stim: stim(self.timepoint))
        stimuli_output = call_stimuli(self.stimuli)
        stimuli_current = (stimuli_output * self.StimuliAdjacency).sum(axis = 1)
        return torch.from_numpy(stimuli_current.reshape(self.N,1)).to(DEVICE)
    
    def run(self):
        self._reset()
        for self.timepoint in range(self.total_timepoints):
            if self.neuron_type == 'LIF':
                ### LIF update
                self.refractory +=1
                self.potential = self.LIF()
            elif self.neuron_type == 'IF':
                ### IF update
                self.refractory +=1
                self.potential = self.IF()
            elif self.neuron_type == 'IZH':
                ### IZH update
                self.refractory +=1
                self.potential = self.IZH()
            if self.save_history:
                self.potential_history[:,self.timepoint] = self.potential.ravel()
            ### Reset currents
            self.current = torch.zeros(self.N,1).to(DEVICE) 
            ### Spikes 
            spikes = self.potential>self.u_thresh
            self.potential[spikes] = self.u_rest
            if self.neuron_type == 'IZH':
                self.recovery[spikes] += 2
            self.spike_train[:,self.timepoint] = spikes.ravel()
            self.refractory *= torch.logical_not(spikes).to(DEVICE)
            ### Transfer currents + external sources
            new_currents = ((spikes * self.weights).sum(axis = 0).reshape(self.N,1) * self.base_current).to(DEVICE)
            open_neurons = self.refractory >= self.refractory_timepoints
            stim_current = self.get_stimuli_current()
            self.current += (stim_current + new_currents) * open_neurons
            if self.save_history:
                self.current_history[:,self.timepoint] = self.current.ravel()

    def _reset(self):
        self.refractory = torch.ones(self.N,1).to(DEVICE) * self.refractory_timepoints
        self.current = torch.zeros(self.N,1).to(DEVICE)
        self.potential = torch.ones(self.N,1).to(DEVICE) * self.u_rest
        self.spike_train = torch.zeros((self.N, self.total_timepoints),dtype=torch.bool).to(DEVICE)


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
        self.N = NeuronGroup.N
        self.weights = NeuronGroup.weights
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
            section = padded_spike_train[:,i+1:i+self.interval_timepoints-1]
            span = section.sum(axis = 1).type(torch.bool).to(DEVICE)
            first = padded_spike_train[:,i]
            last = padded_spike_train[:,i+self.interval_timepoints]
            if reward:
                self.weights += self.reward_pre_post_rate * (first * span.reshape(1, self.N) * self.weights)
                self.weights += self.reward_post_pre_rate * (span  * last.reshape(1, self.N) * self.weights)
            if not reward:
                self.weights += self.pre_post_rate * (first * span.reshape(1, self.N) * self.weights)
                self.weights += self.post_pre_rate * (span  * last.reshape(1, self.N) * self.weights)

