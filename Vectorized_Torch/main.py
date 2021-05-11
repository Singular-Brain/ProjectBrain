import os
import numpy as np
import torch
from AdjacencyMatrix import *

if (torch.cuda.is_available()):
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

DEVICE='cpu'
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
    'base_current': 1E-9, 
    'u_thresh': -48E-3,
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
        self.N_runs = 0
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
        self.save_to_file = self.kwargs.get('save_to_file',  None)
        if self.save_to_file and not self.save_to_file.endswith('.npy'):
            self.save_to_file += '.npy'
        if self.save_to_file and os.path.exists(self.save_to_file):
            ans = input(f'File: "{self.save_to_file}" already exists! Do you want to rewrite? Y/[N]\n')
            if ans.lower() != 'y':
                self.save_to_file = None
        if self.save_to_file:
            data = {
                    "dt":self.dt,
                    "total_timepoints": self.total_timepoints,
                    "save_history": self.save_history,
                    "total_time":self.total_time,
                    "AdjacencyMatrix": self.AdjacencyMatrix,
                    "kwargs": {**self.kwargs},
                    "runs":[],
                    }
            np.save(self.save_to_file, data)
        self.spike_train = torch.zeros((self.N, self.total_timepoints), dtype= torch.bool)
        self.stimuli = np.array(list(stimuli))
        self.StimuliAdjacency = np.zeros((self.N, len(stimuli)),dtype=np.bool)
        for i, stimulus in enumerate(self.stimuli):
            self.StimuliAdjacency[stimulus.neurons, i] = True
        ### Neuron type variables:
        if neuron_type == 'IF':
            self.NeuronType = self.IF
            self.Cm = self.kwargs.get("Cm", 14.7E-12)
        elif neuron_type == 'LIF':
            self.NeuronType = self.LIF
            self.Cm = self.kwargs.get("Cm", 14.7E-12)
            Rm = self.kwargs.get("Rm", 135E6)
            self.tau_m = self.Cm * Rm 
            self.exp_term = np.exp(-self.dt/self.tau_m)
            self.u_base = (1-self.exp_term) * self.u_rest
        elif neuron_type == 'IZH':
            self.NeuronType = self.IZH
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
        #return self.u_base + self.exp_term * self.potential + self.current*self.dt/self.Cm 
        return (-self.dt/self.tau_m)*(self.potential-self.u_rest) + self.dt * self.current / self.Cm

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
            self.refractory +=1
            ### update potentials
            self.potential += self.NeuronType()
            if self.save_history:
                self.potential_history[:,self.timepoint] = self.potential.ravel()
            ### Reset currents
            self.current = torch.zeros(self.N,1).to(DEVICE) 
            ### Spikes 
            spikes = self.potential>self.u_thresh
            self.potential[spikes] = self.u_rest #I think we should only change it to u_reset , which is lower than u_rest, Fig.1.8 Neural dynamics(tau_refractory >> 4 * tau) 
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
        if self.save_to_file:
            data = np.load(self.save_to_file, allow_pickle=True)
            run_data = {"run": self.N_runs,
                "weights": self.weights,
                "spike_train": self.spike_train,
                "current_history": self.current_history,
                "potential_history": self.potential_history,
                }
            data[()]['runs'].append(run_data)
            np.save(self.save_to_file, data)

    def _reset(self):
        self.N_runs +=1
        self.refractory = torch.ones(self.N,1).to(DEVICE) * self.refractory_timepoints
        self.current = torch.zeros(self.N,1).to(DEVICE)
        self.potential = torch.ones(self.N,1).to(DEVICE) * self.u_rest
        self.spike_train = torch.zeros((self.N, self.total_timepoints),dtype=torch.bool).to(DEVICE)


    def _spike_train_repr(self, spike_train):
        b = spike_train.cpu().numpy().tobytes()
        return b.decode('UTF-8').replace('\x01', '|').replace('\x00', ' ')

    def display_spikes(self):
        spike_train_display = ' id\n' + '=' * 5 + '╔' + '═' * self.total_timepoints + '╗\n'
        for i, spike_train in enumerate(self.spike_train):
            spike_train_display += str(i) + ' ' * (5 - len(str(i))) \
            + '║' + self._spike_train_repr(spike_train) + '║\n'  
        spike_train_display +=' ' * 5 + '╚' + '═' * self.total_timepoints + '╝'
        print(spike_train_display)

    def f_I_Curve(self):
        if self.neuron_type == 'LIF':
            self.critical_current = (self.u_thresh - self.u_rest) / self.Rm
            assert self.base_current > self.critical_current, "Please Increase the base current, it can't fire any neuron"
            self.max_freq = 1 / (self.dt * self.self.refractory_timepoints)
            self.freq_base_current = 1 / self.dt * (self.self.refractory_timepoints + np.log(1 / (1 - self.critical_current / self.base_current)) )
            self.freq = 1 / self.dt * (self.self.refractory_timepoints + np.log(1 / (1 - self.critical_current / self.current)) )
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

