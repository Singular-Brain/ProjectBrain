import os
import numpy as np
import torch
from matplotlib import pyplot as plt

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
    'base_current': 6E-11, 
    'u_thresh': -55E-3,
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
    neuron_type = "LIF", biological_plausible = False, reward_function = None,
     **kwargs):
        self.dt = dt
        self.weights = torch.from_numpy(network).to(DEVICE)
        self.neuron_type = neuron_type
        self.AdjacencyMatrix = torch.from_numpy(network.astype(bool))
        self.N = len(network)
        self.total_time = total_time
        self.N_runs = 0
        self.reward_function = reward_function
        self.kwargs = kwargs
        self.total_timepoints = int(total_time/dt)
        if biological_plausible:
            self.kwargs.update(BIOLOGICAL_VARIABLES)
        self.eligibility_trace = torch.zeros((self.N,self.N), device = DEVICE)
        self.dopamine = 0
        self.base_current = self.kwargs.get('base_current', 1E-9)
        self.u_thresh = self.kwargs.get('u_thresh', 35E-3)
        self.u_rest = self.kwargs.get('u_rest', -63E-3)
        self.refractory_timepoints = self.kwargs.get('tau_refractory', 0.002) / self.dt
        self.tau_eligibility = self.kwargs.get('tau_c', 1)
        self.tau_dopamine = self.kwargs.get('tau_dopamine', 0.2)
        self.excitatory_chance = self.kwargs.get('excitatory_chance',  0.8)
        self.refractory = (torch.ones((self.N,1)) * self.refractory_timepoints).to(DEVICE)
        self.reward = torch.zeros(self.total_timepoints)
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
        self.stochastic_spikes = self.kwargs.get('stochastic_spikes', False)

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

    def _stochastic_function(self, u,):
        return 1/self.kwargs.get("stochastic_function_tau", 1) *\
               torch.exp(self.kwargs.get("stochastic_function_b", 1) * (u - self.u_thresh))

    @staticmethod
    def STDP_value(x, LTP_rate = 1, LTD_rate = -1.5, time_constant = 0.01):
        """
        Jesper Sjöström and Wulfram Gerstner (2010), Scholarpedia, 5(2):1362
        http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity
        """
        if x>0:
            return LTP_rate*torch.exp(torch.tensor(-x/time_constant))
        elif x<0:
            return LTD_rate*torch.exp(torch.tensor(x/time_constant))
        return 0    
    
    def _STDP(self, tau):
        return torch.tensor(list(map(lambda x: list(map(self.STDP_value, x)), tau.tolist())), dtype = torch.float64)

    def _DA(self):
        if self.reward[self.timepoint]:
            return self.reward[self.timepoint]
        return 0.01

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
            if self.stochastic_spikes:
                spikes = self._stochastic_function(self.potential) > torch.rand(self.N,1)
            else:
                spikes = self.potential>self.u_thresh   # torch.Size([N, 1])
            self.potential[spikes] = self.u_rest
            # if self.neuron_type == 'IZH':
            #     self.recovery[spikes] += 2
            ### Online Learning
            #pre-post 
            post_pre_connections = spikes.T * self.AdjacencyMatrix
            active_post_pre_connections = torch.sum(post_pre_connections, axis= 1, keepdims=True).bool()
            spike_train_slice = self.spike_train[:,:self.timepoint+1]
            active_slice = active_post_pre_connections * spike_train_slice
            tau_values = torch.argmax(active_slice.flip(dims = [1]).double(), axis = 1).view((1,self.N)) * self.dt #s
            tau = tau_values * post_pre_connections
            STDP = self._STDP(tau)
            #post-pre
            pre_post_connections = spikes * self.AdjacencyMatrix
            active_pre_post_connections = torch.sum(pre_post_connections, axis= 0, keepdims=True).bool()
            active_slice = active_pre_post_connections.T * spike_train_slice
            tau_values = torch.argmax(active_slice.flip(dims = [1]).double(), axis = 1).view((1,self.N)) * self.dt #s
            tau = tau_values * pre_post_connections
            STDP += self._STDP(-tau)
            ### Update eligibility trace
            self.eligibility_trace += (-self.eligibility_trace/self.tau_eligibility + STDP) * self.dt
            ### Update reward
            if self.reward_function:
                self.reward = self.reward_function(self.dt, self.spike_train, self.timepoint, self.reward)
            ### Dopamine
            self.dopamine += (-self.dopamine/self.tau_dopamine + self._DA()) * self.dt
            ### Update weights
            self.weights += self.dopamine * self.eligibility_trace *\
                (1 if self.kwargs.get('plastic_inhibitory', True) else ((np.sign(self.weights.sum(axis = 1)) + 1)/2).view(self.N, 1))
            ### Spike train
            self.spike_train[:,self.timepoint] = spikes.ravel()
            self.refractory *= torch.logical_not(spikes).to(DEVICE)
            ### Transfer currents + external sources
            new_currents = ((spikes * self.weights).sum(axis = 0).view(self.N,1) * self.base_current).to(DEVICE)
            open_neurons = self.refractory >= self.refractory_timepoints
            stim_current = self.get_stimuli_current() if self.stimuli else 0
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

    def plot_spikes(self):
        y, x = np.where(self.spike_train)
        plt.figure(figsize=(20,4))
        plt.plot(x, y, '.', color='black', marker='o', markersize=2)
        plt.show()

    def f_I_Curve(self):
        if self.neuron_type == 'LIF':
            self.critical_current = (self.u_thresh - self.u_rest) / self.Rm
            assert self.base_current > self.critical_current, "Please Increase the base current, it can't fire any neuron"
            self.max_freq = 1 / (self.dt * self.self.refractory_timepoints)
            self.freq_base_current = 1 / self.dt * (self.self.refractory_timepoints + np.log(1 / (1 - self.critical_current / self.base_current)) )
            self.freq = 1 / self.dt * (self.self.refractory_timepoints + np.log(1 / (1 - self.critical_current / self.current)) )
