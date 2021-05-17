import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
 
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
        self.N = len(network)
        self.excitetory_neurons = ((np.sign(self.weights.sum(axis = 1).cpu()) + 1)/2).view(self.N, 1).to(DEVICE)
        self.inhibitory_neurons = ((np.sign(self.weights.sum(axis = 1).cpu()) - 1)/-2).view(self.N, 1).to(DEVICE)
        self.neuron_type = neuron_type
        self.AdjacencyMatrix = torch.from_numpy(network.astype(bool)).to(DEVICE)
        self.total_time = total_time
        self.N_runs = 0
        self.reward_function = reward_function
        self.kwargs = kwargs
        self.total_timepoints = int(total_time/dt)
        if biological_plausible:
            self.kwargs.update(BIOLOGICAL_VARIABLES)
        self.eligibility_trace = torch.zeros((self.N,self.N), device = DEVICE)
        self.dopamine = 0
        self.window_width = self.kwargs.get('window_width', int(10 * self.kwargs.get('STDP_time_constant', 0.01)/self.dt))
        self.base_current = self.kwargs.get('base_current', 1E-9)
        self.u_thresh = self.kwargs.get('u_thresh', 35E-3)
        self.u_rest = self.kwargs.get('u_rest', -63E-3)
        self.refractory_timepoints = self.kwargs.get('tau_refractory', 0.002) / self.dt
        self.tau_eligibility = self.kwargs.get('tau_c', 1)
        self.tau_dopamine = self.kwargs.get('tau_dopamine', 0.2)
        self.excitatory_chance = self.kwargs.get('excitatory_chance',  0.8)
        self.refractory = torch.ones((self.N,1), device = DEVICE) * self.refractory_timepoints
        self.reward = torch.ones(self.total_timepoints, device = DEVICE) * 0.01
        self.current = torch.zeros((self.N,1), device = DEVICE)
        self.potential = torch.ones((self.N,1), device = DEVICE) * self.u_rest
        self.STDP_trace = torch.zeros((self.N,1), device = DEVICE)
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
            self.u_thresh = 30.0
            self.u_rest = -65.0
            self.potential = torch.ones(self.N,1,device=DEVICE) * self.u_rest
            self.recovery = self.potential*0.2
            self.current = 13*(torch.rand(self.N,1)-0.5)
            self.d = torch.ones(self.N, 1, device=DEVICE)
            self.a = torch.ones(self.N, 1, device=DEVICE)
            self.d[self.d * self.excitatory_neurons >= 0.01] = 8
            self.d[self.d * self.inhibitory_neurons >= 0.01] = 2
            self.a[self.a * self.excitatory_neurons >= 0.01] = 0.02
            self.a[self.a * self.inhibitory_neurons >= 0.01] = 0.1

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
 
    def IZH(self):
        """
        Izhikevich Neural Model
        """
        return (0.04*self.potential+5)*self.potential+140-self.recovery+self.current
 
 
    def get_stimuli_current(self):
        call_stimuli =  np.vectorize(lambda stim: stim(self.timepoint))
        stimuli_output = call_stimuli(self.stimuli)
        stimuli_current = (stimuli_output * self.StimuliAdjacency).sum(axis = 1)
        return torch.from_numpy(stimuli_current.reshape(self.N,1)).to(DEVICE)
 
    def _stochastic_function(self, u,):
        return 1/self.kwargs.get("stochastic_function_tau", 1) *\
               torch.exp(self.kwargs.get("stochastic_function_b", 1) * (u - self.u_thresh))
 
    
    def _STDP(self, tau, LTP_rate = 1, LTD_rate = -1.5):
        """
        Jesper Sjöström and Wulfram Gerstner (2010), Scholarpedia, 5(2):1362
        http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity
        """
        return (tau > 0) * LTP_rate * torch.exp(-tau/0.01) + (tau < 0) * LTD_rate * torch.exp(tau/0.01)
 
 
 
    def run(self):
        self._reset()
        target_weights = []
        plot_times = []
        plt.ion()
        fig, axs = plt.subplots(3,figsize=(20,20))
        for self.timepoint in tqdm(range(self.total_timepoints)) if self.kwargs.get('process_bar', False)\
            else range(self.total_timepoints):
            self.current = 13*(torch.rand(self.N,1)-0.5)
            self.refractory +=1
            ### update potentials
            ## Numerical stability
            self.potential += 0.5*self.NeuronType()
            self.potential += 0.5*self.NeuronType()
            if self.save_history:
                self.potential_history[:,self.timepoint] = self.potential.ravel()
            if self.neuron_type == 'IZH':
                self.recovery += self.a*(0.2*self.potential-self.recovery)
            else:
                ### Reset currents
                self.current = torch.zeros(self.N,1).to(DEVICE) 
            ### Spikes 
            if self.stochastic_spikes and self.neuron_type != 'IZH':
                spikes_matrix = self._stochastic_function(self.potential) > torch.rand(self.N,1, device =DEVICE)
            else:
                spikes_matrix = self.potential>self.u_thresh   # torch.Size([N, 1])
            self.potential[spikes_matrix] = self.u_rest
            ### Online Learning
            spikes = spikes_matrix.ravel()
            if self.neuron_type == 'IZH':
                self.recovery[spikes] += self.d[spikes]
            self.STDP_trace *= 0.95 #tau = 20ms
            self.STDP_trace[spikes] = 0.1
            ### STDP matrix
            STDP = torch.zeros((self.N, self.N), device=DEVICE)
            ### post-pre spikes (= pre-post connections)
            STDP[spikes,:] = self.AdjacencyMatrix[spikes,:] * - 1.5 * self.STDP_trace.T
            ### pre-post spikes (= post-pre connections)
            STDP[:,spikes] = self.AdjacencyMatrix[:,spikes]  * self.STDP_trace
            #TODO: handle simultaneous pre-post spikes
            ### Update eligibility trace
            self.eligibility_trace += (-self.eligibility_trace/self.tau_eligibility + STDP) * self.dt
            ### Update reward
            if self.reward_function:
                self.reward = self.reward_function(self.dt, self.spike_train, spikes, self.timepoint, self.reward)
            ### Dopamine
            self.dopamine += (-self.dopamine/self.tau_dopamine + self.reward[self.timepoint]) * self.dt
            ### Update weights
            self.weights += self.dopamine * self.eligibility_trace *\
                (1 if self.kwargs.get('plastic_inhibitory', True) else self.excitetory_neurons)
            ### Hard bound:
            self.weights[self.weights * self.excitetory_neurons < 0] = 0.001
            self.weights[self.weights > 1] = 1
            ### Spike train
            self.spike_train[:,self.timepoint] = spikes
            self.refractory *= torch.logical_not(spikes_matrix).to(DEVICE)
            ### Transfer currents + external sources
            new_currents = ((spikes_matrix * self.weights).sum(axis = 0).view(self.N,1) * self.base_current).to(DEVICE)
            open_neurons = self.refractory >= self.refractory_timepoints
            stim_current = self.get_stimuli_current() if self.stimuli else 0
            self.current += (stim_current + new_currents) * open_neurons
            if self.save_history:
                self.current_history[:,self.timepoint] = self.current.ravel()
            if self.timepoint == self.total_timepoints-1 or self.timepoint%1000==0:
                target_weights.append(self.weights[0][1].cpu())
                plot_times.append(self.timepoint/1000)
                y, x = np.where(self.spike_train[:,:self.timepoint].cpu())
                axs[0].clear()
                axs[1].clear()
                axs[2].clear()
                #spike train
                axs[0].plot(x/1000, y, '.', color='black', marker='o', markersize=100/np.sqrt(len(x)))
                axs[0].set_xlim([0,self.timepoint/1000])
                axs[0].set_ylim([0,self.N])
                # weights histogram
                axs[1].hist(self.weights.cpu()[self.weights>0], bins = 100)
                #target synapse's weight and rewarsd
                axs[2].plot(plot_times, target_weights,)
                rewards = np.where(self.reward.cpu() > 0.1)[0]
                axs[2].plot(rewards, np.ones_like(rewards), 'r*')
                axs[2].set_ylim([-0.1,1.1])
                # local 
                fig.canvas.draw()
                fig.canvas.flush_events()
                # google colab:
                # display.clear_output(wait=True)
                # display.display(plt.gcf())
 
 
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
        y, x = np.where(self.spike_train.cpu())
        plt.figure(figsize=(20,4))
        plt.plot(x, y, '.', color='black', marker='o', markersize=2)
        plt.show()
 
    def fICurve(self):
        if self.neuron_type == 'LIF':
            self.critical_current = (self.u_thresh - self.u_rest) / self.Rm
            assert self.base_current < self.critical_current, "Please Increase the base current, it can't fire any neuron"
            self.max_freq = 1 / (self.dt * self.self.refractory_timepoints)
            self.freq_base_current = 1 / self.dt * (self.self.refractory_timepoints + np.log(1 / (1 - self.critical_current / self.base_current)) )
            self.freq = 1 / self.dt * (self.self.refractory_timepoints + np.log(1 / (1 - self.critical_current / self.current)) )