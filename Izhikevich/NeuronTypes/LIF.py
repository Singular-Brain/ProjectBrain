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
 

 
class NeuronGroup:
    def __init__(self, network, total_time, dt, stimuli = set(),
    biological_plausible = False, reward_function = None, online_learning = False,
     **kwargs):
        self.dt = dt
        self.weights = torch.from_numpy(network).to(DEVICE)
        self.N = len(network)
        self.excitatory_neurons = ((np.sign(self.weights.sum(axis = 1).cpu()) + 1)/2).view(self.N, 1).bool().to(DEVICE)
        self.inhibitory_neurons = ((np.sign(self.weights.sum(axis = 1).cpu()) - 1)/-2).view(self.N, 1).bool().to(DEVICE)
        self.AdjacencyMatrix = torch.from_numpy(network.astype(bool)).to(DEVICE)
        self.total_time = total_time
        self.N_runs = 0
        self.reward_function = reward_function
        self.kwargs = kwargs
        self.total_timepoints = int(total_time/dt)
        ### neurons variables
        if biological_plausible:
            self.kwargs.update(BIOLOGICAL_VARIABLES)        
        self.base_current = self.kwargs.get('base_current', 6E-11)
        self.u_thresh = self.kwargs.get('u_thresh', -55E-3)
        self.u_rest = self.kwargs.get('u_rest', -68E-3)
        self.refractory_timepoints = self.kwargs.get('tau_refractory', 0.002) / self.dt
        self.excitatory_chance = self.kwargs.get('excitatory_chance',  0.8)
        ### neurons variables
        self.refractory = torch.ones((self.N,1), device = DEVICE) * self.refractory_timepoints
        self.reward = torch.zeros(self.total_timepoints, device = DEVICE)
        self.current = torch.zeros((self.N,1), device = DEVICE)
        self.potential = torch.ones((self.N,1), device = DEVICE) * self.u_rest
        self.spike_train = torch.zeros((self.N, self.total_timepoints), dtype= torch.bool)
        ### online learning
        self.online_learning = online_learning
        if self.online_learning:
            self.STDP_trace = torch.zeros((self.N,1), device = DEVICE)
            self.STDP = torch.zeros((self.N, self.N), device=DEVICE)
            self.tau_eligibility = self.kwargs.get('tau_eligibilit', 1)
            self.tau_dopamine = self.kwargs.get('tau_dopamine', 0.2)
            self.dopamine = 0
            self.eligibility_trace = torch.zeros((self.N,self.N), device = DEVICE)
        ### online plot
        self.setup_online_plot = self.kwargs.get('setup_online_plot',  False)
        self.update_online_plot = self.kwargs.get('update_online_plot',  False)
        self.online_plot = self.setup_online_plot and self.update_online_plot
        if (self.setup_online_plot==False) ^ (self.update_online_plot==False):
            print("To use 'online plot' set both 'setup_online_plot' and 'update_online_plot' functions!")
        ### save history
        self.save_history = self.kwargs.get('save_history',  False)
        if self.save_history:
            self.current_history = torch.zeros((self.N, self.total_timepoints), dtype= torch.float32).to(DEVICE)
            self.potential_history = torch.zeros((self.N, self.total_timepoints), dtype= torch.float32).to(DEVICE)
        ### save to file
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
        ### stimuli
        self.stimuli = np.array(list(stimuli))
        self.StimuliAdjacency = np.zeros((self.N, len(stimuli)),dtype=np.bool)
        self.stochastic_spikes = self.kwargs.get('stochastic_spikes', False)
        for i, stimulus in enumerate(self.stimuli):
            self.StimuliAdjacency[stimulus.neurons, i] = True
        ### Neuron type variables:
        self.Cm = self.kwargs.get("Cm", 14.7E-12)
        Rm = self.kwargs.get("Rm", 135E6)
        self.tau_m = self.Cm * Rm 
        # self.exp_term = np.exp(-self.dt/self.tau_m)
        # self.u_base = (1-self.exp_term) * self.u_rest
 
    def IF(self):
        """
        Simple Perfect Integrate-and-Fire Neural Model:
        Assumes a fully-insulated memberane with resistance approaching infinity
        """
        return self.potential + (self.dt * self.current)/self.Cm
 
    def update_potential(self):
        """
        Leaky Integrate-and-Fire Neural Model
        """
        #return self.u_base + self.exp_term * self.potential + self.current*self.dt/self.Cm 
        return (-self.dt/self.tau_m)*(self.potential-self.u_rest) + self.dt * self.current / self.Cm
 
 
    def get_stimuli_current(self):
        call_stimuli =  np.vectorize(lambda stim: stim())
        stimuli_output = call_stimuli(self.stimuli)
        stimuli_current = (stimuli_output * self.StimuliAdjacency).sum(axis = 1)
        return torch.from_numpy(stimuli_current.reshape(self.N,1)).to(DEVICE)
 
    def _stochastic_function(self, u,):
        return 1/self.kwargs.get("stochastic_function_tau", 1) *\
               torch.exp(self.kwargs.get("stochastic_function_b", 1) * (u - self.u_thresh))
 
    
    def _update_STDP(self, spikes, LTP_rate = 1, LTD_rate = -1.5):
        # reset STDP
        self.STDP.fill_(0)
        ### post-pre spikes (= pre-post connections)
        self.STDP[spikes,:] = LTD_rate * self.AdjacencyMatrix[spikes,:] * self.STDP_trace.T
        ### pre-post spikes (= post-pre connections)
        self.STDP[:,spikes] = LTP_rate * self.AdjacencyMatrix[:,spikes] * self.STDP_trace 
        #TODO: handle simultaneous pre-post spikes
 
    def run(self):
        self._reset()
        if self.online_plot:
            fig, axs = self.setup_online_plot(self)
        for self.timepoint in tqdm(range(self.total_timepoints)) if self.kwargs.get('process_bar', False)\
            else range(self.total_timepoints):
            self.refractory +=1
            ### update potentials
            self.potential += 0.5*self.update_potential() # To Improve approximation 
            self.potential += 0.5*self.update_potential() # and numerical stability
            if self.save_history:
                self.potential_history[:,self.timepoint] = self.potential.ravel()
            ### Reset currents
            self.current = torch.zeros(self.N,1).to(DEVICE) 
            ### Spikes 
            if self.stochastic_spikes:
                spikes_matrix = self._stochastic_function(self.potential) > torch.rand(self.N,1, device =DEVICE)
            else:
                spikes_matrix = self.potential>self.u_thresh   # torch.Size([N, 1])
            spikes = spikes_matrix.ravel()
            self.potential[spikes_matrix] = self.u_rest
            ### Online Learning
            if self.online_learning:
                self.STDP_trace *= 0.95 #tau = 20ms
                self.STDP_trace[spikes] = 0.1
                ### update STDP matrix
                self._update_STDP(spikes)
                ### Update eligibility trace
                self.eligibility_trace += (-self.eligibility_trace/self.tau_eligibility) * self.dt + self.STDP
                ### Update reward
                if self.reward_function:
                    self.reward = self.reward_function(self, spikes)
                ### Dopamine
                self.dopamine += (-self.dopamine/self.tau_dopamine ) * self.dt + self.reward[self.timepoint]
                ### Update weights
                self.weights += (0.002+self.dopamine) * self.eligibility_trace *\
                    (1 if self.kwargs.get('plastic_inhibitory', True) else self.excitatory_neurons)
                ### Hard bound:
                self.weights[(self.weights * self.excitatory_neurons) < 0] = 0.001
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
            if self.online_plot:
                self.update_online_plot(self, fig, axs)


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
        self.critical_current = (self.u_thresh - self.u_rest) / self.Rm
        assert self.base_current < self.critical_current, "Please Increase the base current, it can't fire any neuron"
        self.max_freq = 1 / (self.dt * self.self.refractory_timepoints)
        self.freq_base_current = 1 / self.dt * (self.self.refractory_timepoints + np.log(1 / (1 - self.critical_current / self.base_current)) )
        self.freq = 1 / self.dt * (self.self.refractory_timepoints + np.log(1 / (1 - self.critical_current / self.current)) )