from abc import ABC, abstractmethod
import torch
import numpy as np
from math import ceil
from itertools import count
from tqdm import tqdm
from .callbacks import CallbackList
# set DEVICE
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # set DEVICE
print(f'Device is set to {DEVICE.upper()}')
 
#set manual seed
def manual_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
SEED = 2045
#manual_seed(SEED) # We set the seed to 2045 because the Singularity is near!

class Network:
    def __init__(self, total_time, dt, learning_rule = None,
                 callbacks = [], save_history = False):
        self.groups = []
        self.connections = []
        self.groups_connected_to_stimulus = []
        self.architecture()
        self.subNetworks = self.groups + self.connections
        ###
        self.dt = dt
        self.total_time = total_time
        self.N_runs = 0
        self.total_timepoints = int(total_time/dt)
        ### callbacks
        if isinstance(callbacks, CallbackList):
            self.callbacks = callbacks
        else:
            self.callbacks = CallbackList(callbacks)
        for callback in callbacks:
            callback.set_NeuronGroup(self)
        ### online learning
        self.learning_rule = learning_rule
        if self.learning_rule:
            for subNetwork in self.subNetworks:
                self.learning_rule.set_params(self.dt, subNetwork)
        ### save history
        self.save_history = save_history
    
    
    @abstractmethod
    def architecture(self):
        ...

    def _get_stimuli_current(self, subNetwork):
        if not isinstance(subNetwork,NeuronGroup):
            return
        if subNetwork not in self.groups_connected_to_stimulus:
            return 0
        call_stimuli =  np.vectorize(lambda stim: stim(self.timepoint))
        stimuli_output = call_stimuli(subNetwork.stimuli)
        stimuli_current = (stimuli_output * subNetwork.StimuliAdjacency).sum(axis = 1)
        return torch.from_numpy(stimuli_current).to(DEVICE)
 
 
    def _make_stimuli_adjacency(self, stimuli): #TODO: add a warning if there are stimulus but no groups_connected_to_stimulus
        if self.groups_connected_to_stimulus:
            assert type(stimuli)== list and len(stimuli) == len(self.groups_connected_to_stimulus), "Input 'stimuli' must be a list with the same length as the model's groups connected to stimulus."
            self.stimuli = zip(self.groups_connected_to_stimulus, stimuli)
            for group, stim in self.stimuli:
                group.stimuli = np.array(list(stim))
                group.StimuliAdjacency = np.zeros((group.N, len(stim)),dtype=np.bool)
                for i, stimulus in enumerate(stim):
                    group.StimuliAdjacency[stimulus.neurons, i] = True

    def _reset(self):
        self.N_runs +=1
        for subNetwork in self.subNetworks:
            subNetwork._reset(self.total_timepoints, self.save_history)


    def run(self, stimuli = None, progress_bar = False):
        self._reset()
        self._make_stimuli_adjacency(stimuli)
        self.callbacks.on_run_start(self.N_runs)
        for self.timepoint in tqdm(range(self.total_timepoints)) if progress_bar \
        else range(self.total_timepoints):
            self.callbacks.on_timepoint_start(self.timepoint)
            for subNetwork in self.subNetworks:
                subNetwork.stim_current = self._get_stimuli_current(subNetwork)
                self.callbacks.on_subNetwork_start(subNetwork, self.timepoint)
                subNetwork._run_one_timepoint(self.dt, self.timepoint, self.save_history)
                self.callbacks.on_subNetwork_end(subNetwork, self.timepoint)
            self.callbacks.on_timepoint_end(self.timepoint)
            ### handle external rewards with callbacks
            if self.learning_rule:
                self.learning_rule.update_neuromodulators()
                for subNetwork in self.subNetworks:
                    self.learning_rule(subNetwork)
                self.learning_rule.zero_released_neuromodulators()
        self.callbacks.on_run_end(self.N_runs)
 



    def randomConnect(self, from_group, to_group, connection_chance):
        self.connections.append(RandomConnect(from_group, to_group, connection_chance))

    def randomConnections(self, population_size, neuronType, connection_chance,
                          name = None, excitatory_ratio = 0.8, scale_factor = 1,
                          **kwargs):
        neuron_group = RandomConnections(population_size, neuronType, connection_chance, name,
                                             excitatory_ratio, scale_factor, **kwargs)
        self.groups.append(neuron_group)
        return neuron_group

    def connectStimulus(self, group):
        assert isinstance(group,NeuronGroup), "Stimulus can only be connected to a group"
        self.groups_connected_to_stimulus.append(group)

class NeuronGroup:
    def __init__(self, neuronType, stochastic_spikes = True, base_current = 6E-11,
                 stochastic_function_tau = 1, stochastic_function_b = 1):
        ### Neuron Type
        self.neuronType = neuronType
        ### neurons variables
        self.base_current = base_current
        self.stim_current = 0
        ###  stochastic spike
        self.stochastic_spikes = stochastic_spikes
        self.stochastic_function_tau = stochastic_function_tau
        self.stochastic_function_b = stochastic_function_b
    

    def _reset(self, total_timepoints, save_history):
        self.refractory = torch.ones(self.N, device = DEVICE) * 1000 #initialize at a high value
        self.current = torch.zeros(self.N, device = DEVICE)
        self.potential = torch.ones(self.N, device = DEVICE) * self.neuronType.u_rest
        self.spikes = torch.zeros((self.N,1),dtype=torch.bool, device = DEVICE)
        self.spike_train = torch.zeros((self.N, total_timepoints),dtype=torch.bool, device = DEVICE)
        if save_history:
            self.current_history = torch.zeros((self.N, total_timepoints), dtype= torch.float32, device = DEVICE)
            self.potential_history = torch.zeros((self.N, total_timepoints), dtype= torch.float32, device = DEVICE)



    def update_potential(self, dt, repeat = 2):
        """Update the potential for neurons in the sub-network

        Args:
            dt (float): global dt
            repeat (int, optional): Repeat the calculation with N steps to Improve approximation and numerical stability. Defaults to 2.
        """
        for _ in range(repeat):
            self.potential += (1/repeat) * self.neuronType(dt, self.potential, self.current)

    def update_current(self, dt,):
        new_currents = ((self.spikes.reshape(self.N,1) * self.weights).sum(axis = 0) * self.base_current).to(DEVICE)
        self.current = (self.stim_current + new_currents) * (self.refractory >= (self.neuronType.tau_refractory/dt)) # implement current for open neurons 

    def _stochastic_function(self,):
        return 1/self.stochastic_function_tau*torch.exp(self.stochastic_function_b * (self.potential - self.neuronType.u_thresh))

    def generate_spikes(self):
        if self.stochastic_spikes:
            self.spikes = self._stochastic_function() > torch.rand(self.N, device =DEVICE)
        else:
            self.spikes = self.potential>self.neuronType.u_thresh   # torch.Size([N,])
        ### Reset spiked neurons' potential
        self.potential[self.spikes] = self.neuronType.u_rest
        ### Reset the refractory of spiked neurons
        self.refractory *= torch.logical_not(self.spikes).to(DEVICE)


    def _run_one_timepoint(self, dt, timepoint, save_history):
        self.refractory +=1
        ### update potentials
        self.update_potential(dt)  
        if save_history:
            self.potential_history[:,timepoint] = self.potential
        ### Spikes 
        self.generate_spikes()
        ### Spike train
        self.spike_train[:,timepoint] = self.spikes
        ### Transfer currents + external sources
        self.update_current(dt)
        ### save current
        if save_history:
            self.current_history[:,timepoint] = self.current
        



class RandomConnections(NeuronGroup):
    _ids = count(0)
    def __init__(self, population_size, neuronType, connection_chance,
                 name = None, excitatory_ratio = 0.8, scale_factor = 1,
                 **kwargs):
        super().__init__(neuronType, **kwargs)
        self.id = next(self._ids)
        self.name = f'random_connections_{self.id}' if name is None else name
        self.N = population_size
        self.adjacency_matrix = (torch.rand((self.N, self.N), device = DEVICE) + connection_chance).int().bool()
        self.excitatory_neurons = torch.tensor([1]* int(excitatory_ratio * population_size) + [0]* (ceil((1-excitatory_ratio) * population_size)),
                                                device= DEVICE, dtype= torch.bool)
        self.inhibitory_neurons = torch.tensor([0]* int(excitatory_ratio * population_size) + [1]* (ceil((1-excitatory_ratio) * population_size)),
                                                device= DEVICE, dtype= torch.bool)
        weights_values = np.random.rand(self.N,self.N) * scale_factor
        np.fill_diagonal(weights_values, 0)
        self.weights = self.adjacency_matrix * torch.from_numpy(weights_values).to(DEVICE)
        self.weights[self.inhibitory_neurons] *= -1


class Connection(ABC):
    def __init__(self, from_group, to_group):
        self.from_group = from_group
        self.to_group = to_group


    def update_potential(self, dt, repeat = 2):
        """Update the potential for neurons in the sub-network

        Args:
            dt (float): global dt
            repeat (int, optional): Repeat the calculation with N steps to Improve approximation and numerical stability. Defaults to 2.
        """
        for _ in range(repeat):
            self.potential += (1/repeat) * self.neuronType(dt, self.potential, self.current)

class RandomConnect(Connection):
    def __init__(self, from_group, to_group, connection_chance):
        super.__init__(from_group, to_group)
        self.connection_chance = connection_chance
        self.adjacency_matrix = (torch.rand((self.from_group.N, self.to_group.N), device = DEVICE) + connection_chance).int().bool()
        weights_values = torch.rand((self.from_group.N, self.to_group.N), device = DEVICE)
        weights_values[self.from_group.inhibitory_neurons] *= -1
        self.weights = weights_values * self.adjacency_matrix




def random_connections(population_size, connection_chance, excitatory_ratio = 0.8):
    weights_values = np.random.rand(population_size,population_size)
    np.fill_diagonal(weights_values, 0)
    adjacency_matrix = (np.random.rand(population_size, population_size) + connection_chance).astype(np.int)
    excitatory_neurons = np.array([1]* int(excitatory_ratio * population_size) + [-1]* (ceil((1-excitatory_ratio) * population_size))).reshape((population_size, 1))
    weights = adjacency_matrix * excitatory_neurons * weights_values
    return weights

def uniform_connections(population_size, connection_chance, excitetory_weight = 0.25, inhibitory_weight = -0.25, excitatory_ratio = 0.8):
    adjacency_matrix = (np.random.rand(population_size, population_size) + connection_chance).astype(np.int)
    excitatory_neurons = np.array([excitetory_weight]* int(excitatory_ratio * population_size) + [inhibitory_weight]* (ceil((1-excitatory_ratio) * population_size))).reshape((population_size, 1))
    weights = adjacency_matrix * excitatory_neurons 
    np.fill_diagonal(weights, 0)
    return weights
    
def recurrent_layer_wise(layers, recurrent_connection_chance,
    between_connection_chance, inside_connection_chance, excitatory_ratio = 0.8,):
    '''
    recurrent_connection_chance: 'float'
    between_connection_chance: 'float' or 'list of floats' 
    inside_connection_chance: 'float' or 'list of floats'
    excitatory_ratio: 'float' or 'list of floats'
    '''
    N_layers = len(layers)
    between_connection_chance = [between_connection_chance] * N_layers if type(between_connection_chance) != list else between_connection_chance
    inside_connection_chance = [inside_connection_chance] * N_layers if type(inside_connection_chance) != list else inside_connection_chance
    excitatory_ratio = [excitatory_ratio] * N_layers if type(excitatory_ratio) != list else excitatory_ratio
    population_size = sum(layers)
    recurrent_weights_values = np.random.rand(population_size,population_size) * (np.random.rand(population_size, population_size) + recurrent_connection_chance).astype(np.int)
    weights =np.tril(recurrent_weights_values,-1)
    index = 0
    excitatory_neurons = []
    for i, (N, between_chance, inside_chance, excitatory) in enumerate(zip(layers, between_connection_chance, inside_connection_chance,excitatory_ratio)):
        # excitatory neurons
        excitatory_neurons += [1]* int(excitatory * N) + [-1]* (ceil((1-excitatory) * N))
        # inside layer:
        weights_values = np.random.rand(N,N) * (np.random.rand(N, N) + inside_chance).astype(np.int)
        np.fill_diagonal(weights_values, 0)
        weights[index:index+N, index:index+N] = weights_values
        #between layers:
        if i < len(layers)-1:
            next_N = layers[i + 1]
            weights_values = np.random.rand(N,next_N) * (np.random.rand(N,next_N) + between_chance).astype(np.int)
            weights[index:index+N, index+N:index+N+next_N] = weights_values
        index+= N  
    excitatory_neurons = np.array(excitatory_neurons).reshape(population_size, 1)
    return weights * excitatory_neurons
