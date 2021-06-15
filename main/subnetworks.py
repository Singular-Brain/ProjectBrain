from math import ceil
from itertools import count
import torch
import numpy as np
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # set DEVICE


class NeuronGroup:
    order = 0
    type = 'Neuron Group'

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
    

    def _reset(self, dt, total_timepoints, save_history):
        self.dt = dt
        self.refractory = torch.ones(self.N, device = DEVICE) * 1000 #initialize at a high value
        self.current = torch.zeros(self.N, device = DEVICE)
        self.potential = torch.ones(self.N, device = DEVICE) * self.neuronType.u_rest
        self.spikes = torch.zeros((self.N,1),dtype=torch.bool, device = DEVICE)
        self.spike_train = torch.zeros((self.N, total_timepoints),dtype=torch.bool, device = DEVICE)
        self.save_history = save_history
        if save_history:
            self.current_history = torch.zeros((self.N, total_timepoints), dtype= torch.float32, device = DEVICE)
            self.potential_history = torch.zeros((self.N, total_timepoints), dtype= torch.float32, device = DEVICE)


    def _update_potential(self, repeat = 2):
        """Update the potential for neurons in the sub-network

        Args:
            repeat (int, optional): Repeat the calculation with N steps to Improve approximation and numerical stability. Defaults to 2.
        """
        for _ in range(repeat):
            self.potential += (1/repeat) * self.neuronType(self.dt, self.potential, self.current)


    def _update_current(self,):
        new_currents = ((self.spikes.reshape(self.N,1) * self.weights).sum(axis = 0) * self.base_current).to(DEVICE)
        self.current = (self.stim_current + new_currents) * (self.refractory >= (self.neuronType.tau_refractory/self.dt)) # implement current for open neurons 


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


    def _run_one_timepoint(self, timepoint,):
        self.refractory +=1
        ### update potentials
        self._update_potential()  
        if self.save_history:
            self.potential_history[:,timepoint] = self.potential
        ### Spikes 
        self.generate_spikes()
        ### Spike train
        self.spike_train[:,timepoint] = self.spikes
        ### Transfer currents + external sources
        self._update_current()
        ### save current
        if self.save_history:
            self.current_history[:,timepoint] = self.current
        

    @property
    def weight_values(self):
        return self.weight[self.weight != 0]

    def excitetory_potential_change(self, span = slice(None)):
        return (((self.potential_history[span,:] - self.neuronType.u_rest).sum(axis = 1))[self.excitatory_neurons]).sum()

    def inhibitory_potential_change(self, span = slice(None)):
        return (((self.potential_history[span,:] - self.neuronType.u_rest).sum(axis = 1))[self.inhibitory_neurons]).sum()
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


class Connection:
    order = 1
    type = 'Connection'

    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.excitatory_neurons = source.excitatory_neurons
        self.inhibitory_neurons = source.inhibitory_neurons

    def _update_current(self,):
        new_currents = ((self.source.spikes.reshape(self.source.N,1) * self.weights).sum(axis = 0) * self.source.base_current).to(DEVICE)
        self.current = new_currents * (self.destination.refractory >= (self.destination.neuronType.tau_refractory/self.dt)) # implement current for open neurons 

    def _run_one_timepoint(self, timepoint,):
        ### update the external current from 'source'
        self._update_current()
        ### transfer the new current to 'destination'
        self.destination.current += self.current
        ### save current
        if self.save_history:
            self.destination.current_history[:,timepoint] = self.destination.current


    def _reset(self, dt, total_timepoints, save_history):
        self.dt = dt
        self.save_history = save_history

    @property
    def weight_values(self):
        return self.weight[self.weight != 0]

    def EPSP(self, span = slice(None)):
        return self.source.excitetory_potential_change(span)

    def IPSP(self, span = slice(None)):
        return self.source.inhibitory_potential_change(span)

class RandomConnect(Connection,):
    _ids = count(0)

    def __init__(self, source, destination, connection_chance, name = None):
        super().__init__(source, destination)
        self.id = next(self._ids)
        self.name = f'random_connect_{self.id}' if name is None else name
        self.connection_chance = connection_chance
        self.adjacency_matrix = (torch.rand((self.source.N, self.destination.N), device = DEVICE) + connection_chance).int().bool()
        weights_values = torch.rand((self.source.N, self.destination.N), device = DEVICE)
        weights_values[self.source.inhibitory_neurons] *= -1
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
