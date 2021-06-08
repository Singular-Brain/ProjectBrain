from abc import ABC, abstractmethod
import torch
import numpy as np
from math import ceil
from itertools import count
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # set DEVICE

class Network:
    def __init__(self):
        self._connections = []
        self.architecture()
        self._init_network()

    @abstractmethod
    def architecture(self):
        ...

    def _init_network(self):
        idx = 0
        self._groups = []
        excitatory_neurons = []
        inhibitory_neurons = []
        for attr_name,attr in self.__dict__.items():
            if isinstance(attr, NeuronGroup):
                attr.name = attr_name
                attr.idx = slice(idx, idx + attr.N)
                idx += attr.N
                excitatory_neurons += attr.excitatory_neurons.tolist()
                inhibitory_neurons += attr.inhibitory_neurons.tolist()
                self._groups.append(attr)
        self.total_neurons = sum([group.N for group in self._groups])
        self.weights = torch.zeros((self.total_neurons, self.total_neurons), device = DEVICE)
        for group in self._groups:
            self.weights[group.idx, group.idx] = group.weights
        for connection in self._connections:
            self.weights[connection.from_.idx, connection.to.idx] = connection.weights
        self.adjacency_matrix = self.weights.bool()
        self.excitatory_neurons = torch.tensor(excitatory_neurons, dtype= torch.bool, device = DEVICE)
        self.inhibitory_neurons = torch.tensor(inhibitory_neurons, dtype= torch.bool, device = DEVICE)


    def randomConnect(self, from_, to, connection_chance):
        self._connections.append(RandomConnect(from_, to, connection_chance))

class Connection(ABC):
    ...

class RandomConnect(Connection):
    def __init__(self, from_, to, connection_chance):
        self.from_ = from_
        self.to = to
        self.connection_chance = connection_chance
        self.adjacency_matrix = (torch.rand((self.from_.N, self.to.N), device = DEVICE) + connection_chance).int().bool()
        weights_values = torch.rand((self.from_.N, self.to.N), device = DEVICE)
        weights_values[self.from_.inhibitory_neurons] *= -1
        self.weights = weights_values * self.adjacency_matrix

class NeuronGroup(ABC):
    @property
    def weight_values(self):
        return self.weights[self.weights != 0]

class RandomConnections(NeuronGroup):
    def __init__(self, population_size, connection_chance, excitatory_ratio = 0.8,):
        self.name = None
        self.N = population_size
        self.adjacency_matrix = (torch.rand((population_size, population_size), device = DEVICE) + connection_chance).int().bool()
        self.excitatory_neurons = torch.tensor([1]* int(excitatory_ratio * population_size) + [0]* (ceil((1-excitatory_ratio) * population_size)),
                                                device= DEVICE, dtype= torch.bool)
        self.inhibitory_neurons = torch.tensor([0]* int(excitatory_ratio * population_size) + [1]* (ceil((1-excitatory_ratio) * population_size)),
                                                device= DEVICE, dtype= torch.bool)
        weights_values = np.random.rand(population_size,population_size)
        np.fill_diagonal(weights_values, 0)
        self.weights = self.adjacency_matrix * torch.from_numpy(weights_values).to(DEVICE)
        self.weights[self.inhibitory_neurons] *= -1


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
