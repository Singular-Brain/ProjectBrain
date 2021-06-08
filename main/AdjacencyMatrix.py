import numpy as np
from math import ceil

def random_connections(population_size, connection_chance, excitatory_ratio = 0.8):
    weights_values = np.random.rand(population_size,population_size)
    np.fill_diagonal(weights_values, 0)
    AdjacencyMatrix = (np.random.rand(population_size, population_size) + connection_chance).astype(np.int)
    excitatory_neurons = np.array([1]* int(excitatory_ratio * population_size) + [-1]* (ceil((1-excitatory_ratio) * population_size))).reshape((population_size, 1))
    weights = AdjacencyMatrix * excitatory_neurons * weights_values
    return weights

def uniform_connections(population_size, connection_chance, excitetory_weight = 0.25, inhibitory_weight = -0.25, excitatory_ratio = 0.8):
    AdjacencyMatrix = (np.random.rand(population_size, population_size) + connection_chance).astype(np.int)
    excitatory_neurons = np.array([excitetory_weight]* int(excitatory_ratio * population_size) + [inhibitory_weight]* (ceil((1-excitatory_ratio) * population_size))).reshape((population_size, 1))
    weights = AdjacencyMatrix * excitatory_neurons 
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
