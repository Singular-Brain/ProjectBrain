import numpy as np

def random_connections(population_size, connection_chance, excitatory_chance = 0.8):
    weights_values = np.random.rand(population_size,population_size)
    np.fill_diagonal(weights_values, 0)
    AdjacencyMatrix = (np.random.rand(population_size, population_size) + connection_chance).astype(np.int)
    excitatory_neurons = (np.random.rand(population_size,1) + excitatory_chance).astype(np.int) * 2 -1 
    weights = AdjacencyMatrix * excitatory_neurons * weights_values
    return weights

def recurrent_layer_wise(layers, recurrent_connection_chance, between_connection_chance, inside_connection_chance, excitatory_chance = 0.8, between_connection_chance_decay=1.0):
    population_size = sum(layers)
    recurrent_weights_values = np.random.rand(population_size,population_size) * (np.random.rand(population_size, population_size) + recurrent_connection_chance).astype(np.int)
    weights =np.tril(recurrent_weights_values,-1)
    index = 0
    between_connection_chance = between_connection_chance
    for layer in range(len(layers)):
        N = layers[layer]
        # inside layer:
        weights_values = np.random.rand(N,N) * (np.random.rand(N, N) + inside_connection_chance).astype(np.int)
        np.fill_diagonal(weights_values, 0)
        weights[index:index+N, index:index+N] = weights_values
        #between layers:
        if layer < len(layers)-1:
            next_N = layers[layer + 1]
            weights_values = np.random.rand(N,next_N) * (np.random.rand(N,next_N) + between_connection_chance).astype(np.int)
            weights[index:index+N, index+N:index+N+next_N] = weights_values
            between_connection_chance *= between_connection_chance_decay
        index+= N  
    excitatory_neurons = (np.random.rand(population_size,1) + excitatory_chance).astype(np.int) * 2 -1       
    return weights * excitatory_neurons
