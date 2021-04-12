import numpy as np

class Stimulus:
    def __init__(self, dt, output, neurons):
        self.output = output
        self.neurons = neurons
        self.dt = dt

    def __call__(self, timestep):
        return self.output(timestep * self.dt)


class NeuronGroup:
    def __init__(self, dt, population_size, connection_chance, total_time, stimuli = set(),
    neuron_type = "LIF", **kwargs):
        self.dt = dt
        self.N = population_size
        self.total_timepoints = int(total_time/dt)
        self.stimuli = stimuli
        self.kwargs = kwargs
        self.connection_chance = connection_chance
        self.base_current = kwargs.get('base_current', 1000)
        self.u_thresh = kwargs.get('u_thresh', 30)
        self.u_rest = kwargs.get('u_rest', -68)
        # I think we should change all "refractory" to "refactory". please check :)
        self.refractory_timepoints = kwargs.get('tau_refractory', 0.004) / self.dt
        self.excitatory_chance = kwargs.get('excitatory_chance',  0.8)
        self.refractory = np.ones((self.N,1)) * self.refractory_timepoints
        self.current = np.zeros((self.N,1))
        self.potential = np.ones((self.N,1)) * self.u_rest
        self.spike_train = np.zeros((self.N, self.total_timepoints), dtype= np.bool)
        self.weights = np.random.rand(self.N,self.N)
        np.fill_diagonal(self.weights, 0)
        self.connections = (np.random.rand(*self.weights.shape) + self.connection_chance).astype(np.int)
        # self.excitatory_neurons = (np.random.rand(*self.weights.shape) - 1 + self.excitatory_chance).astype(np.int) 
        # I don't understand your point at line 36 but i think the above implementation is correct. please check :) 
        self.excitatory_neurons = (np.random.rand(*self.weights.shape) + self.excitatory_chance).astype(np.int) * 2 -1 
        self.AdjacencyMatrix = self.connections * self.excitatory_neurons * self.weights


    def LIF(self, Rm = 1, Cm = 0.1):
        """
        Leaky Integrate-and-Fire Neural Model
        """
        Rm = self.kwargs.get("Rm", 1)
        Cm = self.kwargs.get("Cm", 0.1)
        tau_m = Rm*Cm
        exp_term = np.exp(-self.dt/tau_m)
        u_base = (1-exp_term) * self.u_rest
        return u_base + exp_term * self.potential + self.current*self.dt/Cm 

    def get_stimuli_current(self):
        stimuli_current = np.zeros((self.N,1))
        for stimulus in self.stimuli:
            stimuli_current[stimulus.neurons] += stimulus(self.timepoint)
        return stimuli_current
    
    def run(self):
        for self.timepoint in range(self.total_timepoints):
            ### LIF update
            self.refractory +=1
            self.potential = self.LIF()
            ### Reset currents
            self.current = np.zeros((self.N,1)) 
            ### Spikes 
            spikes = self.potential>self.u_thresh
            self.spike_train[:,self.timepoint] = spikes.ravel()
            self.refractory *= np.logical_not(spikes)
            ### Transfer currents + external sources
            new_currents = (spikes * self.AdjacencyMatrix).sum(axis = 0).reshape(self.N,1) * self.base_current
            open_neurons = self.refractory >= self.refractory_timepoints
            self.current += new_currents * open_neurons
            self.current += self.get_stimuli_current() #instead of this we can use :
            # self.current = np.sum(self.stimuli_adjancy * self.stimuli_current,axis = 1, keepdims = True) 

            # Note : We should create self.stimuli_adjancy at the first of simulation 
            # self.stimuli_adjancy = np.array.shape(self.N,len(self.stimuli)) it determines that each stimulus is connected to which neurons with 0 / 1
            # stimuli_current is a list/np.array of stimulus that get timepoint as input and return current of each stimulus, its shape is (1, len(self.stimuli))
            

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

if __name__ == "__main__":
    stimuli = {Stimulus(0.001, lambda t: 10000, [0,1]),
            Stimulus(0.001, lambda t: 200000 * t, [2]),
            Stimulus(0.001, lambda t: 200000 * np.sin(500*t), [3])}

    G = NeuronGroup(dt = 0.001, population_size = 100, connection_chance = 1,
                    total_time = 0.1, stimuli = stimuli, base_current = 10000)
    G.run()
    G.display_spikes()