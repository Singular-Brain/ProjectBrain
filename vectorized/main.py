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
        self.kwargs = kwargs
        self.connection_chance = connection_chance
        self.base_current = kwargs.get('base_current', 1E-9)
        self.u_thresh = kwargs.get('u_thresh', 35E-3)
        self.u_rest = kwargs.get('u_rest', -63E-3)
        self.refractory_timepoints = kwargs.get('tau_refractory', 0.002) / self.dt
        self.excitatory_chance = kwargs.get('excitatory_chance',  0.8)
        self.refractory = np.ones((self.N,1))*self.refractory_timepoints
        self.current = np.zeros((self.N,1))
        self.potential = np.ones((self.N,1)) * self.u_rest
        self.spike_train = np.zeros((self.N, self.total_timepoints), dtype= np.bool)
        self.weights = np.random.rand(self.N,self.N)
        np.fill_diagonal(self.weights, 0)
        self.connections = (np.random.rand(*self.weights.shape) + self.connection_chance).astype(np.int)
        self.excitatory_neurons = (np.random.rand(*self.weights.shape) + self.excitatory_chance).astype(np.int) * 2 -1 
        self.AdjacencyMatrix = self.connections * self.excitatory_neurons * self.weights
        self.stimuli = np.array(list(stimuli))
        self.StimuliAdjacency = np.zeros((self.N, len(stimuli)), dtype=np.bool)
        for i, stimulus in enumerate(self.stimuli):
            self.StimuliAdjacency[stimulus.neurons, i] = True

    def IF(self):
        """
        Simple Perfect Integrate-and-Fire Neural Model:
        Assumes a fully-insulated memberane with resistance approaching infinity
        """
        Cm = self.kwargs.get("Cm", 14.7E-12) 
        return self.potential + (self.dt * self.current)/Cm

    def LIF(self):
        """
        Leaky Integrate-and-Fire Neural Model
        """
        Rm = self.kwargs.get("Rm", 135E6)
        Cm = self.kwargs.get("Cm", 14.7E-12)
        tau_m = Rm*Cm
        exp_term = np.exp(-self.dt/tau_m)
        u_base = (1-exp_term) * self.u_rest
        return u_base + exp_term * self.potential + self.current*self.dt/Cm 

    def get_stimuli_current(self):
        call_stimuli =  np.vectorize(lambda stim: stim(self.timepoint))
        stimuli_output = call_stimuli(self.stimuli)
        stimuli_current = (stimuli_output * self.StimuliAdjacency).sum(axis = 1)
        return stimuli_current.reshape(self.N,1)
    
    def run(self):
        self._reset()
        for self.timepoint in range(self.total_timepoints):
            ### LIF update
            self.refractory +=1
            self.potential = self.LIF()
            ### Reset currents
            self.current = np.zeros((self.N,1)) 
            ### Spikes 
            spikes = self.potential>self.u_thresh
            self.potential[spikes] = self.u_rest
            self.spike_train[:,self.timepoint] = spikes.ravel()
            self.refractory *= np.logical_not(spikes)
            ### Transfer currents + external sources
            new_currents = (spikes * self.AdjacencyMatrix).sum(axis = 0).reshape(self.N,1) * self.base_current
            open_neurons = self.refractory >= self.refractory_timepoints
            self.current += new_currents * open_neurons
            self.current += self.get_stimuli_current() * open_neurons

    def _reset(self):
        self.refractory = np.ones((self.N,1))*self.refractory_timepoints
        self.current = np.zeros((self.N,1))
        self.potential = np.ones((self.N,1)) * self.u_rest
        self.spike_train = np.zeros((self.N, self.total_timepoints), dtype= np.bool)


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
        self.AdjacencyMatrix = NeuronGroup.AdjacencyMatrix
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
        padded_spike_train = np.pad(spike_train,self.interval_timepoints)[self.interval_timepoints: -self.interval_timepoints,:]
        for i in range(self.total_timepoints + self.interval_timepoints):
            section = padded_spike_train[:,i:i+self.interval_timepoints]
            span = section.sum(axis = 1).astype(np.bool)
            first = padded_spike_train[:,i]
            last = padded_spike_train[:,i+self.interval_timepoints]
            if reward:
                self.AdjacencyMatrix += self.reward_pre_post_rate * (first * span.reshape(1, self.N) * self.AdjacencyMatrix)
                self.AdjacencyMatrix += self.reward_post_pre_rate * (span  * last.reshape(1, self.N) * self.AdjacencyMatrix)
            if not reward:
                self.AdjacencyMatrix += self.pre_post_rate * (first * span.reshape(1, self.N) * self.AdjacencyMatrix)
                self.AdjacencyMatrix += self.post_pre_rate * (span  * last.reshape(1, self.N) * self.AdjacencyMatrix)


if __name__ == "__main__":
    stimuli = {
            Stimulus(0.001, lambda t: .7E-9, [0,1]),
            Stimulus(0.001, lambda t: 20E-9 * t, [2]),
            Stimulus(0.001, lambda t: 1E-9 * np.sin(500*t), [3])
            }

    G = NeuronGroup(dt = 0.001, population_size = 6, connection_chance = 1/3,
                    total_time = 0.1, stimuli = stimuli, base_current = 1E-9)
    G.run()
    G.display_spikes()
    print(G.AdjacencyMatrix)
    learning = RFSTDP(G)
    learning(reward = True)
    print(G.AdjacencyMatrix)