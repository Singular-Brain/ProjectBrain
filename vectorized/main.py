import numpy as np
### Visualization
import networkx as nx 
import matplotlib.pyplot as plt
import bokeh
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine
from bokeh.plotting import from_networkx, figure


BIOLOGICAL_VARIABLES = {
    'base_current': 1E-9,
    'u_thresh': 35E-3,
    'u_rest': -63E-3,
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
    def __init__(self, dt, population_size, connection_chance, total_time, stimuli = set(),
    neuron_type = "LIF", biological_plausible = False, **kwargs):
        self.dt = dt
        self.N = population_size
        self.total_timepoints = int(total_time/dt)
        self.kwargs = kwargs
        if biological_plausible:
            self.kwargs = {key: BIOLOGICAL_VARIABLES.get(key, self.kwargs[key]) for key in self.kwargs}
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
        weights_values = np.random.rand(self.N,self.N)
        np.fill_diagonal(weights_values, 0)
        self.AdjacencyMatrix = (np.random.rand(self.N, self.N) + self.connection_chance).astype(np.int)
        self.excitatory_neurons = (np.random.rand(self.N, self.N) + self.excitatory_chance).astype(np.int) * 2 -1 
        self.weights = self.AdjacencyMatrix * self.excitatory_neurons * weights_values
        self.stimuli = np.array(list(stimuli))
        self.StimuliAdjacency = np.zeros((self.N, len(stimuli)), dtype=np.bool)
        for i, stimulus in enumerate(self.stimuli):
            self.StimuliAdjacency[stimulus.neurons, i] = True
        ### Neuron type variables:
        if neuron_type == 'IF':
            self.Cm = self.kwargs.get("Cm", 14.7E-12)
        elif neuron_type == 'LIF':
            self.Cm = self.kwargs.get("Cm", 14.7E-12)
            Rm = self.kwargs.get("Rm", 135E6)
            tau_m = self.Cm * Rm 
            self.exp_term = np.exp(-self.dt/tau_m)
            self.u_base = (1-self.exp_term) * self.u_rest

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
        return self.u_base + self.exp_term * self.potential + self.current*self.dt/self.Cm 

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
            new_currents = (spikes * self.weights).sum(axis = 0).reshape(self.N,1) * self.base_current
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

    def _set_pos(self, graph, input_neurons):
        pos = {}
        for y, neuron in enumerate(input_neurons):
            pos[neuron] = (0, (y+1) / (len(input_neurons)+1))
        last_layer = input_neurons
        for x in range(1, len(graph.nodes())):
            counted_neurons = set(pos.keys()) 
            next_layer = set()
            for neuron in last_layer:
                next_layer.update(graph.successors(neuron))
            new_neurons = next_layer - counted_neurons
            if new_neurons == set():
                break
            else:
                for y, neuron in enumerate(new_neurons):
                    pos[neuron] = (x, (y+1)/(len(new_neurons)+1))
            last_layer = next_layer
        not_in_input_successors =set(graph.nodes()) - set(pos.keys()) 
        if not_in_input_successors:
            for y, neuron in enumerate(not_in_input_successors):
                    pos[neuron] = (x, (y+1)/(len(not_in_input_successors)+1))
        return pos

    def show_graph(self, input_neurons, **kwargs):
        rows, cols = np.where(self.AdjacencyMatrix == 1)
        edges = zip(rows.tolist(), cols.tolist())
        graph = nx.DiGraph()
        graph.add_edges_from(edges)
        pos = self._set_pos(graph, input_neurons)
        spiked_neurones = dict([(i[0], i[1]) for i in enumerate((self.spike_train[:,self.timepoint]).astype(str))])
        nx.set_node_attributes(graph, name='spiked_neurons', values=spiked_neurones)        
        HOVER_TOOLTIPS = [("Neuron", "@index")]
        plot = figure(tooltips = HOVER_TOOLTIPS,
        tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
            x_range=Range1d(-.1, sorted(pos.values())[-1][0]+0.1),
            y_range=Range1d(0, 1), title='Neuron Group')
        network_graph = from_networkx(graph, pos, scale=10, center=(0, 0))
        print(network_graph.node_renderer.data_source.data['spiked_neurons'])
        #Set node size and color
        network_graph.node_renderer.glyph = Circle(size=15, 
        fill_color=bokeh.transform.factor_cmap('spiked_neurons', palette=bokeh.palettes.cividis(2), factors=['False', 'True']))
        #Set edge opacity and width
        network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)
        #Add network graph to the plot
        plot.renderers.append(network_graph)
        bokeh.io.show(plot)

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
        self.weights = NeuronGroup.weights
        self.interval_timepoints = int(interval_time / NeuronGroup.dt) #timepoints
        self.total_timepoints = NeuronGroup.total_timepoints
        self.spike_train = NeuronGroup.spike_train
        self.reward_based = True
        self.pre_post_rate = pre_post_rate
        post_pre_rate = post_pre_rate
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
                self.weights += self.reward_pre_post_rate * (first * span.reshape(1, self.N) * self.weights)
                self.weights += self.reward_post_pre_rate * (span  * last.reshape(1, self.N) * self.weights)
            if not reward:
                self.weights += self.pre_post_rate * (first * span.reshape(1, self.N) * self.weights)
                self.weights += post_pre_rate * (span  * last.reshape(1, self.N) * self.weights)


if __name__ == "__main__":
    stimuli = {
            Stimulus(0.001, lambda t: 1, [0]),
            # Stimulus(0.001, lambda t: 20E-9 * t, [2]),
            # Stimulus(0.001, lambda t: 1E-9 * np.sin(500*t), [3])
            }

    G = NeuronGroup(dt = 0.001, population_size = 100, connection_chance = 0.1, total_time = 0.1, stimuli = stimuli,
                    base_current= 10,
                    u_thresh= 1,
                    u_rest= 0,
                    tau_refractory= 0.005,
                    excitatory_chance=  0.8,
                    Rm= 5,
                    Cm= 0.001)

    G.run()
    G.display_spikes()
    G.show_graph(range(1), with_labels = True)
    # print(G.weights)
    # learning = RFSTDP(G)
    # learning(reward = True)
    # print(G.weights)