from abc import ABCMeta, abstractmethod, ABC
import random
import warnings
from itertools import count

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import cv2 
import PIL
print('hi')
class NeuronType(ABC):
    def __init__(self, dt):
        self.dt = dt
        self.mode = None


class IF(NeuronType):
    def __init__(self, dt, Cm = 0.1):
        """
        Simple Integrate-and-Fire Neural Model:
        Assumes a fully-insulated memberane with resistance approaching infinity
        """
        super().__init__(dt)
        self.Cm = Cm #uF
        self.mode = 'if'

    def __call__(self, current, timestep, potential):
        #TODO: Undefined variable 'timepoint'
        pass #return (timestep * current[timepoint])/self.Cm


class LIF(NeuronType):
    def __init__(self, dt, u_rest= -68, Rm = 1, Cm = 0.1):
        """
        Leaky Integrate-and-Fire Neural Model
        """
        super().__init__(dt)
        self.Rm = Rm #ohm
        self.Cm = Cm #uF
        self.tau_m = Rm*Cm
        self.exp_term = np.exp(-1/self.tau_m)
        self.u_base = (1-self.exp_term) * u_rest

    def __call__(self, current, previous_potential):
        return self.u_base + self.exp_term * previous_potential + current* self.dt /self.Cm


class Izhikevich(NeuronType):
    def __init__(self, dt, u_rest=-68, a=0.02, b=0.2, c =-65, d=2,
                 c1=0.04, c2=5, c3=140, c4=1, c5=1, v_rest=30):
        """Izhikevich Neural Model"""
        super().__init__(dt)
        self.c = c  #mv
        self.d = d  #mv
        self.c1 = c1    #mv/ms
        self.c2 = c2    #1/ms
        self.c3 = c3    #mv/ms
        self.c4 = c4    #1/ms
        self.c5 = c5    #mv.ohm/(ms^2.A)
        self.a = a  # dimless
        self.b = b  # dimless
        self.mode = 'izh'
        self.u_rest = u_rest

    def __call__(self, current, timestep, last_spike_timepoint, v_init, recovery):
        v = v_init
        for timepoint in range(last_spike_timepoint, timestep + 1):
            v += self.c1*(v**2)\
                + self.c2*v+self.c3-self.c4*recovery+self.c5*current[timepoint]
            recovery += self.a*(self.b*v-recovery)
        return v, recovery


class Neuron(object):
    _ids = count(0)
    def __init__(self, total_timepoints, dt, model,
                 neurotransmitter = 'excitatory', tau_ref = 0.002, u_rest = -68,
                 u_thresh = +30, save_history = False,):
        """
        Define a new Neuron object
        Args:

        """
        self.id = next(self._ids)
        self.total_timepoints = total_timepoints
        self.model = model
        self.neurotransmitter = neurotransmitter
        ### TODO: add dt to __init__ => tauref = teu_ref/ dt
        self.tau_ref = tau_ref / dt #s => timepoints # refractory period
        self.u_rest = u_rest #mv
        self.u_thresh = u_thresh #mv
        self.u = self.u_rest
        self.save_history = save_history
        self.open = True
        if self.save_history:
            self.current_history = np.zeros(total_timepoints)
            self.potential_history = np.ones(total_timepoints) * self.u_rest
        self.spike_train =  np.zeros(total_timepoints, dtype = np.bool)
        self.connected_to_external_source = False
        self.current = 0
        self.timestep = 0
        if self.model.mode == 'izh':
            self.recovery = self.model.u_rest*self.model.b
        self.refactory_time = 0

    def step(self):
        assert self.timestep < self.total_timepoints, "Simulation interval has finished!"
        if self.timestep == 0:
            self._reset()
            return
        # Check refactory interval
        if not self.open:
            if self.refactory_time < self.tau_ref - 1:
                self.refactory_time += 1
            else:
                self.open = True
        else:
            # Update
            self.u = self.model(self.current, self.u)
            #TODO: Izhikevich Model
            # if self.model.mode == 'izh':
            #     self.u, self.recovery = self.model(self.current_history, self.timestep, self.spike_timepoints[-1], self.u, self.recovery)
            #TODO: add save history (include both)
            # Save potential history
            if self.save_history:
                self.current_history[self.timestep] = self.current
                self.potential_history[self.timestep] = self.u
            # Spike
            if self.u >= self.u_thresh:
                self.spike_train[self.timestep] = True
                self.open = False
                self.refactory_time = 0 
                if self.model.mode == 'izh':
                    self.recovery += self.model.d
                self.u = self.u_rest                
        # Empty neuron's current
        self.current = 0

    @property
    def spike(self):
        return self.spike_train[self.timestep]

    @property
    def spike_timepoints(self):
        return np.where(self.spike_train)

    def _reset(self):
        self.spike_train =  np.zeros(self.total_timepoints, dtype = np.bool)
        self.current = 0
        self.open = True

    def display_spikes(self):
        spike_train = self.spike_train.astype(str)
        spike_train[spike_train=='True'] = '|'
        spike_train[spike_train=='False'] =  " "
        return ''.join(np.array2string(spike_train).split("'")[1:-1:2])


class NeuronGroup(object):
    _ids = count(0)
    order = 0
    def __init__(self, population_size, total_timepoints, dt, neuron_model = LIF,
                 connection_chance=1/10, inhibition_rate= 2/10,
                 base_current = 50, online_learning_rule = None,
                 neuron_attrs = {}, save_gif = False,
                 ):
        """
        Parameters
        ----------
        online_learning_rule: a Learning class

        """
        self.id = next(self._ids)
        self.dt = dt
        self.total_timepoints = total_timepoints
        self.population_size = population_size
        self.connection_chance = connection_chance
        self.base_current = base_current
        self.online_learning_rule = online_learning_rule(dt) if online_learning_rule is not None else None
        self.save_gif = save_gif
        if save_gif:
            warnings.warn('WARNING: "save_gif" is set to True, it can considerably slow down the simulation process. To see the result use "save_gif_to" function after training')
            self.images = []
        self.neurons = {
            Neuron(total_timepoints, dt, model = neuron_model(dt), 
                   neurotransmitter = 'excitatory' if random.random() > inhibition_rate else 'inhibitory',
                   **neuron_attrs)
            for _ in range(self.population_size)} 
        # Graph
        self._define_network_graph()
        self.pos = None
        self.timestep = 0

    def _generate_new_graph(self):
        self.network = nx.DiGraph()
        self.network.add_nodes_from(self.neurons)
        for PreSN in self.network.nodes:
            for PostSN in set(self.network.nodes) - {PreSN}:
                    if random.random() < self.connection_chance:
                        #TODO: add_edge_from a N*N matrix 
                        self.network.add_edge(PreSN, PostSN, weight = np.random.randn(1))

    def _define_network_graph(self):
        self._generate_new_graph()
        find_single_graph = len(list(nx.weakly_connected_components(self.network))) == 1
        if not find_single_graph:
            for _ in range(20):
                self._generate_new_graph()
                find_single_graph = len(list(nx.weakly_connected_components(self.network))) == 1
                if find_single_graph:
                    break
        assert find_single_graph, f"Couldn't make a single graph. Please increase the connection chance"

    def step(self):
        self.active_neurons = set()
        for neuron in self.neurons:
            neuron.timestep = self.timestep
            neuron.step()
            if neuron.spike:
                self.active_neurons.add(neuron)
 
        for neuron in self.active_neurons:
            for preSN, postSN, weight in self.network.out_edges(neuron, data = 'weight'):
                if postSN.open:
                    postSN.current += (-1 if preSN.neurotransmitter == 'inhibitory' else +1) * self.base_current * weight
            # if self.online_learning_rule:
            #     self.online_learning_rule(self.network, neuron)
 
        if self.save_gif:
            fig, _ = self.draw_graph()
            self.images.append(self._fig_to_PIL_image(fig))
    

    def get_input_output(self, N_input_neurons, N_output_neurons):
        input_neurons = random.sample(self.network.nodes(), N_input_neurons)
        input_neurons = sorted(input_neurons, key = lambda x: x.id)
        processed_neurons = input_neurons.copy()
        for neuron in processed_neurons:
            successors = self.network.successors(neuron)
            for successor in successors:
                if successor not in set(processed_neurons):
                    processed_neurons.append(successor)
        output_neurons = processed_neurons[-N_output_neurons:]
        output_neurons = sorted(output_neurons, key = lambda x: x.id)
        return input_neurons, output_neurons

    ### Visualization
    def set_pos(self):
        input_neurons = set()
        for neuron in self.neurons:
            if neuron.connected_to_external_source:
                input_neurons.add(neuron)
        self.pos = {}
        for y, neuron in enumerate(input_neurons):
            self.pos[neuron] = (0, (y+1) / (len(input_neurons)+1))
        last_layer = input_neurons
        for x in range(1, len(self.network.nodes())):
            counted_neurons = set(self.pos.keys()) 
            next_layer = set()
            for neuron in last_layer:
                next_layer.update(self.network.successors(neuron))
            new_neurons = next_layer - counted_neurons
            if new_neurons == set():
                break
            else:
                for y, neuron in enumerate(new_neurons):
                    self.pos[neuron] = (x, (y+1)/(len(new_neurons)+1))
            last_layer = next_layer
        not_in_input_successors =set(self.network.nodes()) - set(self.pos.keys()) 
        if not_in_input_successors:
            for y, neuron in enumerate(not_in_input_successors):
                    self.pos[neuron] = (x, (y+1)/(len(not_in_input_successors)+1))

    def draw_graph(self, display_ids = False):
        # Set graph position 
        if not self.pos:
            self.set_pos()
 
        fig, ax = plt.subplots(figsize=(10,8))
        nx.draw_networkx_edges(self.network, pos=self.pos, ax=ax, edge_color="gray")
        inactive_neurons = nx.draw_networkx_nodes(self.network, pos=self.pos, ax=ax,
            nodelist=set(self.network.nodes()) - (self.active_neurons),
            node_color="gray",)
        inactive_neurons.set_edgecolor("black")
        active_neurons = nx.draw_networkx_nodes(self.network, pos=self.pos, ax = ax,
                                             nodelist=self.active_neurons,
                                             node_color='blue',)
        active_neurons.set_edgecolor("black")
        if display_ids:
            nx.draw_networkx_labels(self.network, pos = self.pos, ax = ax,
                labels = {neuron: neuron.id for neuron in self.network.nodes})
        return fig, ax
            
    def _fig_to_PIL_image(self, fig):
        fig.savefig("test.jpg")
        img = cv2.imread("test.jpg")
        return PIL.Image.fromarray(img)
 
    def save_gif_to(self, path, **kwargs):
        assert self.save_gif, "No gif has been saved during the simulation. To use this function you must set 'save_gif' to True"
        frame0 = self.images.pop(0)
        frame0.save(path, save_all=True, append_images=self.images, **kwargs)
 
    def display_spikes(self):
        spike_train = ' id\n' + '=' * 5 + '╔' + '═' * self.total_timepoints + '╗\n'
        for neuron in sorted(self.neurons, key = lambda x: x.id):
            if neuron.connected_to_external_source:
                spike_train += str(neuron.id) + '*' + ' ' * (4 - len(str(neuron.id))) \
                + '║' + neuron.display_spikes() + '║\n' 
            else:
                spike_train += str(neuron.id) + ' ' * (5 - len(str(neuron.id))) \
                + '║' + neuron.display_spikes() + '║\n'  
        spike_train +=' ' * 5 + '╚' + '═' * self.total_timepoints + '╝'
        print(spike_train)


class Stimulus(object):
    _ids = count(0)
    order = 1
    def __init__(self, output, dt):
        """
        Parameters
        ----------
        output: function
        A function that determines the output of the Stimulus
        in a specific timestep.
        examples:
        output = lambda t: np.sin(2 * t)
        output = lambda t: 2

        Returns
        -------
        Returns the value of output if no connection is set.
        otherwise, sends the value to the connection (returns None).
        """
        self.output = output
        self.connection = None
        self.id = next(self._ids)
        self.timestep = 0
        self.dt = dt

    def connect(self, connection):
        connection.connected_to_external_source = True
        self.connection = connection

    def step(self):
        if self.connection is None:
            warnings.warn(f"WARNING: Stimulus (id: {self.id}) has not connected to any object!")
            return self.output(self.timestep * self.dt)
        self.connection.current += self.output(self.timestep * self.dt)

    @property
    def value(self):
        return self.output(self.timestep * self.dt)
