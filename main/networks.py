from abc import ABC, abstractmethod
import concurrent.futures
import torch
import numpy as np
from tqdm import tqdm
from .subnetworks import *
from .callbacks import CallbackList
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
        self.subNetworks = sorted(self.groups + self.connections, key=lambda x: x.order)
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
            callback.setNetwork(self)
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
            subNetwork._reset(self.dt, self.total_timepoints, self.save_history)

    def _run_one_timepoint(self, subNetwork):
        subNetwork.stim_current = self._get_stimuli_current(subNetwork)
        self.callbacks.on_subnetwork_start(subNetwork, self.timepoint)
        subNetwork._run_one_timepoint(self.timepoint)
        self.callbacks.on_subnetwork_end(subNetwork, self.timepoint)

    def _learn_one_timepoint(self, subNetwork):
        self.callbacks.on_subnetwork_learning_start(subNetwork, self.learning_rule, self.timepoint)
        self.learning_rule(subNetwork)
        self.callbacks.on_subnetwork_learning_end(subNetwork, self.learning_rule, self.timepoint)

    def run(self, stimuli = None, progress_bar = False):
        self._reset()
        self._make_stimuli_adjacency(stimuli)
        self.callbacks.on_run_start(self.N_runs)
        for self.timepoint in tqdm(range(self.total_timepoints)) if progress_bar \
        else range(self.total_timepoints):
            self.callbacks.on_timepoint_start(self.timepoint)
            for subNetwork in self.subNetworks:
                self._run_one_timepoint(subNetwork)
            self.callbacks.on_timepoint_end(self.timepoint)
            if self.learning_rule:
                self.learning_rule.update_neuromodulators()
                self.callbacks.on_learning_start(self.learning_rule, self.timepoint)
                for subNetwork in self.subNetworks:
                    self._learn_one_timepoint(subNetwork)
                self.callbacks.on_learning_end(self.learning_rule, self.timepoint)
        self.callbacks.on_run_end(self.N_runs)

    def run_multiprocessing(self, stimuli = None, progress_bar = False):
        self._reset()
        self._make_stimuli_adjacency(stimuli)
        self.callbacks.on_run_start(self.N_runs)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for self.timepoint in tqdm(range(self.total_timepoints)) if progress_bar \
            else range(self.total_timepoints):
                self.callbacks.on_timepoint_start(self.timepoint)
                futures = [executor.submit(self._run_one_timepoint, args) for args in self.groups]
                concurrent.futures.wait(futures, return_when = 'ALL_COMPLETED')
                futures = [executor.submit(self._run_one_timepoint, args) for args in self.connections]
                concurrent.futures.wait(futures, return_when = 'ALL_COMPLETED')
                self.callbacks.on_timepoint_end(self.timepoint)
                if self.learning_rule:
                    self.learning_rule.update_neuromodulators()
                    self.callbacks.on_learning_start(self.learning_rule, self.timepoint)
                    futures = [executor.submit(self._learn_one_timepoint, args) for args in self.subNetworks]
                    concurrent.futures.wait(futures, return_when = 'ALL_COMPLETED')
                    self.callbacks.on_learning_end(self.learning_rule, self.timepoint)
            self.callbacks.on_run_end(self.N_runs) 

    def randomConnect(self, source, destination, connection_chance):
        self.connections.append(RandomConnect(source, destination, connection_chance))

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


    @property
    def seconds(self):
        return round(self.timepoint * self.dt, 1)