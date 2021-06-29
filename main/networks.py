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
SEED = 2045 #manual_seed(SEED) # We set the seed to 2045 because the Singularity is near!

class Network:
    def __init__(self, total_time: float, dt: float, batch_size:int = 1,
                 learning_rule = None, callbacks: list = [],
                 save_history: bool = False):
        self.groups: list = []
        self.connections: list = []
        self.groups_connected_to_stimulus: list = []
        self.architecture()
        self.subNetworks: list = sorted(self.groups + self.connections, key=lambda x: x.order)
        self.dt: float = dt
        self.total_time: float = total_time
        self.batch_size: int = batch_size
        self.N_runs: int = 0
        self.total_timepoints: int = int(total_time/dt)
        ### callbacks
        if isinstance(callbacks, CallbackList):
            self.callbacks = callbacks
        else:
            self.callbacks = CallbackList(callbacks)
        for callback in callbacks:
            callback.setNetwork(self)
        ### set sub-networks batch size
        for subNetwork in self.subNetworks:
            subNetwork.batch_size = self.batch_size
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

    def _set_subNetworks_stimulus(self, stimuli) -> None:
        if type(stimuli) == list:
            assert len(stimuli) == len(self.groups_connected_to_stimulus), f"length stimuli (= {len(stimuli)}) is not compatible with the number of groups connected to stimuli (= {len(self.groups_connected_to_stimulus)})"
            for i, stimulus, group in enumerate(zip(stimuli, self.groups_connected_to_stimulus)):
                assert stimulus.shape == (self.batch_size, group.N, self.total_timepoints), f"Stimulus array number {i} is not of the correct shape. must be: (batch size, number of group neurons, total timepoints) -> {(self.batch_size, group.N, self.total_timepoints)}, but got: {stimulus.shape}"
                group.stimuli_current = stimulus
        else:
            assert len(self.groups_connected_to_stimulus) ==1, f'stimuli must be a list of length of group connected to stimulus (= {len(self.groups_connected_to_stimulus)} for more than 1 connected group'
            group = self.groups_connected_to_stimulus[0]
            assert stimuli.shape == (self.batch_size, group.N, self.total_timepoints), f"Stimulus array is not of the correct shape. must be: (batch size, number of group neurons, total timepoints) -> {(self.batch_size, group.N, self.total_timepoints)}, but got: {stimuli.shape}"
            group.stimuli_current = stimuli

    def _reset(self) -> None:
        self.N_runs +=1
        for subNetwork in self.subNetworks:
            subNetwork._reset(self.dt, self.total_timepoints, self.save_history)

    def _run_one_timepoint(self, subNetwork) -> None:
        self.callbacks.on_subnetwork_start(subNetwork, self.timepoint)
        subNetwork._run_one_timepoint(self.timepoint)
        self.callbacks.on_subnetwork_end(subNetwork, self.timepoint)

    def _learn_one_timepoint(self, subNetwork) -> None:
        self.callbacks.on_subnetwork_learning_start(subNetwork, self.learning_rule, self.timepoint)
        self.learning_rule(subNetwork)
        self.callbacks.on_subnetwork_learning_end(subNetwork, self.learning_rule, self.timepoint)

    def run(self, stimuli = None, progress_bar = False) -> None:
        self._reset()
        self._set_subNetworks_stimulus(stimuli)
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

    def run_multiprocessing(self, stimuli = None, progress_bar = False) -> None:
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


    def randomConnect(self, source, destination, connection_chance, name=None) -> None:
        self.connections.append(RandomConnect(source, destination, connection_chance, name))

    def randomConnections(self, population_size, neuronType, connection_chance,
                          name = None, excitatory_ratio = 0.8, scale_factor = 1,
                          **kwargs) -> RandomConnections:
        neuron_group = RandomConnections(population_size, neuronType, connection_chance, name,
                                             excitatory_ratio, scale_factor, **kwargs)
        self.groups.append(neuron_group)
        return neuron_group

    def connectStimulus(self, group) -> None:
        assert isinstance(group,NeuronGroup), "Stimulus can only be connected to a group"
        self.groups_connected_to_stimulus.append(group)


    @property
    def seconds(self):
        return round(self.timepoint * self.dt, 1)