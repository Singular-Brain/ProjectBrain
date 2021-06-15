from abc import abstractmethod

class CallbackList:
    def __init__(self, callbacks):
        self.callbacks = callbacks
        for callback in callbacks:
            assert isinstance(callback,Callback), \
                f"All elements must be an instance of 'Callback' class.Found {callback} of type {type(callback)}"

    def __iter__(self):
        for elem in self.callbacks:
            yield elem

    def on_run_start(self, N_runs):
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_run_start(N_runs)

    def on_timepoint_start(self, timepoint,):
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_timepoint_start(timepoint)
            
    def on_run_end(self, N_runs):
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_run_end(N_runs)

    def on_timepoint_end(self, timepoint):
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_timepoint_end(timepoint)    

    def on_subNetwork_start(self, subNetwork, timepoint):
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_subNetwork_start(subNetwork, timepoint) 

    def on_subNetwork_end(self, subNetwork, timepoint):
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_subNetwork_end(subNetwork, timepoint) 

class Callback():
    def setNetwork(self, network):
        self.network = network

    @abstractmethod
    def on_run_start(self, N_runs,):
        ...

    @abstractmethod
    def on_timepoint_start(self, timepoint,):
        ...

    @abstractmethod
    def on_run_end(self, N_runs,):
        ...

    @abstractmethod
    def on_timepoint_end(self, timepoint,):
        ...
    

class TensorBoard(Callback):
    def __init__(self, update_secs = None, log_dir = None,):
        """Add simulation data to the TensorBoard

        Args:
            update_secs ([None or int], optional): If set to None, data will be updated at the end of each run. If set to a float(N), data will be updated every N seconds (simulation time). Defaults to None.
            log_dir ([str or None], optional): Save directory location. Default is runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each run. Use hierarchical folder structure to compare between runs easily. e.g. pass in 'runs/exp1', 'runs/exp2', etc. for each new experiment to compare across them.
        """
        super().__init__()
        from torch.utils.tensorboard import SummaryWriter
        assert type(update_secs) == int, "" #TODO
        self.writer = SummaryWriter(log_dir =log_dir,)
        self.update_secs = update_secs



    def on_run_start(self, N_runs,):# Initial state
        if self.update_secs:
            ### Groups weights histogram
            for group in self.network.network._groups:
                if (weight_values:=self.network.weight_values([group.idx, group.idx])).any():
                    self.writer.add_histogram('Groups/' + group.name + f' weights(Run:{self.network.N_runs})', 
                                            weight_values, 0)
            ### Connections weights histogram
            for connection in self.network.network._connections:
                self.writer.add_histogram('Connections/' + connection.from_.name + ' to ' +
                                        connection.to.name + f' weights(Run:{self.network.N_runs})',
                                        self.network.weight_values([connection.from_.idx, connection.to.idx]),
                                        0)
        else:
            if N_runs == 1: 
                ### Groups weights histogram
                for group in self.network.network._groups:
                    if (weight_values:=self.network.weight_values([group.idx, group.idx])).any():
                        self.writer.add_histogram('Groups/' + group.name + ' weights', weight_values, 0)
                ### Connections weights histogram
                for connection in self.network.network._connections:
                    self.writer.add_histogram('Connections/' + connection.from_.name + ' to ' + connection.to.name,
                                            self.network.weight_values([connection.from_.idx, connection.to.idx]),
                                            0)


    def on_run_end(self, N_runs,):
        if not self.update_secs:
            ### Groups weights histogram
            for group in self.network.network._groups:
                if (weight_values:=self.network.weight_values([group.idx, group.idx])).any():
                    self.writer.add_histogram('Groups/' + group.name + ' weights', weight_values, N_runs)
            ### Connections weights histogram
            for connection in self.network.network._connections:
                self.writer.add_histogram('Connections/' + connection.from_.name + ' to ' + connection.to.name,
                                        self.network.weight_values([connection.from_.idx, connection.to.idx]),
                                        N_runs)
            ### Groups spikes
            combined = {}
            for group in self.network.network._groups:
                spikes = self.network.spike_train[group.idx,:].sum()
                self.writer.add_scalar('Spikes/' + group.name,spikes,N_runs)
                combined[group.name] = spikes
            ### Combined
            self.writer.add_scalars('Spikes/Combined',spikes,N_runs)
            ### EPSP/IPSP
            if self.network.save_history:
                for connection in self.network.network._connections:
                    connections = self.network.adjacency_matrix[connection.from_.idx, connection.to.idx].sum(axis = 1)
                    potential = (self.network.potential_history[connection.from_.idx,:] - self.network.u_rest).sum(axis = 1)
                    EPSP = sum(connections * potential * self.network.excitatory_neurons[connection.from_.idx])
                    IPSP = sum(connections * potential * self.network.inhibitory_neurons[connection.from_.idx])
                    self.writer.add_scalar('EPSP/'+ connection.from_.name + ' to ' + connection.to.name,
                                            EPSP, N_runs) 
                    self.writer.add_scalar('IPSP/'+ connection.from_.name + ' to ' + connection.to.name,
                                            IPSP, N_runs) 

    def on_timepoint_end(self, timepoint): 
        if self.update_secs and (
            (timepoint>0 and timepoint%(step:=int(self.update_secs/self.network.dt)) ==0)
            or timepoint==self.network.total_timepoints-1
            ):
            ### Groups weights histogram
            for group in self.network.network._groups:
                if (weight_values:=self.network.weight_values([group.idx, group.idx])).any():
                    self.writer.add_histogram('Groups/' + group.name + f' weights(Run:{self.network.N_runs})', 
                                            weight_values, self.network.seconds)
            ### Connections weights histogram
            for connection in self.network.network._connections:
                self.writer.add_histogram('Connections/' + connection.from_.name + ' to ' +
                                        connection.to.name + f' weights(Run:{self.network.N_runs})',
                                        self.network.weight_values([connection.from_.idx, connection.to.idx]),
                                        self.network.seconds)
            ### Groups spikes
            combined = {}
            for group in self.network.network._groups:
                spikes = self.network.spike_train[group.idx,timepoint-step:timepoint].sum()
                self.writer.add_scalar('Spikes/' + group.name + f'(Run:{self.network.N_runs})',
                        spikes,
                        self.network.seconds)
                combined[group.name] = spikes
            ### Combined
            self.writer.add_scalars('Spikes/Combined',combined,self.network.seconds)                
            ### EPSP/IPSP
            if self.network.save_history:
                for connection in self.network.network._connections:
                    connections = self.network.adjacency_matrix[connection.from_.idx, connection.to.idx].sum(axis = 1)
                    potential = (self.network.potential_history[connection.from_.idx,timepoint-step:timepoint] - self.network.u_rest).sum(axis = 1)
                    EPSP = sum(connections * potential * self.network.excitatory_neurons[connection.from_.idx])
                    IPSP = sum(connections * potential * self.network.inhibitory_neurons[connection.from_.idx])
                    self.writer.add_scalar('EPSP/'+ connection.from_.name + ' to ' + connection.to.name+ f'(Run:{self.network.N_runs})',
                                            EPSP, self.network.seconds) 
                    self.writer.add_scalar('IPSP/'+ connection.from_.name + ' to ' + connection.to.name+ f'(Run:{self.network.N_runs})',
                                            IPSP, self.network.seconds) 