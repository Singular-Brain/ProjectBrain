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


class Callback():
    def set_NeuronGroup(self, NeuronGroup):
        self.model = NeuronGroup

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
            update_secs ([None or float], optional): If set to None, data will be updated at the end of each run. If set to a float(N), data will be updated every N seconds (simulation time). Defaults to None.
            log_dir ([str or None], optional): Save directory location. Default is runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each run. Use hierarchical folder structure to compare between runs easily. e.g. pass in 'runs/exp1', 'runs/exp2', etc. for each new experiment to compare across them.
        """
        super().__init__()
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir =log_dir,)
        self.update_secs = update_secs



    def on_run_start(self, N_runs,):# Initial state
        if self.update_secs:
            ### Groups weights histogram
            for group in self.model.network._groups:
                if (self.model.weight_values([group.idx, group.idx])).any():
                    self.writer.add_histogram('Groups/' + group.name + f' weights(Run:{self.model.N_runs})', 
                                            self.model.weight_values([group.idx, group.idx]), 0)
            ### Connections weights histogram
            for connection in self.model.network._connections:
                self.writer.add_histogram('Connections/' + connection.from_.name + ' to ' +
                                        connection.to.name + f' weights(Run:{self.model.N_runs})',
                                        self.model.weight_values([connection.from_.idx, connection.to.idx]),
                                        0)
        else:
            if N_runs == 1: 
                ### Groups weights histogram
                for group in self.model.network._groups:
                    if (self.model.weight_values([group.idx, group.idx])).any():
                        self.writer.add_histogram('Groups/' + group.name + ' weights', 
                        self.model.weight_values([group.idx, group.idx]), 0)
                ### Connections weights histogram
                for connection in self.model.network._connections:
                    self.writer.add_histogram('Connections/' + connection.from_.name + ' to ' + connection.to.name,
                                            self.model.weight_values([connection.from_.idx, connection.to.idx]),
                                            0)


    def on_run_end(self, N_runs,):
        if not self.update_secs:
            ### Groups weights histogram
            for group in self.model.network._groups:
                if (self.model.weight_values([group.idx, group.idx])).any():
                    self.writer.add_histogram('Groups/' + group.name + ' weights',
                    self.model.weight_values([group.idx, group.idx]), N_runs)
            ### Connections weights histogram
            for connection in self.model.network._connections:
                self.writer.add_histogram('Connections/' + connection.from_.name + ' to ' + connection.to.name,
                                        self.model.weight_values([connection.from_.idx, connection.to.idx]),
                                        N_runs)
            ### Groups spikes
            for group in self.model.network._groups:
                self.writer.add_scalar('Spikes/' + group.name,
                        self.model.spike_train[group.idx,:].sum(),
                        N_runs)
            ### EPSP/IPSP
            if self.model.save_history:
                for connection in self.model.network._connections:
                    connections = self.model.adjacency_matrix[connection.from_.idx, connection.to.idx].sum(axis = 1)
                    potential = (self.model.potential_history[connection.from_.idx,:] - self.model.u_rest).sum(axis = 1)
                    EPSP = sum(connections * potential * self.model.excitatory_neurons[connection.from_.idx])
                    IPSP = sum(connections * potential * self.model.inhibitory_neurons[connection.from_.idx])
                    self.writer.add_scalar('EPSP/'+ connection.from_.name + ' to ' + connection.to.name,
                                            EPSP, N_runs) 
                    self.writer.add_scalar('IPSP/'+ connection.from_.name + ' to ' + connection.to.name,
                                            IPSP, N_runs) 
        self.writer.flush()

    def on_timepoint_end(self, timepoint): 
        if self.update_secs and (
            (timepoint>0 and timepoint%(int(self.update_secs/self.model.dt)) ==0)
            or timepoint==self.model.total_timepoints-1
            ):
            step=int(self.update_secs/self.model.dt)
            ### Groups weights histogram
            for group in self.model.network._groups:
                if (self.model.weight_values([group.idx, group.idx])).any():
                    self.writer.add_histogram('Groups/' + group.name + f' weights(Run:{self.model.N_runs})', 
                                            self.model.weight_values([group.idx, group.idx]), self.model.seconds)
            ### Connections weights histogram
            for connection in self.model.network._connections:
                self.writer.add_histogram('Connections/' + connection.from_.name + ' to ' +
                                        connection.to.name + f' weights(Run:{self.model.N_runs})',
                                        self.model.weight_values([connection.from_.idx, connection.to.idx]),
                                        self.model.seconds)
            ### Groups spikes
            for group in self.model.network._groups:
                self.writer.add_scalar('Spikes/' + group.name + f'(Run:{self.model.N_runs})',
                        self.model.spike_train[group.idx,timepoint-step:timepoint].sum(),
                        self.model.seconds)
            ### EPSP/IPSP
            if self.model.save_history:
                for connection in self.model.network._connections:
                    connections = self.model.adjacency_matrix[connection.from_.idx, connection.to.idx].sum(axis = 1)
                    potential = (self.model.potential_history[connection.from_.idx,timepoint-step:timepoint] - self.model.u_rest).sum(axis = 1)
                    EPSP = sum(connections * potential * self.model.excitatory_neurons[connection.from_.idx])
                    IPSP = sum(connections * potential * self.model.inhibitory_neurons[connection.from_.idx])
                    self.writer.add_scalar('EPSP/'+ connection.from_.name + ' to ' + connection.to.name+ f'(Run:{self.model.N_runs})',
                                            EPSP, self.model.seconds) 
                    self.writer.add_scalar('IPSP/'+ connection.from_.name + ' to ' + connection.to.name+ f'(Run:{self.model.N_runs})',
                                            IPSP, self.model.seconds) 