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
    def __init__(self, log_dir = None):
        super().__init__()
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir =log_dir)

    def on_run_start(self, N_runs,):
        if N_runs == 1:
            for group in self.model.network._groups:
                self.writer.add_histogram('Layers/' + group.name,
                                          self.model.weight_values([group.idx, group.idx]),
                                          0)
            for connection in self.model.network._connections:
                self.writer.add_histogram('Connections/' + connection.from_.name + ' to ' + connection.to.name,
                                        self.model.weight_values([connection.from_.idx, connection.to.idx]),
                                        0)

    def on_run_end(self, N_runs,):
        for group in self.model.network._groups:
            self.writer.add_histogram('Layers/' + group.name,
                                    self.model.weight_values([group.idx, group.idx]),
                                      N_runs)
        for connection in self.model.network._connections:
            self.writer.add_histogram('Connections/' + connection.from_.name + ' to ' + connection.to.name,
                                      self.model.weight_values([connection.from_.idx, connection.to.idx]),
                                      N_runs)
    

