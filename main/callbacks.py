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


class Callback:
    def __init__(self,):
        ...

    def set_NeuronGroup(self, NeuronGroup):
        self.model = NeuronGroup

    
    def on_run_start(self, N_runs,):
        ...

    def on_timepoint_start(self, timepoint,):
        ...

    def on_run_end(self, N_runs,):
        ...

    def on_timepoint_end(self, timepoint,):
        ...
    

class TensorBoard(Callback):
    def __init__(self,):
        super().__init__()
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter()

    def on_run_start(self, N_runs,):
        if N_runs == 1:
            self.writer.add_histogram('distribution of weights', self.model.weight_values, 0)

    def on_run_end(self, N_runs,):
        self.writer.add_histogram('distribution of weights', self.model.weight_values, 1)
    

        