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

    def on_subnetwork_start(self, subNetwork, timepoint):
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_subnetwork_start(subNetwork, timepoint) 

    def on_subnetwork_end(self, subNetwork, timepoint):
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_subnetwork_end(subNetwork, timepoint) 

    def on_learning_start(self, learning_rule, timepoint):
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_learning_start(learning_rule, timepoint)   

    def on_learning_end(self, learning_rule, timepoint):
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_learning_start(learning_rule, timepoint)  

    def on_subnetwork_learning_start(self, subNetwork, learning_rule, timepoint):
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_subnetwork_learning_start(subNetwork, learning_rule, timepoint) 

    def on_subnetwork_learning_end(self, subNetwork, learning_rule, timepoint):
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_subnetwork_learning_end(subNetwork, learning_rule, timepoint) 


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

    @abstractmethod
    def on_subnetwork_start(self, subNetwork, timepoint,):
        ... 

    @abstractmethod
    def on_subnetwork_end(self, subNetwork, timepoint,):
        ... 

    @abstractmethod
    def on_learning_start(self, learning_rule, timepoint):
        ...

    @abstractmethod
    def on_learning_end(self, learning_rule, timepoint):
        ...
 
    @abstractmethod
    def on_subnetwork_learning_start(self, subNetwork, learning_rule, timepoint):
        ...

    @abstractmethod
    def on_subnetwork_learning_end(self, subNetwork, learning_rule, timepoint):
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
        assert type(update_secs) == int or update_secs is None, "'update_secs' must be an integer"
        self.writer = SummaryWriter(log_dir =log_dir,)
        self.update_secs = update_secs
        #TODO: use := notation to make it more pythonic after google Colab finally decided tp update its python version to something above 3.8.0 :\


    def on_run_start(self, N_runs,):
        # Save hyper-parameters
        self.writer.add_text('Network', str(self.network.__dict__))
        if self.network.learning_rule:
            self.writer.add_text(type(self.network.learning_rule).__name__, str(self.network.learning_rule.__dict__))
        for group in self.network.groups:
            self.writer.add_text('subNetworks/'+ group.name,str(group.__dict__) + str(group.neuronType.__dict__))
        for connection in self.network.connections:
            self.writer.add_text('subNetworks/'+connection.name, str(connection.__dict__))
        # Initial state
        if self.update_secs:
            self.step =int(self.update_secs/self.network.dt)
            ### Weights histogram
            for subNetwork in self.network.subNetworks:
                excitetory_weight = subNetwork.weights[subNetwork.weights>0]
                inhibitory_weight = subNetwork.weights[subNetwork.weights<0]
                if excitetory_weight.any():
                    self.writer.add_histogram(f'Weights/{subNetwork.type}/{subNetwork.name}/Excitatory(Run{N_runs})', 
                                            excitetory_weight, 0)
                if inhibitory_weight.any():
                    self.writer.add_histogram(f'Weights/{subNetwork.type}/{subNetwork.name}/Inhibitory(Run{N_runs})', 
                                              inhibitory_weight, 0)

        else:
            if N_runs == 1: 
                ### Weights histogram
                for subNetwork in self.network.subNetworks:
                    excitetory_weight = subNetwork.weights[subNetwork.weights>0]
                    inhibitory_weight = subNetwork.weights[subNetwork.weights<0]
                    if excitetory_weight.any():
                        self.writer.add_histogram(f'Weights/{subNetwork.type}/{subNetwork.name}/Excitatory', 
                                                  excitetory_weight, 0)
                    if inhibitory_weight.any():
                        self.writer.add_histogram(f'Weights/{subNetwork.type}/{subNetwork.name}/Inhibitory', 
                                                  inhibitory_weight, 0)


    def on_run_end(self, N_runs,):
        if self.update_secs:
            ### Weights histogram
            for subNetwork in self.network.subNetworks:
                excitetory_weight = subNetwork.weights[subNetwork.weights>0]
                inhibitory_weight = subNetwork.weights[subNetwork.weights<0]
                if excitetory_weight.any():
                    self.writer.add_histogram(f'Weights/{subNetwork.type}/{subNetwork.name}/Excitatory(Run{N_runs})', 
                                            excitetory_weight, self.network.seconds)
                if inhibitory_weight.any():
                    self.writer.add_histogram(f'Weights/{subNetwork.type}/{subNetwork.name}/Inhibitory(Run{N_runs})', 
                                              inhibitory_weight, self.network.seconds)
        else:
            combined = {}
            EPSP = {}
            IPSP = {}
            ### Weights histogram
            for subNetwork in self.network.subNetworks:
                excitetory_weight = subNetwork.weights[subNetwork.weights>0]
                inhibitory_weight = subNetwork.weights[subNetwork.weights<0]
                if excitetory_weight.any():
                    self.writer.add_histogram(f'Weights/{subNetwork.type}/{subNetwork.name}/Excitatory', 
                                                excitetory_weight, N_runs)
                if inhibitory_weight.any():
                    self.writer.add_histogram(f'Weights/{subNetwork.type}/{subNetwork.name}/Inhibitory', 
                                                inhibitory_weight, N_runs)
                ### Groups spikes
                if subNetwork.type == "Neuron Group":
                    spikes = subNetwork.spike_train.sum()
                    self.writer.add_scalar('Spikes/' + subNetwork.name,spikes,N_runs)
                    combined[subNetwork.name] = spikes
                ### EPSP/IPSP
                elif subNetwork.type == "Connection" and self.network.save_history:
                    EPSP[subNetwork.name] = subNetwork.EPSP()
                    IPSP[subNetwork.name] = subNetwork.IPSP()
            ### Combined
            self.writer.add_scalars('Spikes/Combined',combined,N_runs)
            ### Pre-Synaptic Potential
            self.writer.add_scalars(f'Pre-Synaptic Potential/EPSP(Run{self.network.N_runs})',EPSP,N_runs)
            self.writer.add_scalars(f'Pre-Synaptic Potential/IPSP(Run{self.network.N_runs})',IPSP,N_runs)

    def on_timepoint_start(self, timepoint):
        if self.update_secs and timepoint>0 and timepoint%self.step ==0 and self.network.save_history:
            self.combined_spikes = {}      
            self.EPSP = {}
            self.IPSP = {}        


    def on_subnetwork_end(self, subNetwork, timepoint): 
        if self.update_secs and timepoint>0 and timepoint%self.step ==0:
            ### Weights histogram
            excitetory_weight = subNetwork.weights[subNetwork.weights>0]
            inhibitory_weight = subNetwork.weights[subNetwork.weights<0]
            if excitetory_weight.any():
                self.writer.add_histogram(f'Weights/{subNetwork.type}/{subNetwork.name}/Excitatory(Run{self.network.N_runs})', 
                                        excitetory_weight, self.network.seconds)
            if inhibitory_weight.any():
                self.writer.add_histogram(f'Weights/{subNetwork.type}/{subNetwork.name}/Inhibitory(Run{self.network.N_runs})', 
                                            inhibitory_weight, self.network.seconds)
            ### Groups spikes
            if subNetwork.type == "Neuron Group":
                spikes = subNetwork.spike_train[:,timepoint-self.step:timepoint].sum()
                self.writer.add_scalar(f'Spikes/{subNetwork.name}(Run{self.network.N_runs})',spikes,
                                    self.network.seconds)
                self.combined_spikes[subNetwork.name] = spikes
            ### EPSP/IPSP
            elif subNetwork.type == "Connection" and self.network.save_history:
                self.EPSP[subNetwork.name] = subNetwork.EPSP(slice(timepoint-self.step,timepoint))
                self.IPSP[subNetwork.name] = subNetwork.IPSP(slice(timepoint-self.step,timepoint))


    def on_timepoint_end(self, timepoint):
        if self.update_secs and timepoint>0 and timepoint%self.step ==0 and self.network.save_history:
            ### Combined Spikes
            self.writer.add_scalars(f'Spikes/Combined(Run{self.network.N_runs})',self.combined_spikes,self.network.seconds)
            self.writer.add_scalars(f'Pre-Synaptic Potential/EPSP(Run{self.network.N_runs})',self.EPSP,self.network.seconds)
            self.writer.add_scalars(f'Pre-Synaptic Potential/IPSP(Run{self.network.N_runs})',self.IPSP,self.network.seconds)


    def on_learning_start(self, learning_rule, timepoint):
        if self.update_secs and timepoint%self.step ==0:
            self.writer.add_scalars(f'Neuromodulators/DA-GABA(Run{self.network.N_runs})',
            {'DA':learning_rule.dopamine + learning_rule.dopamine_base,
             'GABA':learning_rule.gaba + learning_rule.gaba_base },
            self.network.seconds)
            self.writer.add_scalar(f'Neuromodulators/Global Neuromodulator(M)(Run{self.network.N_runs})',
            learning_rule.M,
            self.network.seconds)