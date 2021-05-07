import numpy as np
from Visualization import NetworkPanelData

DataFile = 'RunData.npy'

panel = NetworkPanelData(DataFile, input_neurons = [0], file_path = __file__, verbose = True)
panel.display()