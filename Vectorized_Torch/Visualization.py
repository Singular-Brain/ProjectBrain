import os
import numpy as np
import torch
### Visualization
import networkx as nx 
import matplotlib.pyplot as plt
import bokeh
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import (Button, CategoricalColorMapper, ColumnDataSource,
                          HoverTool, Label, SingleIntervalTicker, Slider,)
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine
from bokeh.plotting import from_networkx, figure

class NetworkPanel:

    def __init__(self,group, input_neurons, file_path):
        self.group = group
        self.file_path = file_path
        self.input_neurons = input_neurons
        rows, cols = np.where(self.group.AdjacencyMatrix == 1)
        edges = zip(rows.tolist(), cols.tolist())
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(edges)
        self.pos = self.group._set_pos(self.graph, self.input_neurons)
        self._update_hover_tooltips(0)
        self.plot = figure(tooltips = self.HOVER_TOOLTIPS,
            tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
            x_range=Range1d(-.1, sorted(self.pos.values())[-1][0]+0.1),
            y_range=Range1d(0, 1), title='Neuron Group')
        self._update_network(0)
        #Slider
        self.slider = Slider(start=0, end= int(self.group.total_time/ self.group.dt) - 1, value=0, step=1, title="timepoint")
        self.slider.on_change('value', self._slider_update)
        #Button
        self.button = Button(label='► Play', width=60)
        self.button.on_click(self._animate)
        self.callback_id = None
    
    def _update_hover_tooltips(self, timepoint):
        if self.group.save_history:
            neurons_potential = dict([(i[0], i[1]) for i in enumerate(self.group.potential_history[:,timepoint])])
            nx.set_node_attributes(self.graph, name='neurons_potential', values=neurons_potential) 
            neurons_current = dict([(i[0], i[1]) for i in enumerate(self.group.current_history[:,timepoint])])
            nx.set_node_attributes(self.graph, name='neurons_current', values=neurons_current)
            self.HOVER_TOOLTIPS = [("Neuron", "@index"),
                                ("Potential", '@neurons_potential'),
                                ("Current", '@neurons_current') ]
        else:
            self.HOVER_TOOLTIPS = [("Neuron", "@index")]

    def _animate_update(self):
        self.timepoint = self.slider.value + 1
        if self.timepoint > int(self.group.total_time/ self.group.dt) -1 :
            self.timepoint = 0
        self.slider.value = self.timepoint

    def _animate(self):
        if self.button.label == '► Play':
            self.button.label = '❚❚ Pause'
            self.callback_id = curdoc().add_periodic_callback(self._animate_update, 200)
        else:
            self.button.label = '► Play'
            curdoc().remove_periodic_callback(self.callback_id)


    def _slider_update(self, attrname, old, new):
        timepoint = self.slider.value
        self._update_network(timepoint)

    def _update_network(self, timepoint):
        self._update_hover_tooltips(timepoint)
        #Update Spikes
        spiked_neurones = dict([(i[0], i[1]) for i in enumerate((self.group.spike_train[:,timepoint].detach().cpu().numpy()).astype(str))])
        nx.set_node_attributes(self.graph, name='spiked_neurons', values=spiked_neurones) 
        #Add network graph from networkx
        self.network_graph = from_networkx(self.graph, self.pos, scale=10, center=(0, 0))
        #Set node size and color
        self.network_graph.node_renderer.glyph = Circle(size=15, 
        fill_color=bokeh.transform.factor_cmap('spiked_neurons', palette=bokeh.palettes.cividis(2), factors=['False', 'True']))
        #Set edge opacity and width
        self.network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)
        #Set edge highlight colors
        self.network_graph.edge_renderer.hover_glyph = MultiLine(line_color='blue')
        #Highlight nodes and edges
        self.network_graph.inspection_policy = bokeh.models.NodesAndLinkedEdges()
        #Append to plot
        self.plot.renderers = [self.network_graph]

    def display(self):
        self.layout = bokeh.layouts.layout([
            [self.slider, self.button],
            [self.plot],
        ], sizing_mode='scale_width')
        curdoc().add_root(self.layout)
        curdoc().title = "Neuron Group"
        os.system(f'cmd /c "bokeh serve --show {self.file_path}"')
