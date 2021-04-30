import os
import numpy as np
### Visualization
import networkx as nx 
import matplotlib.pyplot as plt
import bokeh
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import (Button, CategoricalColorMapper, ColumnDataSource,
                          HoverTool, Label, SingleIntervalTicker, Slider,
                          Arrow, NormalHead,)
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
        self.pos = self._set_pos(self.graph, self.input_neurons)
        self._update_hover_tooltips(0)
        self.plot = figure(
            plot_width=1000, plot_height=520,
            tooltips = self.HOVER_TOOLTIPS,
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
            self.HOVER_TOOLTIPS = [ ("Neuron", "@index"),
                                    ("Potential", '@neurons_potential'),
                                    ("Current", '@neurons_current'),
                                    ]
        else:
            self.HOVER_TOOLTIPS = [("Neuron", "@index")]


    def _set_pos(self, graph, input_neurons):
        pos = {}
        for y, neuron in enumerate(input_neurons):
            pos[neuron] = (0, (y+1) / (len(input_neurons)+1))
        last_layer = input_neurons
        for x in range(1, len(graph.nodes())):
            counted_neurons = set(pos.keys()) 
            next_layer = set()
            for neuron in last_layer:
                next_layer.update(graph.successors(neuron))
            new_neurons = next_layer - counted_neurons
            if new_neurons == set():
                break
            else:
                for y, neuron in enumerate(new_neurons):
                    pos[neuron] = (x, (y+1)/(len(new_neurons)+1))
            last_layer = next_layer
        not_in_input_successors =set(graph.nodes()) - set(pos.keys()) 
        if not_in_input_successors:
            for y, neuron in enumerate(not_in_input_successors):
                    pos[neuron] = (x, (y+1)/(len(not_in_input_successors)+1))
        return pos

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
        spiked_neurones = dict([(i[0], i[1]) for i in enumerate((self.group.spike_train[:,timepoint]).astype(str))])
        nx.set_node_attributes(self.graph, name='spiked_neurons', values=spiked_neurones) 
        #
        elements = np.nditer(self.group.weights, flags=['multi_index'])
        excitetory_inhibitory = {(elements.multi_index[0], elements.multi_index[1]):  str(np.sign(i)) for i in elements if i !=0.0}
        nx.set_edge_attributes(self.graph, name='excitetory_inhibitory', values=excitetory_inhibitory)
        #Add network graph from networkx
        self.network_graph = from_networkx(self.graph, self.pos, scale=10, center=(0, 0))
        #Set node size and color
        self.network_graph.node_renderer.glyph = Circle(size=15, 
        fill_color=bokeh.transform.factor_cmap('spiked_neurons', palette=bokeh.palettes.cividis(2), factors=('False', 'True')))
        #Set edge opacity and width
        self.network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1, line_color = bokeh.transform.factor_cmap('excitetory_inhibitory', palette=['#400000', '#004000'], factors=['-1.0', '1.0']))
        #Set edge highlight colors
        self.network_graph.edge_renderer.hover_glyph = MultiLine(line_color='blue')
        #Highlight nodes and edges
        self.network_graph.inspection_policy = bokeh.models.NodesAndLinkedEdges()
        #Append to plot
        self.plot.renderers = [self.network_graph]

    def display(self):
        self.layout = bokeh.layouts.layout([
            [self.plot],
            [self.slider, self.button]
        ], sizing_mode='scale_width')
        curdoc().add_root(self.layout)
        curdoc().title = "Neuron Group"
        os.system(f'cmd /c "bokeh serve --show {self.file_path}"')


class hv_NetworkPanel:
    import holoviews as hv
    from holoviews import opts
    hv.extension('bokeh')
    #TODO: complete this class based on: http://holoviews.org/user_guide/Network_Graphs.html
    def __init__(self,group, input_neurons, file_path):
        self.group = group
        self.file_path = file_path
        self.input_neurons = input_neurons
        rows, cols = np.where(self.group.AdjacencyMatrix == 1)
        edges = zip(rows.tolist(), cols.tolist())
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(edges)
        self.pos = self.group._set_pos(self.graph, self.input_neurons)
        plot = hv.Graph.from_networkx(self.graph, self.pos).opts(tools=['hover'])
        hv.save(plot, 'plt.html')
