from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np
import random

from settings import settings
from neural_network import NeuralNetwork


class NeuralNetWidget(QtWidgets.QWidget):
    def __init__(self, parent, network):
        super().__init__(parent)

        self.network = network
        self.largest_layer = max(self.network.layer_dim)
        self.neuron_locations = {}       

        self.show()


    def update(self, update_window: bool) -> None:
        if update_window:
            self.repaint()


    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        painter.begin(self)

        self.draw_neural_network(painter)

        painter.end()


    def draw_neural_network(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        height = self.frameGeometry().height()

        layer_gap = settings['layer_offset']
        neuron_radius = settings['neuron_radius']
        neuron_gap = settings['neuron_offset']

        layer_dim = self.network.layer_dim
        output_text = ('up', 'down', 'left', 'right')

        # Draw network nodes
        for layer, n_nodes in enumerate(layer_dim):
            # Calculate the x position of the layer and the offset distance between each neuron in the layer
            x = layer * layer_gap
            neuron_offset = (height - ((2 * neuron_radius + neuron_gap) * n_nodes)) / 2.0

            # Get the neuron activations for the current layer
            if layer == 0:
                activations = self.network.input_vector[0]
            else:
                activations = self.network.activations[layer-1][0]

            for node in range(n_nodes):
                # Calculate the y position of each neuron
                y = node * (neuron_radius * 2 + neuron_gap) + neuron_offset

                # Store the locations of the neurons so we can connect the weights
                neuron = (layer, node)
                if neuron not in self.neuron_locations:
                    self.neuron_locations[neuron] = (x, y + neuron_radius)

                painter.setBrush(QtGui.QBrush(QtCore.Qt.white, QtCore.Qt.NoBrush))
                activation = activations[node]

                if layer == 0:
                # If we are in the input layer
                    if activation > 0:
                        painter.setBrush(QtGui.QBrush(QtGui.QColor(255-(activations[node]*255), 255, 255)))
                    else:
                        painter.setBrush(QtGui.QBrush(QtCore.Qt.white))

                elif layer == len(layer_dim) - 1:
                # If we are in the output layer
                    text = output_text[node]
                    painter.drawText(x + 25, node * (neuron_radius * 2 + neuron_gap) + neuron_offset + 1.5 * neuron_radius, text)

                    if node == np.argmax(activations):
                        painter.setBrush(QtGui.QBrush(QtCore.Qt.cyan))
                    else:
                        painter.setBrush(QtGui.QBrush(QtCore.Qt.white))

                else:
                # If we are in a hidden layer
                    if activation >= 0:
                        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, max(255-(activations[node]*75),0), max(255-(activations[node]*75),0))))
                    else:
                        painter.setBrush(QtGui.QBrush(QtCore.Qt.white))

                # Draw the neuron
                painter.drawEllipse(x, y, neuron_radius * 2, neuron_radius * 2)

        # Draw network weights
        for i, weights in enumerate(self.network.weights):
            previous_nodes = weights.shape[0]
            current_nodes = weights.shape[1]

            for p_node in range(previous_nodes):
                for c_node in range(current_nodes):
                    weight = weights[p_node, c_node]
                    strength = max(weight * 750, 50)
                    strength = min(strength, 65)

                    if weight >= 0:
                        #painter.setPen(QtGui.QPen(QtGui.QColor(strength*2.25, strength, strength)))
                        painter.setPen(QtGui.QPen(QtGui.QColor(strength*2.25, strength, strength)))
                    else:
                        #painter.setPen(QtGui.QPen(QtGui.QColor(strength, strength, strength*1.65)))
                        painter.setPen(QtGui.QPen(QtGui.QColor(strength, strength, strength*1.5)))

                    # Draw the weight
                    start = self.neuron_locations[i, p_node]
                    end = self.neuron_locations[i+1, c_node]
                    painter.drawLine(start[0] + neuron_radius * 2, start[1], end[0], end[1])