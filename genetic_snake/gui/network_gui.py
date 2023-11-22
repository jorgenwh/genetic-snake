from PyQt5 import QtGui, QtCore, QtWidgets
from typing import List
import numpy as np

class NetworkGUI(QtWidgets.QWidget):
    def __init__(self, parent, dims: List[int]):
        super().__init__(parent)
        self.dims = dims
        self.largest_layer = max(self.dims)
        self.neuron_locations = {}
        
        self.activations = None
        self.params = None

        self.layer_gap = 175
        self.neuron_radius = 7.5
        self.neuron_gap = 4.5

        self.show()

    def draw(self, activations: List[np.ndarray], params: List[np.ndarray]):
        self.activations = activations
        self.params = params
        self.repaint()

    def paintEvent(self, event):
        if self.activations is None or self.params is None:
            return 

        painter = QtGui.QPainter()
        painter.begin(self)

        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        self.draw_neurons(painter)
        self.draw_synapses(painter)
        painter.end()

    def draw_neurons(self, painter):
        output_text = ("up", "right", "down", "left")

        for i, layer_dim in enumerate(self.dims):
            x = self.layer_gap * i + 5
            neuron_offset = (self.frameGeometry().height() - ((2 * self.neuron_radius + self.neuron_gap) * layer_dim)) / 2.0
            activations = self.activations[i]

            for n in range(layer_dim):
                y = n * (self.neuron_radius * 2 + self.neuron_gap) + neuron_offset
                neuron = (i, n)
                self.neuron_locations[neuron] = (x, y + self.neuron_radius)

                painter.setBrush(QtGui.QBrush(QtCore.Qt.white, QtCore.Qt.NoBrush))
                activation = activations[0,n]

                if i == 0:
                    if activation > 0:
                        painter.setBrush(QtGui.QBrush(QtGui.QColor(255 - (activation * 255), 255 - (activation * 255), 255 - (activation * 255))))
                    else:
                        painter.setBrush(QtGui.QBrush(QtCore.Qt.white))

                elif i == len(self.dims) - 1:
                    text = output_text[n]
                    painter.drawText(x + 25, n * (self.neuron_radius * 2 + self.neuron_gap) + neuron_offset + 1.5 * self.neuron_radius, text)

                    if n == np.argmax(activations):
                        painter.setBrush(QtGui.QBrush(QtCore.Qt.black))
                    else:
                        painter.setBrush(QtGui.QBrush(QtCore.Qt.white))

                else:
                    if activation >= 0:
                        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, max(255 - (activation * 75), 0), max(255 - (activation * 75), 0))))
                    else:
                        painter.setBrush(QtGui.QBrush(QtCore.Qt.white))
                
                # draw the neuron
                painter.drawEllipse(x, y, self.neuron_radius * 2, self.neuron_radius * 2)
        
    def draw_synapses(self, painter):
        for i, params in enumerate(self.params):
            prev_neurons = params.shape[0]
            cur_neurons = params.shape[1]

            for p_n in range(prev_neurons):
                for c_n in range(cur_neurons):
                    synapse = params[p_n, c_n]

                    if synapse is None:
                        painter.setPen(QtGui.QPen(QtGui.QColor(35, 35, 35)))
                    else:
                        strength = max(synapse * 750, 50)
                        strength = min(strength, 65)
                
                        if synapse >= 0:
                            painter.setPen(QtGui.QPen(QtGui.QColor(strength*2.25, strength, strength)))
                        else:
                            painter.setPen(QtGui.QPen(QtGui.QColor(strength, strength, strength*1.5)))

                    start = self.neuron_locations[i,p_n]
                    end = self.neuron_locations[i+1,c_n]

                    # draw synapse
                    painter.drawLine(start[0] + self.neuron_radius * 2, start[1], end[0], end[1])
