import numpy as np
import pickle
from IPython import embed


class Weights:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, uniform):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # x weight values
        self.wix = np.random.uniform(-uniform, uniform, (self.input_nodes, self.hidden_nodes))
        self.wgx = np.random.uniform(-uniform, uniform, (self.input_nodes, self.hidden_nodes))
        self.wfx = np.random.uniform(-uniform, uniform, (self.input_nodes, self.hidden_nodes))
        self.wox = np.random.uniform(-uniform, uniform, (self.input_nodes, self.hidden_nodes))

        # h weight values
        self.wih = np.random.uniform(-uniform, uniform, (1, self.hidden_nodes))
        self.wgh = np.random.uniform(-uniform, uniform, (1, self.hidden_nodes))
        self.wfh = np.random.uniform(-uniform, uniform, (1, self.hidden_nodes))
        self.woh = np.random.uniform(-uniform, uniform, (1, self.hidden_nodes))

        # bias terms
        self.bi = np.random.uniform(-uniform, uniform, (self.hidden_nodes, 1))
        self.bg = np.random.uniform(-uniform, uniform, (self.hidden_nodes, 1))
        self.bf = np.random.uniform(-uniform, uniform, (self.hidden_nodes, 1))
        self.bo = np.random.uniform(-uniform, uniform, (self.hidden_nodes, 1))

        # output weights
        self.whv = np.random.uniform(-uniform, uniform, (self.hidden_nodes, self.output_nodes))
        self.bv = np.random.uniform(-uniform, uniform, (self.output_nodes, 1))


class State:
    def __init__(self, i, g, f, o, s, h, v):
        self.i = i
        self.g = g
        self.f = f
        self.o = o
        self.s = s
        self.h = h
        self.v = v


class LSTM:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights = Weights(input_nodes, hidden_nodes, output_nodes, 0.08)

    def predict(self, x):
        h = np.zeros((self.hidden_nodes, 1))
        s = np.zeros((self.hidden_nodes, 1))
        v = np.zeros((self.output_nodes, 1))

        state = State(h, h, h, h, s, h, v)
        states = [state]

        for _, token in enumerate(x):
            try:
                assert token.shape == (self.input_nodes, 1)
            except AssertionError:
                print token.shape

            g = np.tanh(np.dot(self.weights.wgx.T, token) + self.weights.wgh.T*h + self.weights.bg)
            i = self.sigmoid(np.dot(self.weights.wix.T, token) + self.weights.wih.T*h + self.weights.bi)
            f = self.sigmoid(np.dot(self.weights.wfx.T, token) + self.weights.wfh.T*h + self.weights.bf)
            o = self.sigmoid(np.dot(self.weights.wox.T, token) + self.weights.woh.T*h + self.weights.bo)

            s = g*i + f*s
            h = o*s

            v = self.softmax(np.dot(self.weights.whv.T, h) + self.weights.bv)

            state = State(i, g, f, o, s, h, v)
            states.append(state)

        return states

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def save_weights(self, filename):
        weights_file = open(filename, 'wb')
        pickle.dump(self.weights, weights_file)
        weights_file.close()
