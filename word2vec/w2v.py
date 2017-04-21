import argparse
import math
import struct
import sys
import time
import warnings
import shutil
import os
import pickle

import numpy as np

from multiprocessing import Pool, Value, Array


class Weights:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, uniform):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.wih = np.random.uniform(-uniform, uniform, (self.input_nodes, self.hidden_nodes))
        self.who = np.random.uniform(-uniform, uniform, (self.hidden_nodes, self.output_nodes))
        self.bih = np.random.uniform(-uniform, uniform, (self.hidden_nodes, 1))
        self.bho = np.random.uniform(-uniform, uniform, (self.output_nodes, 1))


class Word2VecNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights = Weights(input_nodes, hidden_nodes, output_nodes, 0.08)

    def predict(self, x):
        h = relu(np.dot(self.weights.wih.T, x) + self.weights.bih)
        o = relu(np.dot(self.weights.who.T, h) + self.weights.bho)
        return h, o

    def train(self, x, y, epochs):
        for i in range(epochs):
            diff = Weights(self.input_nodes, self.hidden_nodes, self.output_nodes, 0)

            total_cost = 0
            sys.stdout.write("\rEpoch %d" % i)
            for xi, yi in zip(x, y):
                h, output = self.predict(xi)
                total_cost += error(output, yi)
                sys.stdout.write("\rTotal Cost: %d" % total_cost)
                de_drelu = error_gradient(output, yi)
                drelu_dh = drelu_dx(output)
                drelu_dwih = xi
                drelu_dwho = h

                de_dwih = de_drelu*np.dot(drelu_dh.T, drelu_dwih)
                de_dwho = de_drelu*drelu_dwho

                de_dbih = drelu_dh
                de_dbho = de_drelu

                diff.wih += de_dwih / len(y)
                diff.who += de_dwho / len(y)
                diff.bih += de_dbih / len(y)
                diff.bho += de_dbho / len(y)

                sys.stdout.flush()

            sys.stdout.write("Total Cost: %d" % total_cost)

    def save_weights(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.weights, f)

    def load_weights(self, filename):
        f = open(filename)
        self.weights = pickle.load(f)


def error(output, y):
    c = 0
    for i in range(len(y)):
        c += 0.5 * (y[i] - output[i]) ** 2

    return c / len(y)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def relu(x):
    return np.array([max(x_i, 0) for x_i in x])


def error_gradient(output, y):
    return -abs(y - output)


def drelu_dx(x):
    if x > 0:
        return 1
    else:
        return 0


def getNeighboringWords(sentence, index):
    neighbors  = list()
    if index + 1 < len(sentence):
        neighbors.append(sentence[index+1])
    if index - 1 >= 0:
        neighbors.append(sentence[index-1])
    return neighbors

def generateDataset(filename):
    f = open(filename)
    sentences = pickle.load(f)

    dictionary = dict()

    for sentence in sentences:
        for word in sentence:
            if word.lower() not in dictionary:
                dictionary[word] = len(dictionary)

    wordToNeighbor = np.zeros((len(dictionary), len(dictionary)))

    for sentence in sentences:
        for i, word in enumerate(sentence):
            neighbors = [dictionary[x.lower()] for x in getNeighboringWords(sentence, i)]
            for neighbor in neighbors:
                wordToNeighbor[dictionary[word]][neighbor] += 1

    wordToNeighbor = normalize(wordToNeighbor)
    one_hot = np.identity(len(dictionary))

    return dictionary, wordToNeighbor, one_hot

def normalize(x):
    for col in range(x.shape[1]):
        x[:][col] /= np.max(x[:][col])

    return x

def main():
    english_dictionary, english_wordToNeighbor, english_one_hot = generateDataset('english_data.pkl')
    chinese_dictionary, chinese_wordToNeighbor, chinese_one_hot = generateDataset('chinese_data.pkl')

    english_network = Word2VecNetwork(len(english_dictionary), 100, len(english_dictionary))
    english_network.train(english_one_hot, english_wordToNeighbor, 20)
    english_network.save_weights('word2vec/english_network_weights.pkl')

    chinese_network = Word2VecNetwork(len(chinese_dictionary), 100, len(chinese_dictionary))
    chinese_dictionary.train(chinese_one_hot, chinese_wordToNeighbor, 20)
    chinese_network.save_weights('word2vec/chinese_network_weights.pkl')

if __name__ == '__main__':
    main()