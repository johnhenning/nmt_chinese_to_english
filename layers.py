import numpy as np
import pandas as pd

class Layer:
    def __init__(self,output_shape,input_shape):
        self.input_nodes = []
        self.output_nodes = []
        self.output_shape = output_shape
        self.input_shape = input_shape

class Dense(Layer):
    def __init__(self,output_shape,input_shape):
        super(output_shape,input_shape)

