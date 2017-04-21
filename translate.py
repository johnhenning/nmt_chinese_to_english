import numpy as np
from scipy import spatial as sp
import argparse
import cPickle
from model import LSTM
from IPython import embed
import random

def findWord(vector, dataset):
    vec = np.array(vector)

    max = float("-inf")
    maxWord = ""
    dist = 0
    for word, vecValue in dataset.iteritems():
        dist = 1 - random.random()*sp.distance.cosine(vec, vecValue)
        if dist > max:
            max = dist
            maxWord = word

    return maxWord


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test.')
    parser.add_argument(nargs='*', action='store', dest='input', type=str, help='The text to parse.')
    args = parser.parse_args()

    sentence = args.input

    english_dict = cPickle.load(open('english_dictionary.pkl'))
    chinese_dict = cPickle.load(open('chinese_dictionary.pkl'))

    encoder = LSTM(50, 100, 50)
    encoder.load_weights('encoder.pkl')
    decoder = LSTM(50, 100, 50)

    mat = []
    for word in sentence:
        mat.append(english_dict[word])
    mat = np.array(mat)
    mat = mat.reshape((mat.shape[0], mat.shape[1], 1))

    output = encoder.predict(mat)
    final = decoder.predict([output[-1].v], output[-1].h)

    translated_sentence = ''
    for word in final:
        translated_sentence += findWord(word.v, chinese_dict)

    print translated_sentence