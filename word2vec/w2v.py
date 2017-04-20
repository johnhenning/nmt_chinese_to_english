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


class VocabWord:
    def __init__(self, word):
        self.word = word
        self.count = 0
        self.path = None  # Path (list of indices) from the root to the word (leaf)
        self.code = None  # Huffman encoding


class Vocab:
    def __init__(self):
        self.vocab = []
        self.vocabMap = {}
        self.word_count = 0
        self.bytes = 0

    def load_file(self, wf):
        wordfile = pickle.load(wf)

        # Add special tokens <bol> (beginning of line) and <eol> (end of line)
        for sentence in wordfile:
            for token in sentence:
                if token not in self.vocabMap:
                    self.vocabMap[token] = len(self.vocab)
                    self.vocab.append(VocabWord(token))

                # assert vocab[vocabMap[token]].word == token, 'Wrong vocabMap index'
                self.vocab[self.vocabMap[token]].count += 1
                self.word_count += 1

                if self.word_count % 10000 == 0:
                    sys.stdout.write("\rReading word %d" % self.word_count)
                    sys.stdout.flush()

            # Add special tokens <bol> (beginning of line) and <eol> (end of line)

        self.bytes += wf.tell()
        # Total number of words in train file

        # Add special token <unk> (unknown),
        # merge words occurring less than 5 times into <unk>, and
        # sort vocab in descending order by frequency in train file
        self.__sort()

        # assert self.word_count == sum([t.count for t in self.vocab]), 'word_count and sum of t.count do not agree'
        print 'Total words in training file: %d' % self.word_count
        print 'Total bytes in training file: %d' % self.bytes
        print 'Vocab size: %d' % len(self)

        return wordfile

    def __getitem__(self, i):
        return self.vocab[i]

    def __len__(self):
        return len(self.vocab)

    def __iter__(self):
        return iter(self.vocab)

    def __contains__(self, key):
        return key in self.vocabMap

    def __sort(self):
        tmp = []
        tmp.append(VocabWord('<unk>'))
        unk_hash = 0

        count_unk = 0
        for token in self.vocab:
            if token.count < 2:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token: token.count, reverse=True)

        # Update vocabMap
        vocabMap = {}
        for i, token in enumerate(tmp):
            vocabMap[token.word] = i

        self.vocab = tmp
        self.vocabMap = vocabMap

        print
        print 'Unknown vocab size:', count_unk

    def indices(self, tokens):
        return [self.vocabMap[token] if token in self else self.vocabMap['<unk>'] for token in tokens]

    def encode_huffman(self):
        # Build a Huffman tree
        vocab_size = len(self)
        count = [t.count for t in self] + [1e15] * (vocab_size - 1)
        parent = [0] * (2 * vocab_size - 2)
        binary = [0] * (2 * vocab_size - 2)

        pos1 = vocab_size - 1
        pos2 = vocab_size

        for i in xrange(vocab_size - 1):
            # Find min1
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1 = pos1
                    pos1 -= 1
                else:
                    min1 = pos2
                    pos2 += 1
            else:
                min1 = pos2
                pos2 += 1

            # Find min2
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2 = pos1
                    pos1 -= 1
                else:
                    min2 = pos2
                    pos2 += 1
            else:
                min2 = pos2
                pos2 += 1

            count[vocab_size + i] = count[min1] + count[min2]
            parent[min1] = vocab_size + i
            parent[min2] = vocab_size + i
            binary[min2] = 1

        # Assign binary code and path pointers to each vocab word
        root_idx = 2 * vocab_size - 2
        for i, token in enumerate(self):
            path = []  # List of indices from the leaf to the root
            code = []  # Binary Huffman encoding from the leaf to the root

            node_idx = i
            while node_idx < root_idx:
                if node_idx >= vocab_size: path.append(node_idx)
                code.append(binary[node_idx])
                node_idx = parent[node_idx]
            path.append(root_idx)

            # These are path and code from the root to the leaf
            token.path = [j - vocab_size for j in path[::-1]]
            token.code = code[::-1]


class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """

    def __init__(self, vocab):
        vocab_size = len(vocab)
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab])  # Normalizing constant

        table_size = int(1e8)  # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        print 'Filling unigram table'
        p = 0  # Cumulative probability
        i = 0
        for j, unigram in enumerate(vocab):
            p += float(math.pow(unigram.count, power)) / norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]


def sigmoid(z):
    try:
        sigmoid = 1 / (1 + math.exp(-z))
    except OverflowError:
        if z > 0:
            sigmoid = 1
        else:
            sigmoid = 0
    return sigmoid


def init_net(vocab_size):
    # Init syn0 with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim
    tmp = np.random.uniform(low=-0.5 / 50, high=0.5 / 50, size=(vocab_size, 50))
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)

    # Init syn1 with zeros
    tmp = np.zeros(shape=(vocab_size, 50))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)

    return (syn0, syn1)


def train_process(fi):
    alpha = starting_alpha

    word_count = 0
    last_word_count = 0

    sentences = pickle.load(fi)
    for sentence in sentences:
        # Init sent, a list of indices of words in line
        sent = vocab.indices(sentence)

        for sent_pos, token in enumerate(sent):
            if word_count % 10000 == 0:
                global_word_count.value += (word_count - last_word_count)
                last_word_count = word_count

                # Recalculate alpha
                alpha = starting_alpha * (1 - float(global_word_count.value) / vocab.word_count)
                if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001

                # Print progress info
                sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                                 (alpha, global_word_count.value, vocab.word_count,
                                  float(global_word_count.value) / vocab.word_count * 100))
                sys.stdout.flush()

            # Randomize window size, where win is the max window size
            current_win = np.random.randint(low=1, high=win + 1)
            context_start = max(sent_pos - current_win, 0)
            context_end = min(sent_pos + current_win + 1, len(sent))
            context = sent[context_start:sent_pos] + sent[sent_pos + 1:context_end]  # Turn into an iterator?

            # Compute neu1 using CBOW
            neu1 = np.mean(np.array([syn0[c] for c in context]), axis=0)
            assert len(neu1) == 50, 'neu1 and dim do not agree'

            # Init neu1e with zeros
            neu1e = np.zeros(50)

            classifiers = zip(vocab[token].path, vocab[token].code)
            for target, label in classifiers:
                z = np.dot(neu1, syn1[target])
                p = sigmoid(z)
                g = alpha * (label - p)
                neu1e += g * syn1[target]  # Error to backpropagate to syn0
                syn1[target] += g * neu1  # Update syn1

            # Update syn0
            for context_word in context:
                syn0[context_word] += neu1e

            word_count += 1

    # Print progress info
    global_word_count.value += (word_count - last_word_count)
    sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                     (alpha, global_word_count.value, vocab.word_count,
                      float(global_word_count.value) / vocab.word_count * 100))
    sys.stdout.flush()
    fi.close()


def save(vocab, syn0, fo):
    print 'Saving model to', fo
    dictWords = {}
    fo = open(fo, 'w')
    # fo.write('%d %d\n' % (len(syn0), 50))
    for token, vector in zip(vocab, syn0):
        word = token.word
        vector_str = ' '.join([str(s) for s in vector])
        dictWords[word] = vector_str

    pickle.dump(dictWords, fo)
    fo.close()


def __init_process(*args):
    global vocab, syn0, syn1, table, starting_alpha
    global win, global_word_count, fi

    vocab, syn0_tmp, syn1_tmp, table, starting_alpha, win, num_processes, global_word_count = args[:-1]
    fi = open('temp.pkl', "r")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp)
        syn1 = np.ctypeslib.as_array(syn1_tmp)


def train(fi, fo):
    # Read train file to init vocab
    global vocab, syn0, syn1, table, starting_alpha
    global global_word_count, win

    starting_alpha = 0.025
    win = 5

    vocab = Vocab()

    all_data = []
    for filename in os.listdir(fi.split("/")[0]):
        if filename.find(fi.split("/")[1]) >= 0:
            with open(fi.split('/')[0] + '/' + filename, 'rb') as readfile:
                print fi.split('/')[0] + '/' + filename
                all_data.extend(vocab.load_file(readfile))


    ff = open('temp.pkl', 'wb')
    pickle.dump(all_data, ff)
    ff.close()
    f = open('temp.pkl', "r")
    # Init net
    syn0, syn1 = init_net(len(vocab))

    global_word_count = Value('i', 0)
    table = None
    print 'Making tree'
    vocab.encode_huffman()

    # Begin training using num_processes workers
    t0 = time.time()

    train_process(f)
    t1 = time.time()
    print 'total time: ', t1 - t0
    print 'done'

    # Save model to file
    save(vocab, syn0, fo)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='Training file', dest='fi', required=True)
    parser.add_argument('-model', help='Output model file', dest='fo', required=True)
    parser.add_argument('-alpha', help='Starting alpha', dest='alpha', default=0.025, type=float)
    parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int)
    parser.add_argument('-processes', help='Number of processes', dest='num_processes', default=1, type=int)
    # TO DO: parser.add_argument('-epoch', help='Number of training epochs', dest='epoch', default=1, type=int)
    args = parser.parse_args()
    # trains model with 50 dimension word embeddings
    train(args.fi, args.fo)