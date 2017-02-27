import numpy as np
import nltk
import h5py
import os

class Sentence:
    def __init__(self, sentence, tag):
        self.tag = tag
        self.sentence = sentence

class DatasetProcessor:
    def __init__(self):
        self.ChineseDictionary = {}
        self.EnglishDictionary = {}
        self.EnglishDataset = []
        self.ChineseDataset = []

    def CreateDataset(self, filename, saveDictionary=True, saveDataset=True):
        english_corpus_files, chinese_corpus_files = self.LoadCorpusFiles(filename)

        for f in english_corpus_files:
            self.EnglishDataset.extend(ProcessCorpusFile(f)) #returns tokenized sentences

        for f in chinese_corpus_files:
            self.ChineseDataset.extend(ProcessCorpusFile(f))

        if saveDictionary:
            self.saveDictionaries()
        if saveDataset:
            self.saveDatasets()

    def LoadCorpusFiles(self, filename):
        english_corpus_files = []
        chinese_corpus_files = []

        return english_corpus_files, chinese_corpus_files

    def CloseCorpusFiles(self, files):
        for f in files:
            f.close()

    def ProcessCorpusFile(self):

    def Tokenize(self):


    def FillDictionaryWithCorpus(self,language,corpus):
        d = {}
        if language == 'English':
            d = self.EnglishDictionary

        elif language == 'Chinese':
            d = self.ChineseDictionary

        for sentence in corpus:
            for word in sentence:
                if d[word] is not None:
                    d[word] = len(d.keys())

    """
    Takes in set of all words in the processed corpus
    """

    def FillDictionaryWithSet(self,language,words):
        d = {}
        if language == 'English':
            d = self.EnglishDictionary

        elif language == 'Chinese':
            d = self.ChineseDictionary

        for word in words:
            if d[word] is not None:
                d[word] = len(d.keys())


    def saveDictionary(self):

    def saveDataset(self):



