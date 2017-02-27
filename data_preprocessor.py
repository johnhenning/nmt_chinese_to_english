import numpy as np
import nltk
import h5py

class DatasetProcessor:
    def __init__(self):
        self.ChineseDictionary = {}
        self.EnglishDictionary = {}
        self.EnglishDataset = []
        self.ChineseDataset = []

    def CreateDataset(self, filename):
        corpus_file = LoadCorpusFile(filename)
        english_data, chinese_data = SeparateLanguages(corpus_file)
        english_corpus = ProcessCorpus(english_data, 'English')
        chinese_corpus = ProcessCorpus(chinese_data, 'Chinese')

    def LoadCorpusFile(self, filename):

    def SeparateLanguages(self, corpus_file):

    def ProcessCorpus(self, language_data, language):

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


    def TokenizeSentence(self,sentence):


