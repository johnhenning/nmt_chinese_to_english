import numpy as np
import nltk
import h5py
import os
import xml.etree.ElementTree as ET


class CorpusFileMapping:
    def __init__(self,english_filename,chinese_filename,sentence_mappings):
        self.english_filename = english_filename
        self.chinese_filename = chinese_filename
        self.sentence_mappings = sentence_mappings

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
        sentence_mappings = self.ReadSentenceMapping(filename)

        self.ProcessSentenceMappings(sentence_mappings)

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

    def ProcessSentenceMappings(self,file_mappings):
        for fm in file_mappings:
            english_data = self.ProcessCorpusFile(fm.english_file,'English')
            chinese_data = self.ProcessCorpusFile(fm.chinese_file, 'Chinese')
            english_data, chinese_data = self.AlignDatasets(english_data,chinese_data,fm.sentence_mappings)
            self.EnglishDataset.extend(english_data)
            self.ChineseDataset.extend(chinese_data)

    def ReadSentenceMapping(self,xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        file_maps = []
        for linkGroup in root:
            english_file = linkGroup.attrib['fromDoc']
            chinese_file = linkGroup.attrib['toDoc']
            sentence_mappings = []
            for link in linkGroup:
                mapping = processXMLMapping(link.attrib['xtargets'])
                sentence_mappings.append(mapping)

            file_map = CorpusFileMapping(english_file,chinese_file,sentence_mappings)
            file_maps.append(file_map)

        return file_maps

    def AlignDatasets(self,english_data,chinese_data,sentence_mappings):
        edata = []
        cdata = []
        for sm in sentence_mappings:
            english = []
            for i in sm[0]:
                english.extend(english_data[i])
            chinese = []
            for i in sm[1]:
                chinese.extend(chinese_data[i])

            edata.append(english)
            cdata.append(chinese)

        return edata, cdata


    def processXMLMapping(self,link_attrib):
        english_chinese_split = link_attrib.split(';')
        english_chinese_split[0] = map(int,english_chinese_split[0].split(' '))
        english_chinese_split[1] = map(int,english_chinese_split[1].split(' '))
        return english_chinese_split

    #this will need to change based on different xml structures, but for our data set, this splits and tokenizes the sentences
    def ProcessCorpusFile(self, filename, language):
        data =[]
        tree = ET.parse(filename)
        root = tree.getroot()

        for child in root:
            data.append("<s>")
            for token in child:
                if(token.tag == 'w'):
                    self.AddToDictionary(token.text, language)
                    data.append(token.text)
            data.append("</s>")

        return data



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
