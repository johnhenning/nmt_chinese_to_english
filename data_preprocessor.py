import numpy as np
import os
import gzip
import pickle
from IPython import embed
import xml.etree.ElementTree as ET


class CorpusFileMapping:
    def __init__(self, english_filename, chinese_filename, sentence_mappings):
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
        sentence_mappings = self.read_sentence_mapping(filename)
        self.ProcessSentenceMappings(sentence_mappings)

        if saveDictionary:
            self.save_dictionaries()


    def LoadCorpusFiles(self, filename):
        english_corpus_files = []
        chinese_corpus_files = []

        return english_corpus_files, chinese_corpus_files

    def CloseCorpusFiles(self, files):
        for f in files:
            f.close()

    def ProcessSentenceMappings(self, file_mappings, saveDatasets=True):
        dataset_count = 0
        for i, fm in enumerate(file_mappings):
            print "Processing " + fm.english_filename + " and " + fm.chinese_filename
            english_data = self.ProcessCorpusFile(fm.english_filename, 'English')
            chinese_data = self.ProcessCorpusFile(fm.chinese_filename, 'Chinese')
            english_data, chinese_data = self.AlignDatasets(english_data, chinese_data, fm.sentence_mappings)
            print "Aligned " + fm.english_filename + " and " + fm.chinese_filename
            self.EnglishDataset.extend(english_data)
            self.ChineseDataset.extend(chinese_data)
            if i % 25 == 24:
                if saveDatasets:
                    print "Saving Dataset" + str(dataset_count)
                    self.saveDatasets(dataset_count)
                    dataset_count += 1

                self.EnglishDataset = []
                self.ChineseDataset = []

        self.saveDatasets(dataset_count)

    def read_sentence_mapping(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        file_maps = []
        for linkGroup in root:
            english_file = linkGroup.attrib['fromDoc']
            chinese_file = linkGroup.attrib['toDoc']
            sentence_mappings = []
            for link in linkGroup:
                mapping = self.processXMLMapping(link.attrib['xtargets'])
                sentence_mappings.append(mapping)

            file_map = CorpusFileMapping(english_file, chinese_file, sentence_mappings)
            file_maps.append(file_map)

        return file_maps

    def AlignDatasets(self, english_data, chinese_data, sentence_mappings):
        edata = []
        cdata = []
        for sm in sentence_mappings:
            english = []
            for i in sm[0]:
                try:
                    english.extend(english_data[i - 1])
                except:
                    print len(english_data)
                    print i
            chinese = []
            for i in sm[1]:
                chinese.extend(chinese_data[i - 1])

            edata.append(english)
            cdata.append(chinese)

        return edata, cdata

    def processXMLMapping(self, link_attrib):
        english_chinese_split = link_attrib.split(';')
        for s in range(len(english_chinese_split)):
            if english_chinese_split[s] is '':
                english_chinese_split[s] = '-1'

        english_chinese_split[0] = map(int, english_chinese_split[0].split(' '))
        english_chinese_split[1] = map(int, english_chinese_split[1].split(' '))
        return english_chinese_split

    # this will need to change based on different xml structures, but for our data set, this splits and tokenizes the sentences
    def ProcessCorpusFile(self, filename, language):
        with gzip.open(filename, 'rb') as f:
            tree = ET.parse(f)
        data = []
        root = tree.getroot()
        f.close()

        for child in root:
            sentence = []
            for token in child:
                if (token.tag == 'w'):
                    text = token.text
                    if language is 'English':
                        text = self.fix_lower_l(text)

                    self.add_to_dictionary(text, language)
                    sentence.append(text)
            sentence.append("</s>")
            data.append(sentence)
        return data

    def fix_lower_l(self, text):
        if 'l' in text:
            if text.replace('l', '') == text.replace('l', '').upper():
                text = text.replace('l', 'I')
        return text

    def add_to_dictionary(self, word, language):
        d = None
        if language is 'English':
            d = self.EnglishDictionary

        elif language is 'Chinese':
            d = self.ChineseDictionary

        if word not in d.keys():
            d[word] = len(d.keys())

    def save_dictionaries(self):
        with open('Chinese_Dictionary.pkl', 'wb') as f:
            pickle.dump(self.ChineseDictionary, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        with open('English_Dictionary.pkl', 'wb') as f:
            pickle.dump(self.EnglishDictionary, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def saveDatasets(self, dataset_count):
        e_filename = "pickle/english_dataset_" + str(dataset_count) + ".pkl"
        c_filename = "pickle/chinese_dataset_" + str(dataset_count) + ".pkl"
        e_file = open(e_filename, 'wb')
        c_file = open(c_filename, 'wb')
        pickle.dump(self.EnglishDataset, e_file)
        pickle.dump(self.ChineseDataset, c_file)
        e_file.close()
        c_file.close()

def main():
    dp = DatasetProcessor()
    dp.CreateDataset('en-zh_cn.xml')
    embed()


if __name__ == '__main__':
    main()
