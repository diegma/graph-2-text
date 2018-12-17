# -*- coding: utf-8 -*-
"""
Laura Perez

This class knows the structure of the server output (given by corenlp.py)
"""

import jsonrpc
import json

class CoreNLPService:
    server = None #Instance of Standford CoreNLP server

    def __init__(self):
       self.server = self.get_server()

    def get_server(self, hostname='localhost', port=3456, timeout=300.0):
        version = jsonrpc.JsonRpc20()
        transport = jsonrpc.TransportTcpIp(addr=(hostname, port), timeout=timeout)
        server = jsonrpc.ServerProxy(version, transport)
        return server

    def getParse(self, text):
        return json.loads(self.server.parse(text))

    def transform(self, sentence):
        """
         Each dependency is: [u'compound', u'member', u'5', u'crew', u'4']
        """
        #for rel, _, head, word, n in sentence['dependencies']:
        for rel, head, nh, word, nw in sentence['dependencies']:
             n = int(nh)

             word_info = sentence['words'][n - 1][1]
             tag = word_info['PartOfSpeech']
             lemma = word_info['Lemma']
             if rel == 'root':
                 # NLTK expects that the root relation is labelled as ROOT!
                 rel = 'ROOT'

             # Hack: Return values we don't know as '_'.
             #       Also, consider tag and ctag to be equal.
             # n is used to sort words as they appear in the sentence.
             yield nh, '_', word, lemma, tag, tag, '_', head, rel, '_', '_'


    def getDependencyTuples(self, parses):
        """
        From server output, extract dependency parse formatted as a list of * rel(head,word) *

        :param parses:
        :return: a dictionary with each sentence and its dependency parse
        """
        sents = {}
        for i in range(len(parses['sentences'])):
            tuples = []
            for rel, head, nh, word, nw in parses['sentences'][i]['dependencies']:  ##[u'compound', u'member', u'5', u'crew', u'4']
                word_info = parses['sentences'][i]['words'][int(nh) - 1][1]
                htag = word_info['PartOfSpeech']
                word_info = parses['sentences'][i]['words'][int(nw) - 1][1]
                wtag = word_info['PartOfSpeech']
                if head == 'ROOT':
                    htag = 'ROOT'
                #print "{}({}:{}-{}, {}:{}-{})".format(rel, head, htag, nh, word, wtag, nw)
                tuples.append("{}({}:{}-{}, {}:{}-{})".format(rel.encode('utf-8'), head.encode('utf-8'), htag.encode('utf-8'), nh, word.encode('utf-8'), wtag.encode('utf-8'), nw))
            sents[i] = tuples
        return sents


    def getLemmaPoS(self, parses):
        """
        Extracts from JSON file output by the Parser API, a dictionary of sentences where each sentences is formatted as
          a list of Lemma_POS as required by LCA.

        Example: word_info is [u'is', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd': u'9', u'CharacterOffsetBegin': u'7', u'PartOfSpeech': u'VBZ', u'Lemma': u'be'}]

        Note1: when there is a URL in the parsed text, the parser would return the URL as word but the associated info is empty,
        then we add a control for this and return URL_SYM for this case.

        Note2: For spacial characters in text, e.g. "mehreen syed -lrb- urdu : م ﮩ رين سيد -rrb- born august 2 , 1982 is a pakistani dta , the ceo of ifap and an actor ."
        the parser returns None for the 7th word. For the others it converted to something equivalent word, eg." : m None ryn syd )"

        :param parses:
        :return: formatted POS tagged sentences
        """

        sents = {}
        for i in range(len(parses['sentences'])):
            sentence = []
            for word_info in parses['sentences'][i]['words']:
                if word_info[0]==None:
                    wlemma = 'UNK'
                    if 'PartOfSpeech' in word_info[1].keys():
                        wtag = word_info[1]['PartOfSpeech']
                    else:
                        wtag = 'SYM'
                elif word_info[0].startswith("https://"):
                    wlemma = 'URL'
                    wtag = 'SYM'
                else:
                    if 'Lemma' in word_info[1].keys():
                        wlemma = word_info[1]['Lemma']
                        wtag = word_info[1]['PartOfSpeech']
                    else:
                        wlemma = word_info[0]
                        wtag = 'SYM'
                sentence.append("{}_{}".format(wlemma.encode('utf8'), wtag.encode('utf8')))
            sents[i] = sentence
        return sents


    def getWordPoS(self, parses):
        """
        Extracts from JSON file output by the Parser API.

        Example: word_info is [u'is', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd': u'9', u'CharacterOffsetBegin': u'7', u'PartOfSpeech': u'VBZ', u'Lemma': u'be'}]

        Similar to getLemmaPoS(parses) for the treatment of URLs in parsed text.

        :param parses:
        :return:  Sentences formatted as a list of Word_PoS as required by the Collins Parser (pre-processing step for the SCA).
        """

        sents = {}
        for i in range(len(parses['sentences'])):
            sentence = []
            for word_info in parses['sentences'][i]['words']:
                if word_info[0]==None:
                    word = 'UNK'
                    if 'PartOfSpeech' in word_info[1].keys():
                        wtag = word_info[1]['PartOfSpeech']
                    else:
                        wtag = 'SYM'
                elif word_info[0].startswith("https://"):
                    word = 'URL'
                    wtag = 'SYM'
                else:
                    word = word_info[0]
                    if 'PartOfSpeech' in word_info[1].keys():
                        wtag = word_info[1]['PartOfSpeech']
                    else:
                        wtag = 'SYM'
                if word == "(":
                    word = "-LRB-"
                    wtag = "-LRB-"
                if word == ")":
                    word = "-RRB-"
                    wtag = "-RRB-"
                sentence.append("{} {}".format(word.encode('utf8'), wtag.encode('utf8')))
            sents[i] = sentence
        return sents


    def countTokens(self, parses):
        """
        Counts the total number of tokens in the text (i.e. sums up through all sentences in the text).
        """
        totalTokens = 0
        for i in range(len(parses['sentences'])):
            totalTokens += len(parses['sentences'][i]['words'])

        return totalTokens

    def getSentences(self, parse):
        sents = []
        for i in range(len(parse['sentences'])):
            sents.append(parse['sentences'][i]['text'])
        return sents

    def getSentencesSpaced(self, parse):
        sents = []
        for i in range(len(parse['sentences'])):
            sentence = []
            for word_info in parse['sentences'][i]['words']:
                if word_info[0] == None:
                    word = 'UNK'
                elif word_info[0].startswith("https://"):
                    word = 'URL'
                else:
                    word = word_info[0]
                if word == "(":
                    word = "-LRB-"
                    wtag = "-LRB-"
                if word == ")":
                    word = "-RRB-"
                    wtag = "-RRB-"
                sentence.append("{} ".format(word.encode('utf8')))
            sents.append("".join(sentence))
        return sents

    def getSentenceSplit(self, text):
        return  self.getSentencesSpaced(self.getParse(text))

    def getWordNER(self, parses):
        """
        Extracts from JSON file output by the Parser API.

        Example: word_info is [u'is', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd': u'9', u'CharacterOffsetBegin': u'7', u'PartOfSpeech': u'VBZ', u'Lemma': u'be'}]

        Similar to getLemmaPoS(parses) for the treatment of URLs in parsed text.

        :param parses:
        :return:  Sentences formatted as a list of Word_NER as required by the Collins Parser (pre-processing step for the SCA).
        """

        sents = {}
        for i in range(len(parses['sentences'])):
            sentence = []
            for word_info in parses['sentences'][i]['words']:
                if word_info[0] == None:
                    word = 'UNK'
                    wner = None
                elif word_info[0].startswith("https://"):
                    word = 'URL'
                    wner = 'URL'
                else:
                    word = word_info[0]
                    if 'NamedEntityTag' in word_info[1].keys():
                        wner = word_info[1]['NamedEntityTag']
                    else:
                        wner = None
                if word == "(":
                    word = "-LRB-"
                    wner = None
                if word == ")":
                    word = "-RRB-"
                    wner = None
                sentence.append((word.encode('utf8'), wner.encode('utf8')))
            sents[i] = sentence
        return sents
