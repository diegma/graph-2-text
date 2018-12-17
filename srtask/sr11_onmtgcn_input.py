# -*- coding: utf-8 -*-

import getopt, re, datetime, os, codecs
import sys
from operator import itemgetter
import json

sys.path.append('../')
from utils.CoreNLPService import CoreNLPService

#constants
EXAMPLE_START = "sentId="
NAMED_ENT_LABEL = "NAME_"
PREDICATE_ARG_REG = re.compile("^[a-z]+\.[0-9]{2}$")
PREDICATE_SU_REG = re.compile("^[a-z$]+\.SU$")
EMPTY_FEATURE = "<pad>"
TEST_GOLD_SENTS = "SRTESTB_sents.txt"

parseService = CoreNLPService()
stopWords = []
f = codecs.open('NLTKstopwords.txt','r', encoding='utf8')
for l in f.readlines():
    stopWords.append(l.strip())
vocabularyNEs = {}


def anonymiseSentence(s, anonymisedNodeTokens):
    s = s.lower()
    for token, type in anonymisedNodeTokens:
        if type:
            replaceSubstring = token.replace("_", " ").lower()
            s = s.replace(replaceSubstring, type)
    return s

def compactSentence(s, anonymisedNodeTokens):
    for token, type in anonymisedNodeTokens:
        if "_" in token:
            replaceSubstring = token.replace("_", " ")
            s = s.replace(replaceSubstring, token)
    return s

def anonymise(tokens):
    typesCount = {}
    anonymisedTokens = []
    for token, type in tokens:
        newType = type
        result1 = PREDICATE_ARG_REG.match(token)
        result2 = PREDICATE_SU_REG.match(token)
        # avoid calling the parser when not necessary
        if not result1 and not result2 and not token in stopWords:
            strToken = token.replace("_", " ")
            strToken = " ".join(e.title() for e in strToken.split())
            if not strToken in vocabularyNEs.keys():
                sents = parseService.getWordNER(parseService.getParse(strToken))
                entityTypes = set()
                for s in sents.keys():
                    for tok,net in sents[s]:
                        if not net == "O":
                            entityTypes.add(net)
                if entityTypes and len(entityTypes) == 1:
                    newType = entityTypes.pop()
                    vocabularyNEs[strToken] = newType
                else:
                    vocabularyNEs[strToken] = "O"

            else:
                #token already analysed with type, just take it
                if not vocabularyNEs[strToken] =="O":
                    newType = vocabularyNEs[strToken]
        if newType:
            if not newType in typesCount.keys():
                typesCount[newType] = -1
            typesCount[newType] += 1
            newType += str(typesCount[newType])
        anonymisedTokens.append((token, newType))
    return anonymisedTokens

def reindex(nodeIndex, removeFromPosition):
    newIndex = nodeIndex
    for pos in removeFromPosition:
        if nodeIndex > pos:
            newIndex -=1
    if nodeIndex in removeFromPosition:
        newIndex -=1
    return newIndex

def formatTree(tree):
    features = []
    nodes = []
    tokens = []
    edgesLabel = []
    edgesNode1 = []
    edgesNode2 = []
    namesForNodes = {}
    removeFromPosition = []
    previous_orig_nodeIdx = set()
    for nodeEdge in tree:
        # construct node_token and (label, nodeIdx, headIdx) edge

        orig_nodeIdx = int(nodeEdge[1]) - 1
        headIdx = int(nodeEdge[2]) - 1  if int(nodeEdge[2]) > 0 else 0
        orig_headIdx = headIdx

        nodeIdx = reindex(orig_nodeIdx, removeFromPosition)
        headIdx = reindex(headIdx, removeFromPosition)

        label = nodeEdge[0]
        nodeToken = None
        if len(nodeEdge) > 3 :
            nodeToken = nodeEdge[3]
        nodeFeatures = None
        if len(nodeEdge) > 4 :
            nodeFeatures = "_".join(nodeEdge[4:])

        if label.startswith(NAMED_ENT_LABEL):
            if not headIdx in namesForNodes.keys():
                namesForNodes[headIdx] = []
            namesForNodes[headIdx].append( (int(label.split(NAMED_ENT_LABEL)[1]), nodeToken) )
            if not orig_nodeIdx in removeFromPosition\
                    and not orig_nodeIdx in previous_orig_nodeIdx:
                removeFromPosition.append(orig_nodeIdx)
        else:
            if nodeToken:
                nodes.append((nodeIdx, nodeToken))
                features.append( nodeFeatures if nodeFeatures else EMPTY_FEATURE)

            # [19 20 21 22 23 24 25 26 27] sentID=28736 extra edge goes from a node to a node within the deleted edges
            # then need to make this point to the remaining compacted node rather than taking the next available position
            if not orig_nodeIdx in removeFromPosition \
                or not orig_headIdx in removeFromPosition:

                edgesLabel.append(label)
                edgesNode1.append(str(nodeIdx))
                edgesNode2.append(str(headIdx))
            else:
                print("both deleted")

        previous_orig_nodeIdx.add(orig_nodeIdx)

    for node in nodes:
        strName = node[1]
        k = node[0]
        entity = None
        if k in namesForNodes.keys():
            #reconstruct named entity name
            for x in namesForNodes[k]:
                if not x[1]:
                    print("WARNING: name recostruction partial failure")
            names = [x[1].title() if x[1] else "" for x in sorted(namesForNodes[k],key=itemgetter(0)) ]
            names = names + [strName.title()]
            strName = "_".join(names)
            entity = "NAME"
        tokens.append((strName, entity))

    assert(len(features) == len(tokens))
    return tokens, " ".join(features), (" ".join(edgesLabel), " ".join(edgesNode1), " ".join(edgesNode2))

def readTree(treeLines):
    tree = []
    for l in treeLines:
        tree.append(l.split())
    return tree

def conllReader(fileName):
    f = codecs.open(fileName, "r", encoding='utf8')
    conllTrees = []
    tgtSentences = []
    treeLines = []
    sentence = None
    nextIs = None
    #data file finishes with a empty line
    for line in f.readlines():
        line = line.strip()
        if line.startswith(EXAMPLE_START):
            #processing a new example
            nextIs = "tree"
        elif line and nextIs=="tree":
            treeLines.append(line.split())
        elif not line and nextIs == "tree":
            #finished reading tree, next will be target sentence
            nextIs = "tgt"
        elif line and nextIs == "tgt":
            sentence = line
        elif not line and nextIs == "tgt":
            #finished reading the example
            nextIs = None
            conllTrees.append(treeLines)
            tgtSentences.append(sentence)
            treeLines = []
            sentence = None

    print("* examples: {}".format(len(conllTrees)))
    return conllTrees, tgtSentences

def readTestSentence(fileSentence):
    fs = codecs.open(fileSentence, "r", encoding='utf8')
    tgtSentences = []
    nextIs = None
    for line in fs.readlines():
        line = line.strip()
        if line.startswith(EXAMPLE_START):
            #processing a new example
            nextIs = "tgt"
        elif line and nextIs=="tgt":
            tgtSentences.append(line)
        elif not line and nextIs == "tgt":
            #finished reading tree,
            nextIs = None
    return tgtSentences

def conllReaderTest(fileGraph, fileSentence):
    fg = codecs.open(fileGraph, "r", encoding='utf8')
    conllTrees = []
    treeLines = []
    nextIs = None
    #data file finishes with a empty line
    for line in fg.readlines():
        line = line.strip()
        if line.startswith(EXAMPLE_START):
            #processing a new example
            nextIs = "tree"
        elif line and nextIs=="tree":
            treeLines.append(line.split())
        elif not line and nextIs == "tree":
            #finished reading tree,
            nextIs = None
            conllTrees.append(treeLines)
            treeLines = []

    tgtSentences = readTestSentence(fileSentence)
    print("* test trees: {} sentences: {}".format(len(conllTrees), len(tgtSentences)))
    return conllTrees, tgtSentences

def tokensLine(tokens, option):
    if option== 'anonym':
        return " ".join([x[1] if x[1] else x[0] for x in tokens])
    else:
        return " ".join([x[0] for x in tokens])

def getAnonymisationDict(anonymisedNodeTokens):
    dict = {}
    for token, type in anonymisedNodeTokens:
        if type:
            dict[type] = token
    return dict

def format2gcninput(inputdir, dataset, task, option, conllTrees, target_out):
    new_target_out = []
    source_nodes_out = []
    source_features_out = []
    source_edges_out_labels = []
    source_edges_out_node2 = []
    source_edges_out_node1 = []
    anonymisationDictionary = {}
    sentID = 1
    for t, s in zip(conllTrees, target_out):
        #if sentID<12969:
        #if sentID<440:
        #    sentID +=1
        #    continue
        print("sentId={}".format(sentID))
        nodeTokens, features, source_edges = formatTree(t)

        if option== 'anonym':
            anonymisedNodeTokens = anonymise(nodeTokens)
            target_sentence = s
            if dataset in ['train', 'devel']:
                target_sentence = anonymiseSentence(s, anonymisedNodeTokens)

            source_nodes = tokensLine(anonymisedNodeTokens, option)
            anonymisationDictionary[sentID] = getAnonymisationDict(anonymisedNodeTokens)
        else:
            source_nodes = tokensLine(nodeTokens, option)
            target_sentence = compactSentence(s, nodeTokens)

        new_target_out.append(target_sentence)
        source_nodes_out.append(source_nodes)
        source_features_out.append(features)
        source_edges_out_labels.append(source_edges[0])
        source_edges_out_node1.append(source_edges[1])
        source_edges_out_node2.append(source_edges[2])
        sentID +=1

    if option== 'anonym':
        with codecs.open(inputdir + dataset + '-sr11-{}-'.format(task) + '-anonym.json', 'w+',
                         encoding='utf8') as f:
            f.write(json.dumps(anonymisationDictionary))
    with codecs.open(inputdir + dataset + '-sr11-{}-'.format(task) + option + '-src-nodes.txt', 'w+', encoding='utf8') as f:
        f.write('\n'.join(source_nodes_out))
    with codecs.open(inputdir + dataset + '-sr11-{}-'.format(task) + option + '-src-labels.txt', 'w+', encoding='utf8') as f:
        f.write('\n'.join(source_edges_out_labels))
    with codecs.open(inputdir + dataset + '-sr11-{}-'.format(task) + option + '-src-node1.txt', 'w+', encoding='utf8') as f:
        f.write('\n'.join(source_edges_out_node1))
    with codecs.open(inputdir + dataset + '-sr11-{}-'.format(task) + option + '-src-node2.txt', 'w+', encoding='utf8') as f:
        f.write('\n'.join(source_edges_out_node2))
    with codecs.open(inputdir + dataset + '-sr11-{}-'.format(task) + option + '-src-features.txt', 'w+', encoding='utf8') as f:
        f.write('\n'.join(source_features_out))
    with codecs.open(inputdir + dataset + '-sr11-{}-'.format(task) + option + '-tgt.txt', 'w+', encoding='utf8') as f:
        f.write('\n'.join(new_target_out))
    print("Files saved.")

def processDataFiles(inputdir, task, partitions = ['devel', 'train'] ):

    #options = ['anonym', 'notanonym']
    options = ['anonym']
    for dataset in partitions:
        for op in options:
            print("Processing task= {} partition= {}".format(task, dataset))
            if dataset=='test':
                conllTrees, targetSentences = \
                    conllReaderTest(inputdir +"/"+ dataset +"/"+ task +"/"+ dataset+task.title()+".hfg",
                                    inputdir +"/"+ dataset +"/"+TEST_GOLD_SENTS)
            else:
                conllTrees, targetSentences = \
                    conllReader(inputdir +"/"+ dataset +"/"+ task +"/"+ dataset+task.title()+".hfg")
            format2gcninput(inputdir, dataset, task, op, conllTrees, targetSentences)

def main(argv):
    usage = 'usage:\npython sr11_onmtgcn_input.py -i <dir> [-t TASK] [-p PARTITION]' \
            '\n TASK is either deep or shallow'\
            '\n PARTITION which partition to process, by default test/devel will be done.'
    try:
        opts, args = getopt.getopt(argv, 'i:t:p:', ['inputdir=', 'task=', 'partition='])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    input_data = False
    task = 2
    partition = None
    for opt, arg in opts:
        if opt in ('-i', '--inputdir'):
            inputdir = arg
            input_data = True
        elif opt in ('-t', '--task'):
            task = arg
        elif opt in ('-p', '--partition'):
            partition = arg
        else:
            print(usage)
            sys.exit()
    if not input_data:
        print(usage)
        sys.exit(2)
    print('Input directory is ', inputdir)
    if partition:
        processDataFiles(inputdir, task, [partition])
    else:
        processDataFiles(inputdir, task)

if __name__ == "__main__":
    main(sys.argv[1:])
