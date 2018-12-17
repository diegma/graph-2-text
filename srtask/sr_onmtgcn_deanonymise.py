# -*- coding: utf-8 -*-

import getopt, sys, os, codecs
import json

ANONYMIZATION_DICT = "{}-sr11-{}--anonym.json"

def writeDeanonymPredictions(fpreds, deanonymDict, deanonymFileName):
    outDeanonymFile = codecs.open(deanonymFileName,'w', encoding='utf8')

    sentID = 1
    for line in fpreds:
        line = line.strip()
        sentK = str(sentID)
        sentDict = deanonymDict[sentK]
        # deanSent = " ".join([sentDict[w].replace("_"," ") if w in sentDict.keys() else w for w in line.split() ])
        deanSent = " ".join([sentDict[w].replace("_"," ") if w in sentDict.keys() else w for w in line.split() ])
        deanSent=line.strip()
        for ent in sentDict:
            deanSent = deanSent.replace(ent, sentDict[ent].replace('_', ' '))

        outDeanonymFile.write(deanSent + "\n")
        sentID +=1

def processDataFile(inputdir, dataFile, partition, task):
    k = dataFile.rfind(".")
    deanonymFileName = dataFile[:k] + "_deanonym.txt"
    with codecs.open(dataFile,'r', encoding='utf8') as fpreds:
        with codecs.open(os.path.join(inputdir,
            ANONYMIZATION_DICT.format(partition, task)),'r', encoding='utf8') as fanonym:

            writeDeanonymPredictions(fpreds, json.load(fanonym), deanonymFileName)



def main(argv):
    usage = 'usage:\npython sr_onmtgcn_deanonymise.py -i <dir> -f <prediction-file> -p <partition> -t <task>' \
            '\n '
    try:
        opts, args = getopt.getopt(argv, 'i:f:p:t:', ['inputdir=','prediction-file=', 'partition=', 'task='])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    givenInputData = False
    givenInputDir = False
    task = 'deep'
    partition = 'devel'

    for opt, arg in opts:
        if opt in ('-i', '--inputdir'):
            inputdir = arg
            givenInputDir = True
        elif opt in ('-f', '--prediction-file'):
            dataFile = arg
            givenInputData = True
        elif opt in ('-t', '--task'):
            task = arg
        elif opt in ('-p', '--partition'):
            partition = arg
        else:
            print(usage)
            sys.exit()
    if not givenInputData or not givenInputDir:
        print(usage)
        sys.exit(2)
    # print('Input directory is ', inputdir)
    processDataFile(inputdir, dataFile, partition, task)

if __name__ == "__main__":
    main(sys.argv[1:])