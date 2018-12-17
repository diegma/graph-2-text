# -*- coding: utf-8 -*-

import getopt, re, datetime, os, codecs
import sys
from operator import itemgetter
import json
import networkx as nx


def readGraphFile(filename):
    ret = []
    with codecs.open(filename, 'r', encoding='utf8') as f:
        for line in f.readlines():
            ret.append(line.strip().split())
    return ret


def format2linearinput(inputdir, dataset, task, option):

    outlinear = codecs.open(
        inputdir + dataset + '-sr11-{}-'.format(task) + option + '-src-linear.txt', 'w', encoding='utf8')

    labels = readGraphFile(inputdir + dataset + '-sr11-{}-'.format(task) + option + '-src-labels.txt')
    node1 = readGraphFile(inputdir + dataset + '-sr11-{}-'.format(task) + option + '-src-node1.txt')
    node2 = readGraphFile(inputdir + dataset + '-sr11-{}-'.format(task) + option + '-src-node2.txt')
    nodes = readGraphFile(inputdir + dataset + '-sr11-{}-'.format(task) + option + '-src-nodes.txt')
    print("* examples: {}".format(len(labels)))
    if labels and node1 and node2 and nodes:
        for exID in range(len(labels)):
            print("sentID={}".format(exID+1))
            #G = nx.MultiDiGraph()
            G = nx.DiGraph()
            for n in range(len(nodes[exID])):
                G.add_node(n)
            for label, node, head in zip(labels[exID], node1[exID], node2[exID]):
                G.add_edge(nodes[exID][int(head)], nodes[exID][int(node)], l=label)
                #if node=='0' and head=='0' and not len(labels)==1:
                #    continue
                G.add_edge(int(head), int(node), l=label)

            linearised = []
            edgeTuples = list(nx.dfs_labeled_edges(G,0))
            prevTokens = []
            for u, v, d in edgeTuples:
                if u in prevTokens and v in prevTokens and u==v and u==0:
                    continue
                if d['dir'] == 'reverse' :
                    continue
                # we want to pass through d['dir'] == 'nontree' and 'forward'
                # nontree is when there is a cycle and the has already been through
                # target node subtree

                if not u in prevTokens:
                    linearised.append(nodes[exID][u])
                    prevTokens.append(u)
                #do not control whether dep node was already printed/visited,
                #just print it again after the label
                linearised.append(G[u][v]['l'])
                linearised.append(nodes[exID][v])
                prevTokens.append(v)

            outlinear.write(" ".join(linearised)+"\n")
    else:
        print("Some file was not found or did not contained data.")

    outlinear.close()
    print("Linear input saved.")

def processDataFiles(inputdir, task, partitions = ['train', 'devel'] ):

    options = ['anonym']
    for dataset in partitions:
        for op in options:
            print("Processing task= {} partition= {}".format(task, dataset))
            format2linearinput(inputdir, dataset, task, op)

def main(argv):
    usage = 'usage:\npython sr11_linear_input.py -i <dir> [-t TASK]' \
            '\n TASK is either deep or shallow'
    try:
        opts, args = getopt.getopt(argv, 'i:t:', ['inputdir=', 'task'])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    input_data = False
    task = 2
    for opt, arg in opts:
        if opt in ('-i', '--inputdir'):
            inputdir = arg
            input_data = True
        elif opt in ('-t', '--task'):
            task = arg
        else:
            print(usage)
            sys.exit()
    if not input_data:
        print(usage)
        sys.exit(2)
    print('Input directory is ', inputdir)
    processDataFiles(inputdir, task)

if __name__ == "__main__":
    main(sys.argv[1:])
