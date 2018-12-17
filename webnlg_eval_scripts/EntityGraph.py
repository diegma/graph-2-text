"""
Laura Perez

networkx documentation:
https://networkx.github.io/documentation/networkx-1.10/reference/classes.multidigraph.html#networkx.MultiDiGraph
"""

import networkx as nx


def entityGraph(t_subjects, t_objects, t_properties):
    DG = nx.MultiDiGraph()
    for i in range(0,len(t_properties)):
        DG.add_edge(t_subjects[i], t_objects[i], prop=t_properties[i])

    return DG

def getAllEdgeLabel(DG, u, v):
    ret=[]
    for i in range(0, len(DG[u][v])):
        ret.append(DG[u][v][i]['prop'])
    return ret

