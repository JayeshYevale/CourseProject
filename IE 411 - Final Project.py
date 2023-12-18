from PIL import Image
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import gurobipy as gp
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

im = Image.open('tower.png').convert('L') #convert to grayscale
h = im.height
w = im.width

tic = time.time()

#grayscale values; list of size h*w , where each values is integer in range [0,255]
intensity = list(im.getdata()) 

def get_weight(n1,n2):
    if n2 == t:
        diff = 0
    else:
        diff = intensity[n1-1] - intensity[n2-1]
    return np.abs(diff)

#define indexes
def idx(i,j):
    return w*(i-1)+j

#specify source node
s = 50

for q in range(25):

    #list of coordinates
    coords = [(i,j) for i in range(1,h+1) for j in range(1,w+1)]

    nodes = [idx(i,j) for i,j in coords]

    #add destination (t) to nodes
    t = h*w+1
    nodes.append(t)

    #define neighbors
    ngbrs = {idx(i,j): [idx(i2,j2) for (i2,j2) in [(i+1,j-1),(i+1,j),(i+1,j+1)] 
                          if 1<= i2 <= h and 1<= j2 <= w] for (i,j) in coords}

    #add neighbors for destination nodes
    for (i,j) in coords:
        if i ==h:
            ngbrs[idx(i,j)].append(t) 

    #define edges
    edges = [(n1,n2) for n1 in nodes[:-1] for n2 in ngbrs[n1]]
    
    #define incident matrix
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    A = nx.incidence_matrix(G, oriented= True)
    
    #intialization
    cap = np.zeros(G.number_of_edges())
    r = np.zeros(G.number_of_edges())
    for i in range(G.number_of_edges()):
        cap[i] = get_weight(edges[i][0],edges[i][1])
    
    #define the primal problem
    m = gp.Model("Primal Image Seaming")
    m.setParam('OutputFlag', 0)
    m.Params.LogToConsole = 0
    m.Params.Method = 0
    f = m.addMVar(G.number_of_edges(), vtype=GRB.CONTINUOUS,lb=0)
    b = np.zeros(G.number_of_nodes())
    b[s-1] = -1
    b[-1] = 1
    m.addConstr(A@f==b)    
    obj= cap@f
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    print("\nIteration: ",q+1,"\nPrimal Objective: ",m.getObjective().getValue())
    
    if q == 0:
        toc = time.time()
    
    #remove the seam nodes
    flows = m.getAttr("X", m.getVars())
    to_remove = [] # get the indices of nodes (pixels) to be removed
    for i in range(G.number_of_edges()):
        if flows[i] >= 0.9:
            to_remove.append(edges[i][0]-1) #0 indexed
    intensity = [I for idx,I in enumerate(intensity) if idx not in to_remove]
    w = w - 1
    
    #DUAL PROBLEM
    mm = gp.Model("Dual Image Seaming")
    mm.Params.LogToConsole = 0
    mm.Params.Method = 0
    z = mm.addMVar(shape=G.number_of_nodes(), vtype=GRB.CONTINUOUS, lb=0, name="z")
    obj = b@z

    mm.addConstr(cap>=A.transpose()@z)
    mm.setObjective(obj, GRB.MAXIMIZE)
    mm.optimize()
    print("Dual objective:   ", mm.getObjective().getValue())
    dual_vars = mm.getAttr("X", mm.getVars())

    #Check complementary slackness conditions
    cuts = dual_vars[:G.number_of_edges()]
    y = dual_vars[G.number_of_edges():]
    cmaxflows = A@flows - b
    print('Complementary slackness: ', cmaxflows@cuts)
    Aty_plus_z_min_r = cap - A.transpose()@dual_vars
    print('Complementary slackness: ', Aty_plus_z_min_r@flows)

# resize final intensities and save as an image
arr = np.reshape(intensity,(h,w)).astype('uint8')
final_image = Image.fromarray(arr)
final_image.save('final_image.png')


print ('elapsed ', toc - tic)