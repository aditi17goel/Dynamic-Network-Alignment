import networkx as nx
import pandas as pd
import csv
import glob
import numpy
from matplotlib import pyplot
from math import log2
import seaborn as sn


def kl_divergence(p, q):
	kl_divergence = 0
	for i in range(min(len(p),len(q))):
		kl_divergence += p[i] * log2(p[i]/q[i])
	return kl_divergence

def js_divergence(p, q):
  p = [float(x) for x in p]
  q = [float(x) for x in q]
  m = []
  for i in range(min(len(p),len(q))):
    m.append((p[i]+q[i])/2)
  return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def get_df(n):
    df = []
    path = 'coauthor'
    all_files = glob.glob(path + "/*.txt")
    for i in range(n):
        temp = pd.read_csv(all_files[i], delimiter="\t", header = None, on_bad_lines='skip')
        df.append(temp)
    return df

def make_graph(df):
  G =[]
  for i in range(len(df)):
    graph = nx.Graph()
    for index, row in df[i].iterrows():
      if(numpy.issubdtype(type(row[0]), int) and numpy.issubdtype(type(row[1]), int)):
        graph.add_edge(row[0], row[1])
    G.append(graph)
  return G

def get_degree_distribution(G): 
  degree_distribution = []
  for i in range(len(G)):
    dd = [G[i].degree(n) for n in G[i].nodes()]
    dd[:] = [x for x in dd if x > 5]
    if(len(dd)):
      degree_distribution.append(dd)

  for i in range(len(degree_distribution)):
    Sum = sum(degree_distribution[i])
    for idx, degree in enumerate(degree_distribution[i]):
      degree_distribution[i][idx] = degree/Sum
  return degree_distribution


def heatmap_kl_divergence(degree_distribution):
  heatmap = []

  for i in range(0, len(degree_distribution)):
    p = degree_distribution[i]
    row = []
    for j in range(0,len(degree_distribution)):
      q = degree_distribution[j]
      row.append(kl_divergence(p,q))
    heatmap.append(row)

  hm = sn.heatmap(data = heatmap)

  pyplot.show()

def heatmap_js_divergence(degree_distribution):
  heatmap = []

  for i in range(0, len(degree_distribution)):
    p = degree_distribution[i]
    row = []
    for j in range(0,len(degree_distribution)):
      q = degree_distribution[j]
      row.append(js_divergence(p,q))
    heatmap.append(row)

  hm = sn.heatmap(data = heatmap)

  pyplot.show()

if __name__ == "__main__":
  num = int(input ("Enter number of snapshots:"))
  df = get_df(num)
  G = make_graph(df)
  degree_distribution = get_degree_distribution(G)
  heatmap_kl_divergence(degree_distribution)
  heatmap_js_divergence(degree_distribution)