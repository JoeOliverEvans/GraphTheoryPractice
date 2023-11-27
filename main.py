import math
import copy
import numpy as np


class Graph:
    def __init__(self, vert, edg, directed=False):
        self.vertices = vert
        self.edges = edg
        self.directed = directed
        self.dictionary = self.constructDictionaryList()
        self.adjacency = self.constructAdjacency()
        #self.bridges = self.detectBridges()

    def constructDictionaryList(self):
        dictionary = []
        for vertex in self.vertices:
            if vertex not in dictionary:
                dictionary.append(vertex)
        return dictionary

    def constructAdjacency(self):
        """
        Calculate the matrix
        :return A: adjacency matrix
        """
        adjacency = np.zeros((len(self.vertices), len(self.vertices)))
        for edge in self.edges:
            i = self.dictionary.index(edge[0])
            j = self.dictionary.index(edge[1])
            adjacency[i, j] = 1
            if not self.directed:
                adjacency[j, i] = 1
        return np.array(adjacency)

    def PageRank(self, maxiter=5000, tolerance=1 * 10 ** (-6)):
        """
        Implementation of the pagerank algorithm
        :return p: ranking on vertexes
        """
        if np.max(self.adjacency) == 0:
            return self.adjacency

        norm_adjacency = np.zeros(self.adjacency.shape)
        for row in range(self.adjacency.shape[0]):
            if np.sum(self.adjacency[row, :]) == 0:
                norm_adjacency[row, :] = np.zeros(norm_adjacency[row, :].shape)
            else:
                norm_adjacency[row, :] = self.adjacency[row, :] / np.sum(self.adjacency[row, :])

        e = 1 / len(self.dictionary) * np.ones(len(self.dictionary))

        dangling_nodes = []
        for index, value in enumerate(np.sum(self.adjacency, axis=1)):
            if value == 0:
                dangling_nodes.append(self.dictionary[index])

        for dangler in dangling_nodes:
            norm_adjacency[self.dictionary.index(dangler), :] = e

        p0 = np.zeros(len(self.vertices))
        p = p0

        d = 0.85  # len(self.dictionary)

        for i in range(maxiter):
            p_last = p
            p = d * (norm_adjacency.T @ p_last) + (1 - d) * e
            if np.linalg.norm(p - p_last) < tolerance:
                break
        return p

    def connectedComponents(self):
        """
        Returns the number of connected components and the list of visited connected components
        :param self:
        :return k, components[]:
        """
        k = 0
        components = []
        visited = []
        for v in self.vertices:
            if v not in visited:
                nodes = []
                visited, nodes = self.depthFirstSearch(v, visited, nodes)
                k += 1
                components.append(nodes)
        return k, components

    def depthFirstSearch(self, v, visited, nodes):
        """
        Performs a depth first search
        :param v:
        :param visited:
        :param nodes:
        :return visited, nodes:
        """
        visited.append(v)
        nodes.append(v)
        A = self.adjacency
        adjNodes = [v]
        for i in range(len(A[:, self.dictionary.index(v)])):
            if A[i, self.dictionary.index(v)] == 1:
                adjNodes.append(self.dictionary[i])
        for node in adjNodes:
            if node not in visited:
                visited, nodes = self.depthFirstSearch(node, visited, nodes)
        return visited, nodes

    def edgeRemoval(self, e, set_values=True):
        """
        Removes an edge
        :param set_values: whether to alter the graph
        :param e:
        :return self:
        """
        A = copy.copy(self.adjacency)
        A[self.dictionary.index(e[0]), self.dictionary.index(e[1])] = 0
        if not self.directed:
            A[self.dictionary.index(e[1]), self.dictionary.index(e[0])] = 0
        if set_values:
            self.vertices, self.edges = verticesAndEdgesFromAdjacency(A)
            self.adjacency = A
        return Graph(*verticesAndEdgesFromAdjacency(A))

    def nodeRemoval(self, v, set_values=True):
        """
        Removes a node and connections
        :param set_values: whether to alter the graph
        :param v:
        :return self:
        """
        A = copy.copy(self.adjacency)
        A = np.delete(A, self.dictionary.index(v), 0)
        A = np.delete(A, self.dictionary.index(v), 1)
        if set_values:
            self.vertices, self.edges = verticesAndEdgesFromAdjacency(A)
            self.adjacency = A
            self.dictionary = self.constructDictionaryList()
        return Graph(*verticesAndEdgesFromAdjacency(A))
    
    def detectBridges(self):
        """
        Find bridges in the graph
        :return bridges: 
        """
        bridges = []
        A = self.adjacency
        for i in range((A.shape[0])):
            for j in range((A.shape[0])):
                valid = False
                if A[i, j] == 1 and i != j:
                    if not self.directed:
                        if i < j:
                            valid = True
                    else:
                        valid = True
                    if valid:
                        prunedgraph = copy.copy(self)
                        prunedgraph = prunedgraph.edgeRemoval([self.dictionary[i], self.dictionary[j]], set_values=False)
                        if prunedgraph.connectedComponents()[0] > self.connectedComponents()[0]:
                            bridges.append([self.dictionary[i], self.dictionary[j]])
                        del prunedgraph
        return bridges
        

def wattsStrogatz(N, K, p):
    """
    Generate a random graph using the Watts-Strogatz method
    :param N: Graph order
    :param K: Mean degree (even integer)
    :param p: rewiring probability
    :return A: random graph
    """
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if j < i:
                if (np.abs(i - j) % (N - 1 - K / 2)) > 0 and (np.abs(i-j) % (N-1-K/2)) <= K/2:
                    A[i, j], A[j, i] = 1, 1
    for i in range(N):
        for j in range(N):
            if j < i:
                if A[i, j] == 1 and np.random.rand() < p:
                    k = int(N * np.random.rand())
                    while k == i or A[i, k] == 1:
                        k = int(N * np.random.rand())
                    A[i, j], A[j, i] = 0, 0
                    A[i, k], A[k, i] = 1, 1
    return A


def verticesAndEdgesFromAdjacency(A):
    """
    Finds edges and vertices from the adjacency matrix
    :param A:
    :return vertices, edges:
    """
    vertices = np.arange(A.shape[0])
    edges = np.argwhere(A == 1)
    return vertices, edges


newGraph = Graph(['A', 'B', 'C', 'D'], [['B', 'A'], ['B', 'C'], ['C', 'A'], ['D', 'A'], ['D', 'B'], ['D', 'C']], True)
newGraph2 = Graph(['A', 'B', 'C', 'D', 'E'], [['B', 'A'], ['A', 'C'], ['D', 'E'], ['A', 'D']], False)
newGraph3 = Graph(['A', 'B', 'C', 'D', 'E'], [['B', 'A'], ['A', 'C'], ['D', 'E']], False)

graph2 = Graph(*verticesAndEdgesFromAdjacency(wattsStrogatz(4, 2, 0.1)), False)
print(newGraph2.connectedComponents())
print(newGraph3.connectedComponents())
print(newGraph3.detectBridges())
print(graph2.adjacency)
