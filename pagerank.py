import numpy as np


class Graph:
    def __init__(self, vert, edg, directed=False):
        self.vertices = vert
        self.edges = edg
        self.directed = directed
        self.dictionary = self.constructDictionaryList()
        self.adjacency = self.constructAdjacency()

    def constructDictionaryList(self):
        dictionary = []
        for vertex in self.vertices:
            if vertex not in dictionary:
                dictionary.append(vertex)
        return dictionary

    def constructAdjacency(self):
        """
        Calculate the matrix
        :return: adjacency matrix
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
        :return: ranking on vertexes
        """
        if np.max(self.adjacency) == 0:
            return self.adjacency

        norm_adjacency = np.zeros(self.adjacency.shape)
        for row in range(self.adjacency.shape[0]):
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


newGraph = Graph(['A', 'B', 'C', 'D'], [['B', 'A'], ['B', 'C'], ['C', 'A'], ['D', 'A'], ['D', 'B'], ['D', 'C']], True)

print(newGraph.adjacency)
print(newGraph.PageRank())
