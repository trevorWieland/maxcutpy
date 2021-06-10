import numpy as np
import pandas as pd
import networkx as nx

from typing import Optional, List

from abc import ABC, abstractmethod


class AbstractMaxCut(ABC):

    def __init__(self, seed: Optional[int] = 12345, matrix: Optional[np.array] = None, graph: Optional[nx.Graph] = None):
        self.seed = seed
        self.generator = np.random.default_rng(self.seed)

        if matrix is not None:
            self.matrix = matrix
        elif graph is not None:
            self.matrix = nx.to_numpy_matrix(graph)
        else:
            self.matrix = np.array([[0, 1, 1], [1, 0, 2], [1, 2, 0]])

        self.batches_split = False
        self.best_cut_vector = None
        self.best_cut_score = None

    def score_cut_vectors(self, cut_vectors: np.array) -> List[np.float64]:
        """A function to calculate the score of a set of sample cut vectors

        Args:
            cut_vectors:    A numpy matrix of the cut vectors.
                Must be of shape (N, n) where n is the number of vertices and
                N is the number of sample cut vectors

        Returns:
            A List of np.float64 scores, N in length, which is the score of each
            cut vector

        """

        ##Needs to be replaced by a better, vectorized (or parallel) method
        scores = []
        for i in range(cut_vectors.shape[0]):
            cut_0 = np.argwhere(cut_vectors[i,:] == 0).reshape((1,-1))[0]
            cut_1 = np.argwhere(cut_vectors[i,:] == 1).reshape((1,-1))[0]

            pairs = np.array(np.meshgrid(cut_0, cut_1)).T.reshape(-1, 2)

            score_i = sum([self.matrix[x, y] for x, y in pairs])
            scores.append(score_i)

        return scores

    def generate_cut_vectors(self, N: int, n: int, p: List[np.float64]) -> np.array:
        """A function to generate a sample set of cut vectors

        Args:
            N: The integer number of samples to generate
            n: The integer number of vertices in each cut
            p: A list of float probabilities used in generating the cut vectors

        Returns:
            A numpy matrix of shape (N, n) where each row is a cut vector

        """
        assert len(p) == n

        cut_vectors = self.generator.binomial(n=1, p=p, size=(N, n))

        return cut_vectors

    @abstractmethod
    def batch_split(self) -> np.array:
        """A function to split the batches.

        This function is an abstract method in the class AbstractMaxCut.
        Should rely on internal attributes, and take no input parameters.

        Should return a single cut vector, which is a numpy array of shape (1, n), where
        n is the number of vertices in the matrix
        """
        pass
