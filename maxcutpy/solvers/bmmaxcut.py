import numpy as np
import pandas as pd
import networkx as nx

from typing import Optional, List
from tqdm import tqdm

from maxcutpy.solvers.abstractmaxcut import AbstractMaxCut
from maxcutpy.rtr import RiemannianTrustRegion

class BureirMonteiroMaxCut(AbstractMaxCut):

    def __init__(self, seed: Optional[int] = 12345, matrix: Optional[np.array] = None, graph: Optional[nx.Graph] = None, dataframe: Optional[pd.DataFrame] = None, dim_p=None, **kwargs):

        super().__init__(seed=seed, matrix=matrix, graph=graph, dataframe=dataframe)

        if dim_p is None:
            dim_p = np.ceil(np.sqrt(2 * self.matrix.shape[0]))
        self.dim_p = int(dim_p)

        self._kwargs = kwargs


    def batch_split(self):
        if self.batches_split:
            return self.best_cut_vector
        else:
            rtr = RiemannianTrustRegion(self.matrix, self.dim_p, **self._kwargs)
            candidates = rtr.get_candidates()

            cut_vectors = np.array([self._getpartitions(vectors) for vectors in candidates])
            scores = np.array(self.score_cut_vectors(cut_vectors))

            self.best_cut_vector = cut_vectors[np.argmax(scores), :]
            self.best_cut_score = max(scores)
            self.batches_split = True
            return self.best_cut_vector

    def _getpartitions(self, vectors):
        random = self.generator.standard_normal(size=vectors.shape[1])
        random /= np.linalg.norm(random, 2)

        # Compute partition probabilities and round the cut.
        unclipped = np.sign(np.dot(vectors, random))
        clipped = np.clip(unclipped, 0, 1)
        return clipped


if __name__ == "__main__":
    matrix = np.array([[0,1,1],[1,0,5],[1,5,0]])
    maxcut = BureirMonteiroMaxCut(matrix=matrix)

    print(maxcut.batch_split())
