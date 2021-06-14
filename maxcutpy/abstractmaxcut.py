import numpy as np
import pandas as pd
import networkx as nx

from typing import Optional, List

from abc import ABC, abstractmethod

from sklearn.preprocessing import LabelEncoder


class AbstractMaxCut(ABC):

    def __init__(self, seed: Optional[int] = 12345, matrix: Optional[np.array] = None, graph: Optional[nx.Graph] = None, dataframe: Optional[pd.DataFrame] = None):
        """Creates a new AbstractMaxCut object. Contains initialization for any MaxCut object.

        Sets up the randomness generator using the input seed, loads the input data, and sets up any other parameters.
        At least one of matrix, graph, or dataframe must be given.

        Args:
            seed: An optional integer, to use as the seed for random number generation. Default is 12345
            matrix: An optional numpy array of data. Should be a square matrix, which represents the adjacency matrix of a graph.
            graph: An optional networkx graph object. Will be transformed into a Undirected Graph by averaging weights.
            dataframe: An optional pandas dataframe. Should be in long format with 3 columns, representing triples of (NODEA, NODEB, EDGE_WEIGHT).

        """
        self.seed = seed
        self.generator = np.random.default_rng(self.seed)
        self.le = None

        if matrix is not None:
            self.matrix = matrix
        elif graph is not None:
            UG = self._transform_D_2_U(graph)
            self.matrix = nx.to_numpy_matrix(UG)
        elif dataframe is not None:
            if dataframe.shape[1] != 3:
                raise RuntimeError("Dataframe should be in long format: [NODEA, NODEB, EDGE_WEIGHT]!")

            dataframe.columns = ["NODEA", "NODEB", "EDGE_WEIGHT"]
            UG = self._transform_long_2_nx(dataframe)
            self.matrix = nx.to_numpy_matrix(UG)

        else:
            raise RuntimeError("At least one of `matrix`, `graph`, or `dataframe` must not be None!")

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

    def _transform_long_2_nx(self, dataframe):

        nodes_left = pd.concat([dataframe["NODEA"], dataframe["NODEB"]]).unique()

        self_pointing = pd.DataFrame(columns=["NODEA", "NODEB", "EDGE_WEIGHT"])
        self_pointing["NODEA"] = nodes_left
        self_pointing["NODEB"] = nodes_left
        self_pointing["EDGE_WEIGHT"] = 0

        partial_df = pd.concat([dataframe, self_pointing])

        self.le = LabelEncoder()
        self.le.fit(list(set(partial_df["NODEA"].values).union(set(partial_df["NODEB"].values))))
        partial_df.loc[:, "NODEA"] = self.le.transform(partial_df["NODEA"])
        partial_df.loc[:, "NODEB"] = self.le.transform(partial_df["NODEB"])

        G = nx.DiGraph()
        nw_view = list(partial_df.itertuples(index=False, name=None))
        G.add_weighted_edges_from(nw_view)

        return self._transform_D_2_U(G)

    def _transform_D_2_U(self, G):
        UG = G.to_undirected()
        for node in G:
            for ngbr in nx.neighbors(G, node):
                if node in nx.neighbors(G, ngbr):
                    UG.edges[node, ngbr]['weight'] = (
                        .5*G.edges[node, ngbr]['weight'] + .5*G.edges[ngbr, node]['weight']
                    )

        return UG

    def get_best_cut_vector(self, format: Optional[str] ="Indicator"):
        if self.best_cut_vector is None:
            raise RuntimeError("Best Cut Vector has not yet been found! Run batch_split to find the best cut vector!")

        l_format = format.lower()
        if l_format == "indicator":
            return self.best_cut_vector
        elif l_format == "index":
            return self._split_vec_2_parts(self.best_cut_vector)
        elif l_format == "labels":
            if self.le is None:
                raise RuntimeError(f"`labels` mode is only usable when this object was created with a dataframe")

            p0, p1 = self._split_vec_2_parts(self.best_cut_vector)
            p0_labels = self.le.inverse_transform(p0)
            p1_labels = self.le.inverse_transform(p1)

            return p0_labels, p1_labels
        else:
            raise RuntimeError(f"Unknown Best Cut Vector format: {format}")

    def _split_vec_2_parts(self, vector):
        part_0, part_1 = [], []

        for e, v in enumerate(vector):
            if v == 0:
                part_0.append(e)
            else:
                part_1.append(e)

        return part_0, part_1
