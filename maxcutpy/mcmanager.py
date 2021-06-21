from maxcutpy.solvers.cemaxcut import CrossEntropyMaxCut
from maxcutpy.solvers.bmmaxcut import BureirMonteiroMaxCut
from maxcutpy.solvers.randommaxcut import RandomMaxCut
from maxcutpy.solvers.abstractmaxcut import AbstractMaxCut

from typing import List, Optional
from sklearn.preprocessing import LabelEncoder

import math

import numpy as np
import pandas as pd
import networkx as nx

class MaxCutManager:

    def __init__(self, solver_name: str, num_batches: int, solver_args: Optional[dict] = {}, seed: Optional[int] = 12345, matrix: Optional[np.array] = None, graph: Optional[nx.Graph] = None, dataframe: Optional[pd.DataFrame] = None):

        self.seed = seed
        self.generator = np.random.default_rng(self.seed)
        self.le = None

        self.solver_args = solver_args
        self.solver_name = solver_name
        self.num_batches = num_batches

        if matrix is not None:
            self.origin_matrix = matrix
        elif graph is not None:
            UG = self._transform_D_2_U(graph)
            self.origin_matrix = nx.to_numpy_matrix(UG)
        elif dataframe is not None:
            if dataframe.shape[1] != 3:
                raise RuntimeError("Dataframe should be in long format: [NODEA, NODEB, EDGE_WEIGHT]!")

            dataframe.columns = ["NODEA", "NODEB", "EDGE_WEIGHT"]
            UG = self._transform_long_2_nx(dataframe)
            self.origin_matrix = nx.to_numpy_matrix(UG)
        else:
            raise RuntimeError("At least one of `matrix`, `graph`, or `dataframe` must not be None!")

        self.partition_vector = None

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

    def split_all(self):
        if self.partition_vector is None:
            binary_style = self._split_line(self.origin_matrix, 0)

            integer_style = [int(bs, 2) for bs in binary_style]

            self.partition_vector = integer_style

        return self.partition_vector

    def _split_line(self, matrix_part, batch_depth):

        if self.solver_name == "RandomMaxCut":
            solver = RandomMaxCut(seed=self.seed, matrix=matrix_part, **self.solver_args)
        elif self.solver_name == "CrossEntropyMaxCut":
            solver = CrossEntropyMaxCut(seed=self.seed, matrix=matrix_part, **self.solver_args)
        elif self.solver_name == "BureirMonteiroMaxCut":
            solver = BureirMonteiroMaxCut(seed=self.seed, matrix=matrix_part, **self.solver_args)
        else:
            raise RuntimeError(f"Invalid solver {self.solver_name}!")

        cut_vector = solver.batch_split()

        if (batch_depth + 1 >= int(math.log(self.num_batches, 2.0))):
            #Base condition
            #Return partition
            return [str(v) for v in cut_vector]
        else:
            #Split matrix_part into two submatrices based on partition
            batch_1, batch_2 = solver._split_vec_2_parts(cut_vector)

            matrix_part_1 = matrix_part[:, batch_1][batch_1, :]
            matrix_part_2 = matrix_part[:, batch_2][batch_2, :]

            part_1, part_2 = self._split_line(matrix_part_1, batch_depth+1), self._split_line(matrix_part_2, batch_depth+1)

            part_1 = ["0" + v for v in part_1]
            part_2 = ["1" + v for v in part_2]

            #Recombine part_1 and part_2 into the same state
            recombined_partition = []
            p1_c = 0
            p2_c = 0
            for i in range(matrix_part.shape[0]):
                if i in batch_1:
                    recombined_partition.append(part_1[p1_c])
                    p1_c += 1
                else:
                    recombined_partition.append(part_2[p2_c])
                    p2_c += 1

            return recombined_partition

    def label_batches(self):
        if self.le is None:
            raise RuntimeError(f"This function is only usable when this object was created with a dataframe")
        if self.partition_vector is None:
            raise RuntimeError(f"This function is only usable after calling `split_all`")

        labeled_df = pd.DataFrame()
        labeled_df["NODE"] = self.le.inverse_transform(list(range(len(self.partition_vector))))
        labeled_df["BATCH"] = self.partition_vector

        return labeled_df

    def score_batches(self):
        if self.partition_vector is None:
            raise RuntimeError(f"This function is only usable after calling `split_all`")



if __name__ == "__main__":
    dataframe = pd.DataFrame()
    dataframe["NODE1"] = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    dataframe["NODE2"] = ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "A"]
    dataframe["Weight"] = list(range(12))

    manager = MaxCutManager("CrossEntropyMaxCut", num_batches=4, dataframe=dataframe)

    manager.split_all()

    print(manager.label_batches())
