import numpy as np
import pandas as pd
import networkx as nx

from typing import Optional, List
from tqdm import tqdm
from maxcutpy.abstractmaxcut import AbstractMaxCut

class CrossEntropyMaxCut(AbstractMaxCut):

    def __init__(self, seed: Optional[int] = 12345, matrix: Optional[np.array] = None, p_cutoff: Optional[np.float64] = 0.1, N: Optional[int] = 1000, convergence_threshold: Optional[np.float64] = 0.0001, max_iters: Optional[int] = 1000):

        super().__init__(seed=seed, matrix=matrix)

        self.p_cutoff = p_cutoff
        self.N = N
        self.convergence_threshold = convergence_threshold
        self.max_iters = max_iters


    def batch_split(self):
        if self.batches_split:
            return self.best_cut_vector
        else:
            p = 0.5 * np.ones((self.matrix.shape[0]))
            t = 1

            for t in tqdm(range(self.max_iters)):
                cut_vectors = self.generate_cut_vectors(self.N, self.matrix.shape[0], p)
                scores = np.array(self.score_cut_vectors(cut_vectors))

                gamma = np.quantile(scores, 1 - self.p_cutoff, interpolation="nearest")

                thresh_scores = (scores >= gamma).astype(int)
                total_acheived = thresh_scores.sum()

                new_p = np.sum((cut_vectors * thresh_scores.reshape(-1, 1)), axis=0) / total_acheived

                if np.abs(new_p - p).mean() < self.convergence_threshold:
                    print("Results Converged!")
                    break

                p = new_p
                t += 1
            if t == self.max_iters - 1:
                print("Results failed to converge!")
                print(f"Current score is {max(scores)}")
            else:
                print(f"Converged at {t} iterations!")
                print(f"Found best score of {max(scores)}")

            self.best_cut_vector = cut_vectors[np.argmax(scores), :]
            self.best_cut_score = max(scores)
            self.batches_split = True
            return self.best_cut_vector


if __name__ == "__main__":
    maxcut = CEMaxCut(N=1000)

    print(maxcut.batch_split())
