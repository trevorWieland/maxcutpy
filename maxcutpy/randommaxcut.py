import numpy as np
import pandas as pd
import networkx as nx

from typing import Optional, List
from tqdm import tqdm
from maxcutpy.abstractmaxcut import AbstractMaxCut

class RandomMaxCut(AbstractMaxCut):

    def batch_split(self):
        if self.batches_split:
            return self.best_cut_vector
        else:
            p = 0.5 * np.ones((self.matrix.shape[0]))
            cut_vec = self.generate_cut_vectors(1, self.matrix.shape[0], p)
            self.best_cut_score = self.score_cut_vectors(cut_vec)[0]

            self.best_cut_vector = cut_vec[0]

            self.batches_split = True
            return self.best_cut_vector
