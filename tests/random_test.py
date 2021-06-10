from maxcutpy import RandomMaxCut
import numpy as np

def test_random_initialization():
    matrix = np.array([[0,1,1],[1,0,5],[1,5,0]])

    random_cut = RandomMaxCut(seed=12345, matrix=matrix)
    assert random_cut.batches_split == False
    assert random_cut.best_cut_vector is None
    assert random_cut.best_cut_score is None

def test_random_functionality():
    matrix = np.array([[0,1,1],[1,0,5],[1,5,0]])

    random_cut = RandomMaxCut(seed=12345, matrix=matrix)
    assert random_cut.batches_split == False
    assert random_cut.best_cut_vector is None
    assert random_cut.best_cut_score is None

    best_cut_vector = random_cut.batch_split()

    assert (best_cut_vector == np.array([0, 0, 1])).all()
    assert random_cut.batches_split == True
    assert (random_cut.best_cut_vector == np.array([0, 0, 1])).all()
    assert random_cut.best_cut_score == 6
