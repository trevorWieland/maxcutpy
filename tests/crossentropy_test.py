from maxcutpy import CrossEntropyMaxCut
import numpy as np

def test_ce_initialization():
    matrix = np.array([[0,1,1],[1,0,5],[1,5,0]])

    ce_cut = CrossEntropyMaxCut(seed=12345, matrix=matrix)
    assert ce_cut.batches_split == False
    assert ce_cut.best_cut_vector is None
    assert ce_cut.best_cut_score is None

def test_ce_functionality():
    matrix = np.array([[0,1,1],[1,0,5],[1,5,0]])

    ce_cut = CrossEntropyMaxCut(seed=12345, matrix=matrix)
    assert ce_cut.batches_split == False
    assert ce_cut.best_cut_vector is None
    assert ce_cut.best_cut_score is None

    best_cut_vector = ce_cut.batch_split()

    assert (best_cut_vector == np.array([1, 0, 1])).all()
    assert ce_cut.batches_split == True
    assert (ce_cut.best_cut_vector == np.array([1, 0, 1])).all()
    assert ce_cut.best_cut_score == 6
