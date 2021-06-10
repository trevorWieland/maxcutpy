from maxcutpy import RandomMaxCut
import numpy as np

def test_basic_initialization():
    matrix = np.array([[0,1,1],[1,0,5],[1,5,0]])

    random_cut = RandomMaxCut(seed=12345, matrix=matrix)
    assert random_cut.batches_split == False
    assert random_cut.best_cut_vector is None
    assert random_cut.best_cut_score is None

def test_basic_functionality():
    matrix = np.array([[0,1,1],[1,0,5],[1,5,0]])

    random_cut = RandomMaxCut(seed=12345, matrix=matrix)

    cut_vectors = random_cut.generate_cut_vectors(1, 3, [0.5, 0.5, 0.5])
    assert (cut_vectors == np.array([[0, 0, 1]])).all()

    scores = random_cut.score_cut_vectors(cut_vectors)
    assert scores == [6]
