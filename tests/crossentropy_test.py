from maxcutpy import CrossEntropyMaxCut
import numpy as np
import pandas as pd

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

def test_ce_fromdataframe():
    df = pd.DataFrame()

    df["N1"] = ["A", "A", "B"]
    df["N2"] = ["B", "C", "C"]
    df["E"] = [1, 1, 2]

    ce_cut = CrossEntropyMaxCut(dataframe=df)

    best_cut_vector = ce_cut.batch_split()

    assert (best_cut_vector == np.array([1,0,1])).all()

    bcv_indicator = ce_cut.get_best_cut_vector()
    bcv_index = ce_cut.get_best_cut_vector(format="index")
    bcv_labels = ce_cut.get_best_cut_vector(format="labels")

    assert (bcv_indicator == np.array([1,0,1])).all()

    assert (bcv_index[0] == [1])
    assert (bcv_index[1] == [0, 2])

    assert (bcv_labels[0] == ["B"])
    assert all(bcv_labels[1] == ["A", "C"])
