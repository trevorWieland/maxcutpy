# maxcutpy
 A Python Implementation of Graph Max Cut Solutions

## Problem Statement
The goal of this script is to provide simple python libraries for solving the [Maximum cut problem](https://en.wikipedia.org/wiki/Maximum_cut).

## Current Solvers

### Universal
The Structure of all Solvers is the same, from an API point of view.
This allows for quick testing of multiple different solvers, to find the best one.

In general, the flow will look like this:

    from maxcutpy import RandomMaxCut
    import numpy as np

    matrix = np.array([[0,1,1],[1,0,5],[1,5,0]])
    random_max_cut = RandomMaxCut(matrix=matrix, seed=12345)

    best_cut_vector = random_cut.batch_split()

`best_cut_vector` will be a numpy array of shape `(1,n)` where n is the number of rows in the input matrix. Each number inside of `best_cut_vector` will be an integer 0 or 1, depending on if it belongs to the 0th Slice or the 1st Slice.

You can then generally check the score of this `best_cut_vector` by using the same object again:

    best_cut_score = random_cut.best_cut_score

At this time, a single Solver Object is Single-Use only, meaning it gets created for a specific matrix, and provides a batch split for this matrix only. Once the result is calculated, it will be stored, and repeated calls to `batch_split()` will only return the cached result.

For now, the matrix must be in the form of a numpy adjacency matrix, but support will be added soon for networkx Graph objects, as well as helper functions to transform between the two.

### RandomMaxCut
This solve provides a method to compare against, as all this Solver does is select a random set of vertices to cut, giving each vertex a 50% chance of occurring in a different slice of the graph.

    from maxcutpy import RandomMaxCut
    import numpy as np

    matrix = np.array([[0,1,1],[1,0,5],[1,5,0]])
    random_max_cut = RandomMaxCut(matrix=matrix, seed=12345)

    best_cut_vector = random_cut.batch_split()

`best_cut_vector` will be a numpy array of shape `(1,n)` where each entry is a random integer 0 or 1, with 50% probability for either event. This means this could lead to the case where no cuts are made at all, yielding a score of 0.
