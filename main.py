import numpy as np
import random
from collections import Counter
from itertools import islice
import pandas as pd
from numpy import dtype

# 1a

def levenshtein(seq1, seq2):
    size_1 = len(seq1) + 1
    size_2 = len(seq2) + 1
    matrix = np.zeros((size_1, size_2))
    for x in range(size_1):
        matrix[x, 0] = x
    for y in range(size_2):
        matrix[0, y] = y

    for x in range(1, size_1):
        for y in range(1, size_2):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                    )
    print(matrix.astype(int))
    return(int(matrix[size_1 - 1, size_2 - 1]))

print(levenshtein(["Data", "Mining"], ["Web", "Mining"]))
"""
Output
    [[0 1 2]
    [1 1 2]
    [2 2 1]]
    1
"""

#1b

def distance_matrix(sequences):

    x = len(sequences)

    matrix = [levenshtein(x,y) for i, x in enumerate(sequences) for j, y in enumerate(sequences)]

    print(np.array([matrix[i:i + x] for i in range(0, len(matrix), x)]))


distance_matrix([["Data", "Mining"], ["Web", "Mining"], ["DSAL", "Test"]])
""" 
Output is the correct matrix but with all previous matrices as well. That needs to be changed to just the result matrix
"""

#1c

def sequence_dbscan(sequences, eps, min_samples):
    C=0
    labels = [0]*len(sequences)
    for x in range(0, len(sequences)):
        if not (labels[x]==0):
            continue

        N = regionQuerry(x, eps)
        if len(N)< min_samples:
            lables[x] = -1
        else:
            C+=1
            recursiveExpandCluster(sequences,labels,N,x, C, eps, min_samples)
    return labels

def recursiveExpandCluster(sequences, labels, N, x, C, eps, min_samples):
    labels[x] = C
    i = 0
    while i< len(N):
        next_point = N[i]

        if next_point[next_point] == -1:
            labels[next_point] = C

        elif labels[next_point] == 0:
            labels[next_point] = C
            next_Point_N = regionQuerry(sequences, next_point, eps)

            if len(next_Point_N):
                N = N + next_Point_N

        i+=1

def regionQuerry(sequences, x, eps):
    neighbours = []
    for i in range(0, len(sequences)):
        if np.linalg.norm(sequences[x]-sequences[i]) < eps:
            neighbours.append(i)
    return neighbours

#2a

def fit_first_order_mc(sequences):
    sequences.insert(0,"R")
    sequences.insert(len(sequences), "R")
    states = Counter(sequences).keys()
    liste = []
    a = Counter(zip(sequences[:-1], sequences[1:]))
    b = Counter(a)

    c = np.array([[b[(i, j)] for j in states] for i in states], dtype=float)
    row_sums = c.sum(axis=1)
    new_matrix = c / row_sums[:, np.newaxis]

    print(np.around(new_matrix,3))

print(fit_first_order_mc(["G","G","G","B","B","G","B","G","G","G","G"]))
"""
Output
[[0.    1.    0.   ]
 [0.125 0.625 0.25 ]
 [0.    0.667 0.333]]
None
"""

def fit_second_order_mc(sequences):
    np.seterr(divide='ignore', invalid='ignore')

    sequences.insert(len(sequences), "R")
    sequences.insert(0, "R")
    x =[v + sequences[i + 1] for i, v in enumerate(sequences[:-1])]
    x.insert(0,"RR")
    print(x)
    states = Counter(x).keys()
    print(Counter(x))
    

    word = set(states)
    print("old: ", word)
    word.update(["R" + "R"] + ["R" + str(x) for x in set(sequences)] + [str(x) + "R" for x in set(sequences)])
    print("new: ", word)
    word = list(word)
    print("word: ", word)
    
    row_names = word
    col_names = word

    a = Counter(zip(x[:-1], x[1:]))
    b = Counter(a)

    c = np.array([[b[(i, j)] for j in word] for i in word], dtype=float)
    row_sums = c.sum(axis=1)
    t_m = c / row_sums[:, np.newaxis]
    t_m[np.isnan(t_m)] = 0

    return (np.around(t_m, 2), row_names, col_names)

print(fit_second_order_mc(["G", "G", "G", "B", "B", "G", "B", "G", "G", "G", "G"]))

#2b

def AIC_BIC(sequences, t_m, row_names, col_names):
#     print(len(row_names))
        
    a = Counter(zip(sequences[:-1], sequences[1:]))
    b = dict(a)
    print(a)
    print(b)
    n = len(row_names) * len(col_names)
    
    L = 1
    for i in range(len(row_names)):
        for j in range(len(col_names)):
           
            L *= t_m[i][j] ** b.get((row_names[i], col_names[j]), 0)
    
    k = len(row_names) + len(col_names)
    
    AIC = 2 * k - 2 * np.log(L)
    BIC = np.log(n) * k - 2 * np.log(L)
    
    print(BIC)
    
    return AIC
    return BIC

sequences=["G", "G", "G", "B", "B", "G", "B", "G", "G", "G", "G"]
t_m, row_names, col_names = fit_second_order_mc(sequences)

print(AIC_BIC(sequences, t_m, row_names, col_names))
