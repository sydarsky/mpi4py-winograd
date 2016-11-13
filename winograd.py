#!/usr/bin/python
# -*- coding: utf-8 -*-

from optparse import OptionParser
from math import ceil, log
from profiler import Profiler


def saveMatrix(matrix_temp, filename="result.in"):
    f = open(filename, 'w')
    for line in matrix_temp:
            f.write(str(line) + "\n")


def read(filename):
    lines = open(filename, 'r').read().splitlines()
    A = []
    B = []
    matrix = A
    for line in lines:
        if line != "":
            matrix.append(list(map(int, line.split("\t"))))
        else:
            matrix = B
    return A, B


def printMatrix(matrix):
    for line in matrix:
        print("\t".join(map(str, line)))


def ikjMatrixProduct(A, B):
    n = len(A)
    C = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for k in range(n):
            for j in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def add(A, B):
    n = len(A)
    C = [[0 for j in range(0, n)] for i in range(0, n)]
    for i in range(0, n):
        for j in range(0, n):
            C[i][j] = A[i][j] + B[i][j]
    return C


def subtract(A, B):
    n = len(A)
    C = [[0 for j in range(0, n)] for i in range(0, n)]
    for i in range(0, n):
        for j in range(0, n):
            C[i][j] = A[i][j] - B[i][j]
    return C


def strassenW(A, B):
    """
        Implementation of the strassen-winograd algorithm.
    """
    n = len(A)

    if n <= LEAF_SIZE:
        return ikjMatrixProduct(A, B)
    else:
        # initializing the new sub-matrices
        newSize = int(n / 2)
        a11 = [[0 for j in range(0, newSize)] for i in range(0, newSize)]
        a12 = [[0 for j in range(0, newSize)] for i in range(0, newSize)]
        a21 = [[0 for j in range(0, newSize)] for i in range(0, newSize)]
        a22 = [[0 for j in range(0, newSize)] for i in range(0, newSize)]

        b11 = [[0 for j in range(0, newSize)] for i in range(0, newSize)]
        b12 = [[0 for j in range(0, newSize)] for i in range(0, newSize)]
        b21 = [[0 for j in range(0, newSize)] for i in range(0, newSize)]
        b22 = [[0 for j in range(0, newSize)] for i in range(0, newSize)]

        tResult = [[0 for j in range(0, newSize)] for i in range(0, newSize)]

        # dividing the matrices in 4 sub-matrices:
        for i in range(0, newSize):
            for j in range(0, newSize):
                a11[i][j] = A[i][j]                      # top left
                a12[i][j] = A[i][j + newSize]            # top right
                a21[i][j] = A[i + newSize][j]            # bottom left
                a22[i][j] = A[i + newSize][j + newSize]  # bottom right

                b11[i][j] = B[i][j]                      # top left
                b12[i][j] = B[i][j + newSize]            # top right
                b21[i][j] = B[i + newSize][j]            # bottom left
                b22[i][j] = B[i + newSize][j + newSize]  # bottom right

        # Calculating s1 to s8:
        s1 = add(a21, a22)                  # s1 = (a12+a22)
        s2 = subtract(s1, a11)              # s2 = (s1-a11)
        s3 = subtract(a11, a21)             # p3 = (a11-a21)
        s4 = subtract(a12, s2)              # s4 = (a12-s2)
        s5 = subtract(b12, b11)             # s5 = (b12-b11)
        s6 = subtract(b22, s5)              # s6 = (b22-s55)
        s7 = subtract(b22, b12)             # s7 = (b22-b21)
        s8 = subtract(s6, b21)              # s8 = (s6-b21)

        # calculating p1 to p7:
        p1 = strassenW(s2, s6)              # p1 = (s1*s6)
        p2 = strassenW(a11, b11)            # p2 = (a11*b11)
        p3 = strassenW(a12, b21)            # p3 = (a12*b21)
        p4 = strassenW(s3, s7)              # p4 = (s3*s7)
        p5 = strassenW(s1, s5)              # p5 = (s1*s5)
        p6 = strassenW(s4, b22)             # p6 = (s4*b22)
        p7 = strassenW(a22, s8)             # p7 = (a22*s8)

        # calculating t1 and t2:
        t1 = add(p1, p2)                    # t1 = (p1+p2)
        t2 = add(t1, p4)                    # t2 = (t1+p4)

        # calculating c11, c12, c21, c22:
        c11 = add(p2, p3)                   # c11 = (p2+p3)

        tResult = add(p5, p6)
        c12 = add(t1, tResult)              # c12 = (t1+p5+p6)

        c21 = subtract(t2, p7)              # c21 = (t2-p7)
        c22 = add(t2, p5)                   # c22 = (t2+p5)

        # Grouping the results obtained in a single matrix:
        C = [[0 for j in range(0, n)] for i in range(0, n)]
        for i in range(0, newSize):
            for j in range(0, newSize):
                C[i][j] = c11[i][j]
                C[i][j + newSize] = c12[i][j]
                C[i + newSize][j] = c21[i][j]
                C[i + newSize][j + newSize] = c22[i][j]
        return C


def strassen(A, B):
    assert type(A) == list and type(B) == list
    assert len(A) == len(A[0]) == len(B) == len(B[0])

    # Make the matrices bigger so that we can apply the strassen-winograd
    # algorithm recursively without having to deal with odd
    # matrix sizes
    nextPowerOfTwo = lambda n: 2**int(ceil(log(n, 2)))
    n = len(A)
    m = nextPowerOfTwo(n)
    APrep = [[0 for i in range(m)] for j in range(m)]
    BPrep = [[0 for i in range(m)] for j in range(m)]
    for i in range(n):
        for j in range(n):
            APrep[i][j] = A[i][j]
            BPrep[i][j] = B[i][j]
    CPrep = strassenW(APrep, BPrep)
    C = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            C[i][j] = CPrep[i][j]
    return C


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", dest="filename", default="2000.in",
                      help="input file with two matrices", metavar="FILE")
    parser.add_option("-l", dest="LEAF_SIZE", default="8",
                      help="when do you start using ikj", metavar="LEAF_SIZE")
    (options, args) = parser.parse_args()

    LEAF_SIZE = int(options.LEAF_SIZE)
    A, B = read(options.filename)

    with Profiler() as timer:
        C = strassen(A, B)
    # printMatrix(C)
    saveMatrix(C)
