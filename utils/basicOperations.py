import operator
import numpy as np

def linker_goody(x1, x2, x3):
    return operator.truediv(x1, (x2 + x3))
def root(x1):
    return np.sqrt(x1)