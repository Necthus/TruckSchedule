import numpy as np

def softmax(x):
    e_x = np.exp(x)  
    return e_x / np.sum(e_x)

def normalized_reciprocal(x):
    recip = 1/(x+1)  
    return recip / np.sum(recip)