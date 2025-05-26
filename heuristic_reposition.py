

import random
from utils import *
import numpy as np


def normalized_reciprocal(x):
    recip = 1/(x+1)  
    return recip / np.sum(recip)

def RandomReturn(stations_features):
    
    station_num = len(stations_features)
        
    return random.choice(range(station_num))

def NearestReturn(stations_features):
        
    dists = [feature[2] for feature in stations_features]
    min_index = np.argmin(dists)
    
    return min_index

def NearestPropReturn(stations_features):
    
    station_num = len(stations_features)
    dists = [feature[2] for feature in stations_features]
    dists = np.array(dists)
    prop = normalized_reciprocal(dists)
    
    return np.random.choice(range(station_num),p=prop)

def LeastTruckReturn(stations_features):
        
    truck_nums = [feature[1] for feature in stations_features]
    min_index = np.argmin(truck_nums)
    
    return min_index


def LeastTruckProbReturn(stations_features):

    station_num = len(stations_features)
    truck_nums = [feature[1] for feature in stations_features]    
    truck_nums = np.array(truck_nums)
    
    prop = normalized_reciprocal(truck_nums)
    
    return np.random.choice(range(station_num),p=prop)