# -*- coding: utf-8*-
"""
Created on Fri Jun 30 14:02:07 2023

@author: zhuhe
"""

import numpy as np
from typing import List, Union


def _direct_to_cartesian(coord: np.ndarray,
                         lat: np.ndarray) -> np.ndarray :
    """
    convert a direct coordinate to a cartesian coordinate

    """
    return np.dot(coord, lat)
    

def _cartesian_to_direct(coord: np.ndarray,
                         lat: np.ndarray) -> np.ndarray :
    """
    convert a cartesian coordinate to a direct coordinate

    """
    return np.dot(coord, np.linalg.inv(lat))
    

def _convert_pair_to_str(index_prm_1: int,
                         index_prm_2: int,
                         lattice_vec: Union[List[int], np.ndarray]
                         ) -> str:
    """
    hash the atomic a pair by converting the information to a string
    
    """
    temp = str(index_prm_1) + str(index_prm_2)
    for x in lattice_vec:
        temp += str(int(x))
    return temp

    
def _cartesian_to_direct(coord: np.ndarray,
                         lat: np.ndarray):
    """
    convert a cartesian coordinate to a direct coordinate

    """
    return np.dot(coord, np.linalg.inv(lat))
        

def _modify_freq(eig_vals: np.ndarray):
    """
    convert the imaginary numbers to negative numbers

    """
    for j, eig_val in enumerate(eig_vals): 
        if eig_val >= 0:    
            eig_vals[j] = np.sqrt(eig_val)
        else:
            eig_vals[j] = -np.sqrt(-eig_val)


def _set_ylabel(unit: str) -> str:
    """
    determine the ylabel of the plot based on unit specified 

    """
    
    if isinstance(unit, str):
        return "Frequency ({})".format(unit)
    return "Frequency"


def _coth(x: float) -> float:
    return (np.exp(2*x) + 1) / (np.exp(2*x) - 1)


def _make_k_grid(xy_range,
                 num):
    """
    make a 2d k_grid; num points along each axis, between [-xy_range, xr_range]
    """
    interval = xy_range/num
    a = np.arange(-xy_range, xy_range+interval, interval)
    kx,ky = np.meshgrid(a,a)
    return kx, ky