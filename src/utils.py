import torch as th

"""
Author: Josue N Rivera
"""

def d3ls(dist1:th.Tensor, dist2:th.Tensor):
    return th.mean((dist1 - dist2)**2)