import torch.utils.data as data
import torch
import numpy as np
import sys
sys.path.append('.')
import copy
import os
from lib.dataset.utils import flip_points,check_size

"""
    the landmarks output from all head methods should be correspond to the heatmap size. 
"""

def gen_heat(method="Gauss",sigma=0.):

    if sigma == 0 :
        return np.array([[1]])

    tmp_size = sigma * 3
    size = 2 * tmp_size + 1
    x = np.arange(0,size,1,dtype=np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return g


def gen_woo_head(target_w_size,heat,config):
    # without offset map head: normal head
    target_map = np.zeros((config.num_landmarks,config.heatmap_size,config.heatmap_size),dtype=np.float32)

    if config.heatmap_method == "GAUSS":
        heat = heat.copy()
        tmp_size = config.heatmap_sigma * 3
        is_sigma_int = True if config.heatmap_sigma % 1 < 1e-6 else False
        
        for i in range(target_w_size.shape[0]):
            # pt : [x,y], pt_map : [heatmap_size x heatmap_size]
            pt,pt_map = target_w_size[i],target_map[i]
            if is_sigma_int:
                ul =  np.array([round(pt[0] - tmp_size), round(pt[1] - tmp_size)],dtype=int)
                br =  np.array([round(pt[0] + tmp_size), round(pt[1] + tmp_size)],dtype=int)
            else:
                ul = np.array([pt[0] - tmp_size, pt[1] - tmp_size],dtype=int)
                br = np.array([pt[0] + tmp_size, pt[1] + tmp_size],dtype=int)

            # Check that any part of the gaussian is in-bounds
            if (ul[0] >= pt_map.shape[1] or ul[1] >= pt_map.shape[0] or
                    br[0] < 0 or br[1] < 0):
                # If not, just return the image as is
                continue
            br += 1 # take the same process as shape, thus add 1
            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], pt_map.shape[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], pt_map.shape[0]) - ul[1]
            # Image range
            pt_x = max(0, ul[0]), min(br[0], pt_map.shape[1])
            pt_y = max(0, ul[1]), min(br[1], pt_map.shape[0])

            pt_map[pt_y[0]:pt_y[1], pt_x[0]:pt_x[1]] = heat[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target_map, None

    elif config.heatmap_method == "DIRECT":
        for i in range(target_w_size.shape[0]):
            # pt : [x,y], pt_map : [heatmap_size x heatmap_size]
            pt,pt_map = target_w_size[i],target_map[i]
            pt_x = pt[0]
            pt_y = pt[1]
            # seen on the corner does better than unseen
            pt_x = max(0,pt_x)
            pt_y = max(0,pt_y)
            pt_x = int(min(pt_map.shape[1] - 1,pt_x))
            pt_y = int(min(pt_map.shape[0] - 1,pt_y))

            pt_map[pt_y,pt_x] = 1
        return  target_map, None
    else:
        print("Not support method for gen heatmap.")
        exit(-1)


def gen_afc_head(target_w_size,heat,config):
    # add fc as offset
    target_map = np.zeros((config.num_landmarks,config.heatmap_size,config.heatmap_size),dtype=np.float32)
    target_int = np.floor(target_w_size)
    target_int = np.clip(target_int,0,config.heatmap_size - 1)
    offset_float = target_w_size - target_int

    if config.heatmap_method == "GAUSS":
        heat = heat.copy()
        tmp_size = config.heatmap_sigma * 3
        is_sigma_int = True if config.heatmap_sigma % 1 < 1e-6 else False
        for i in range(target_int.shape[0]):
            # pt : [x,y], pt_map : [heatmap_size x heatmap_size] 
            pt,pt_map = target_int[i],target_map[i]
            if is_sigma_int:
                ul =  np.array([round(pt[0] - tmp_size), round(pt[1] - tmp_size)],dtype=int)
                br =  np.array([round(pt[0] + tmp_size), round(pt[1] + tmp_size)],dtype=int)
            else:
                ul = np.array([pt[0] - tmp_size, pt[1] - tmp_size],dtype=int)
                br = np.array([pt[0] + tmp_size, pt[1] + tmp_size],dtype=int)

            # Check that any part of the gaussian is in-bounds
            if (ul[0] >= pt_map.shape[1] or ul[1] >= pt_map.shape[0] or
                    br[0] < 0 or br[1] < 0):
                # If not, just return the image as is
                continue
            br += 1 # take the same process as shape, thus add 1
            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], pt_map.shape[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], pt_map.shape[0]) - ul[1]
            # Image range
            pt_x = max(0, ul[0]), min(br[0], pt_map.shape[1])
            pt_y = max(0, ul[1]), min(br[1], pt_map.shape[0])

            pt_map[pt_y[0]:pt_y[1], pt_x[0]:pt_x[1]] = heat[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target_map,offset_float
    
    elif config.heatmap_method == "DIRECT":
        for i in range(target_int.shape[0]):
            # pt : [x,y], pt_map : [heatmap_size x heatmap_size] 
            pt,pt_map = target_int[i],target_map[i]
            pt_x = pt[0]
            pt_y = pt[1]
            # seen on the corner will do better than unseen
            pt_x = max(0,pt_x)
            pt_y = max(0,pt_y)
            pt_x = int(min(pt_map.shape[1] - 1,pt_x))
            pt_y = int(min(pt_map.shape[0] - 1,pt_y))

            pt_map[pt_y,pt_x] = 1
        return  target_map,offset_float
    
    else:
        print("Not support method for gen heatmap.")
        exit(-1)


def gen_hih_head(target_w_size,heat,config):
    # add another heatmap as offset
    # target_w_size is target * config.heatmap_size
    target_map, offset_float = gen_afc_head(target_w_size,heat,config)    
    offset_w_size = offset_float * config.offset_size
    
    offset_config = copy.copy(config)
    offset_config.heatmap_size = config.offset_size
    offset_config.heatmap_method = config.offset_method
    offset_config.heatmap_sigma = config.offset_sigma

    if config.heatmap_sigma != config.offset_sigma:
        offset_heat = gen_heat(config.offset_method,config.offset_sigma)
    else:
        offset_heat = heat

    offset_map, _ = gen_woo_head(offset_w_size,offset_heat,offset_config)

    return target_map, offset_map

def gen_od_head(target_w_size,heat,config):
    # od:object_detection
    # offset head like cornernet and extremenet
    offset_map = np.zeros((2,config.heatmap_size,config.heatmap_size),dtype=np.float32) # 2,h,w  2->(x-,y-)
    target_map, offset_float = gen_afc_head(target_w_size,heat,config)
    
    target_int = np.floor(target_w_size)
    target_int = np.clip(target_int,0,config.heatmap_size - 1).astype(np.uint8)
    for i in range(target_int.shape[0]):
        pt,offset = target_int[i],offset_float[i]
        offset_map[0,pt[1],pt[0]] = offset[0] # offset_x
        offset_map[1,pt[1],pt[0]] = offset[1] # offset_y

    return target_map,offset_map

