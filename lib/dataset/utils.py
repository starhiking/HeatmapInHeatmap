import numpy as np
from PIL import Image

def flip_points(data_type="WFLW"):
    data_type = data_type.upper()
    points_flip = None
    if data_type == '300W':
        points_flip = [16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,26,25,24,23,22,21,20,19,18,17,27,28,29,30,35,34,33,32,31,45,44,43,42,47,46,39,38,37,36,41,40,54,53,52,51,50,49,48,59,58,57,56,55,64,63,62,61,60,67,66,65]
        assert len(points_flip) == 68
    elif data_type == 'WFLW':
        points_flip = [32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,46,45,44,43,42,50,49,48,47,37,36,35,34,33,41,40,39,38,51,52,53,54,59,58,57,56,55,72,71,70,69,68,75,74,73,64,63,62,61,60,67,66,65,82,81,80,79,78,77,76,87,86,85,84,83,92,91,90,89,88,95,94,93,97,96]
        assert len(points_flip) == 98
    elif data_type == 'COFW':
        points_flip = [1,0,3,2,6,7,4,5,9,8,11,10,14,15,12,13,17,16,19,18,20,21,23,22,24,25,26,27,28]
        assert len(points_flip) == 29
    elif data_type == 'AFLW':
        points_flip = [5,4,3,2,1,0,11,10,9,8,7,6,14,13,12,17,16,15,18]
        assert len(points_flip) == 19
    else:
        print('No such data!')
        exit(0)
    return points_flip

def check_size(imgs,config):
    """
        check dataset read success and size is same as input size.
        Args:
            imgs: [Image,...]
            config : config py
    """
    for i in range(len(imgs)):
        img = imgs[i] 
        if img.height != config.input_size or img.width != config.input_size:
            print("{}th Image is not applicable ({},{}),need delete or resize.".format(i+1,img.height,img.width))
            exit(-1)

