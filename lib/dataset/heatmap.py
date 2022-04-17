"""
    last version for generate heatmap and decode cordinate

    Attention : decode Heatmap to cordinate is value with int type. But native cordinate may be with decimal.
    Thus heatmap will produce loss at 0.7% when 2 times downsample, 1.3% when 4 times downsample


"""

import torch
import numpy as np
import math

def generate_target(img, pt, sigma):
    """
        return: a heatmap image.
    """
    
    tmp_size = sigma * 3

    if sigma % 1 > 1e-6:
        # sigma type float
        ul = np.array([pt[0] - tmp_size, pt[1] - tmp_size],dtype=int)
        br = np.array([pt[0] + tmp_size, pt[1] + tmp_size],dtype=int)
    else:
        # sigma type int
        ul =  np.array([round(pt[0] - tmp_size), round(pt[1] - tmp_size)],dtype=int)
        br =  np.array([round(pt[0] + tmp_size), round(pt[1] + tmp_size)],dtype=int)

    # Check that any part of the gaussian is in-bounds
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img
    
    # Generate gaussian
    size = 2 * tmp_size + 1 # ensure size is odd number
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian i s not normalized, we want the center value to equal 1
    # max value is 1,and to keep diff with other 0 pixels ,don't sub min,thus normalization is no use
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    br += 1
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])
    # print(g_x,g_y,img_x,img_y)

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def get_pts(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1)

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0]) % scores.size(3) 
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / scores.size(3)) 

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float() # gt(a,b) 比较a是否大于b 大于则为1，不大于则为0
    preds *= pred_mask
    return preds 

def get_pts_grad(hm):
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].remainder_(hm.size(3))
    preds[..., 1].div_(hm.size(2)).floor_()

    # add gradients in dx and dy, but a little effect.
    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]), int(preds[i, j, 1])
            if pX > 0 and pX < 63 and pY > 0 and pY < 63: # make sure don't overflow
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    return preds

def get_pts_all(target_maps):
    max_v,idx = torch.max(target_maps.view(target_maps.size(0),target_maps.size(1),target_maps.size(2)*target_maps.size(3)), 2)
    preds = idx.view(idx.size(0),idx.size(1),1).repeat(1,1,2).float()
    max_v = max_v.view(idx.size(0),idx.size(1),1)
    pred_mask = max_v.gt(0).repeat(1, 1, 2).float()

    preds[..., 0].remainder_(target_maps.size(3))
    preds[..., 1].div_(target_maps.size(2)).floor_()

    # add gradients in dx and dy, but get a little effects.
    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = target_maps[i, j, :]
            pX, pY = int(preds[i, j, 0]), int(preds[i, j, 1])
            if pX > 0 and pX < target_maps.size(3) - 1 and pY > 0 and pY < target_maps.size(2) - 1:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds *= pred_mask
    # print(preds.size())
    return preds
