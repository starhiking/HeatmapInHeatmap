import torch
import torch.nn as nn
import numpy as np
import sys
from torch.nn import functional as F
from torch.nn.modules.loss import KLDivLoss
sys.path.append('.')

from lib.dataset.decode_heads import *
from torch import distributed as dist
from scipy.integrate import simps

def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size


class SoftKLLoss(nn.KLDivLoss):
    """
        l_n = (Softmax y_n) \cdot \left( \log (Softmax y_n) - \log (Softmax x_n) \right)
    """
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', log_target: bool = False) -> None:
        super(SoftKLLoss,self).__init__(size_average=size_average, reduce=reduce, reduction=reduction, log_target=log_target)
        self.logSoftmax = nn.LogSoftmax(dim=-1)
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """  
        input is pred
        target is gt
        """
        input = input.view(-1, input.shape[-2]*input.shape[-1])
        target = target.view(-1, target.shape[-2]*target.shape[-1])
        input = self.logSoftmax(input)
        target = self.Softmax(target)
        return F.kl_div(input, target, reduction=self.reduction, log_target=self.log_target)

def get_loss(config):
    criterion = None
    offset_criterion = None

    assert hasattr(config,"criterion_heatmap") , "Not exist criterion_heatmap in config file"

    criterion_str = config.criterion_heatmap.upper()
    if criterion_str in ['L2','MSE']:
        criterion = nn.MSELoss()
    elif criterion_str == 'L1':
        criterion = nn.L1Loss()
    elif criterion_str == 'SMOOTH_L1':
        criterion = nn.SmoothL1Loss()
    elif criterion_str == 'CLS':
        criterion = SoftKLLoss()
    else:
        raise ("Not support {} loss now.".format(criterion_str))

    
    if hasattr(config,"criterion_offset"):
        offset_criterion_str = config.criterion_offset.upper()
        if offset_criterion_str in ['L2','MSE']:
            offset_criterion = nn.MSELoss()
        elif offset_criterion_str == 'L1':
            offset_criterion = nn.L1Loss()
        elif offset_criterion_str == 'SMOOTH_L1':
            offset_criterion = nn.SmoothL1Loss()
        elif offset_criterion_str == 'CLS':
            offset_criterion = SoftKLLoss()
        else:
            raise ("Not support {} loss now.".format(offset_criterion_str))
    
    return criterion,offset_criterion


def calc_loss(config,criterion,stack_pred_heatmap,gt_heatmap,offset_criterion=None,stack_pred_offset=None,gt_offset=None):
    """
        calculate the loss for training in one iteration
    """
    # stack_pred_heatmap (5 dim): n_batch,n_stack,n_landmark,heatmap_size,heatmap_size
    # gt_heatmap (4 dim): n_batch,n_landmark,heatmap_size,heatmap_size
    # calculate the train loss for all stacks
    loss = 0.0 
    stack_gt_heatmap = gt_heatmap.unsqueeze(1).expand([-1,stack_pred_heatmap.size(1)]+list(gt_heatmap.size()[1:]))
    loss = criterion(stack_pred_heatmap,stack_gt_heatmap)
    if offset_criterion is not None:
        loss *= config.loss_heatmap_weight
        stack_gt_offset = gt_offset.unsqueeze(1).expand([-1,stack_pred_offset.size(1)]+list(gt_offset.size()[1:]))
        loss += config.loss_offset_weight * offset_criterion(stack_pred_offset,stack_gt_offset)
    
    return loss

def calc_inference_loss(config,criterion,pred_heatmap,gt_heatmap,offset_criterion=None,pred_offset=None,gt_offset=None):
    """
        calculate the loss for testing in one iteration
    """
    # pred_heatmap (4 dim): n_batch,n_landmark,heatmap_size,heatmap_size
    # gt_heatmap (4 dim): n_batch,n_landmark,heatmap_size,heatmap_size
    # calculate the loss for the last stack, and the nme
    loss = criterion(pred_heatmap,gt_heatmap)
    if offset_criterion is not None:
        loss *= config.loss_heatmap_weight
        loss += config.loss_offset_weight * offset_criterion(pred_offset,gt_offset) 
    
    return loss

def calc_nme(config,target_w_size,pred_heatmap,pred_offset=None):

    """
        Args: 
                target_w_size: tensor in pytorch (n,98,2)
                pred_heatmap : tensor in pytorch (n,98,64,64) 
            
        Return:
                Sum_ION : the sum Ion of this batch data
    """
    assert len(pred_heatmap.size()) == 4 , "the pred_heatmap must be 4 dim, use inference function"

    decode_head_func = eval('decode_'+config.head_type+'_head')
    preds = decode_head_func(pred_heatmap,pred_offset)

    ION = []
    norm_indices = None  # ION
    if config.data_type == "300W":
        norm_indices = [36,45]
    elif config.data_type == "COFW":
        norm_indices = [8,9] # ION
        # norm_indices = [17,16] # IPN
    elif config.data_type == "WFLW":
        norm_indices = [60,72]
    elif config.data_type == "AFLW":
        pass
    else:
        print("No such data!")
        exit(0)

    # target_w_size and preds : n, 98 , 2
    target_np = target_w_size.cpu().numpy().reshape(pred_heatmap.size(0),-1,2)
    pred_np = preds.cpu().numpy().reshape(pred_heatmap.size(0),-1,2)

    for target,pred in zip(target_np,pred_np):
        diff = target - pred
        norm = np.linalg.norm(target[norm_indices[0]] - target[norm_indices[1]]) if norm_indices is not None else config.heatmap_size
        c_ION = np.sum(np.linalg.norm(diff,axis=1))/(diff.shape[0]*norm)
        ION.append(c_ION)


    Sum_ION = np.sum(ION) # the ion of this batch 
    # need div the dataset size to get nme

    return Sum_ION, ION

def compute_fr_and_auc(nmes, thres=0.10, step=0.0001):
    num_data = len(nmes)
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thres
    nme = np.mean(nmes)

    print("NME %: {}".format(np.mean(nmes)*100))
    print("FR_{}% : {}".format(thres,fr*100))
    print("AUC_{}: {}".format(thres,auc))
    return nme, fr, auc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def need_val(epoch,config):
    val_info = np.array(config.val_epoch)
    val_epochs = val_info[:,0]
    val_steps = val_info[:,1]

    for end_epoch,step in zip(val_epochs,val_steps):
        if epoch < end_epoch:
            if epoch % step == 0:
                return True
            else:
                return False
        

def lr_repr(optim):
    _lr_repr_ = ''
    for pg in optim.param_groups:
        _lr_repr_ += ' {} '.format(pg['lr'])
    return _lr_repr_