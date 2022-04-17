import torch, cv2, os
import numpy as np
import sys
sys.path.append('.')
from lib.utils.loss import AverageMeter,calc_loss,calc_inference_loss,calc_nme,compute_fr_and_auc
from lib.dataset.decode_heads import decode_woo_head,decode_hih_head
import math

def train_model(config,device,model,train_loader,optimizer,criterion,offset_criterion=None):
    
    losses = AverageMeter()
    losses.reset()
    model.train()

    for i,(imgs,target_maps,offset_maps,gt_landmarks) in enumerate(train_loader):
        imgs = imgs.to(device)
        target_maps = target_maps.to(device)
        offset_maps = offset_maps.to(device) if config.head_type.upper != 'WOO' else None

        stack_pred_heatmap,stack_pred_offset = model(imgs)

        index = [i for i in range(config.num_stack-1, -1, -config.per_stack_heatmap)]
        stack_pred_heatmap = stack_pred_heatmap[:,index,...]
        loss = calc_loss(config,criterion,stack_pred_heatmap,target_maps,offset_criterion,stack_pred_offset,offset_maps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(),imgs.size(0))
    
    return losses.avg

def inference_model(config,device,model,val_loader,criterion,offset_criterion=None):
    losses = AverageMeter()
    losses.reset()
    model.eval()
    dataset_size = len(val_loader.dataset)
    SME = 0.0
    IONs = None
    inference_indice = config.inference_indice if hasattr(config,'inference_indice') else -1
    decode_head_func = eval('decode_'+config.head_type+'_head')

    with torch.no_grad():
        for i,(imgs,target_maps,offset_maps,gt_landmarks) in enumerate(val_loader):
            imgs = imgs.to(device)
            target_maps = target_maps.to(device)
            offset_maps = offset_maps.to(device) if config.head_type.upper != 'WOO' else None
            gt_landmarks = gt_landmarks.to(device)
            
            pred_heatmap,pred_offset = model.inference(imgs,inference_indice)
            loss = calc_inference_loss(config,criterion,pred_heatmap,target_maps,offset_criterion,pred_offset,offset_maps)
            sum_ion, ion = calc_nme(config,gt_landmarks,pred_heatmap,pred_offset)
            SME += sum_ion
            IONs = np.concatenate((IONs,ion),0) if IONs is not None else ion
            losses.update(loss.item(),imgs.size(0))

    nme,fr,auc = compute_fr_and_auc(IONs)
    
    NME = SME / dataset_size
    return losses.avg, NME

def split_hih_losses(config,device,model,val_loader,criterion,offset_criterion,threshold):

    model.eval()
    inference_indice = config.inference_indice if hasattr(config,'inference_indice') else -1

    match_losses = []
    mismatch_losses = []

    with torch.no_grad():
        for i,(imgs,target_maps,offset_maps,gt_landmarks) in enumerate(val_loader):
            imgs = imgs.to(device)
            target_maps = target_maps.to(device)
            offset_maps = offset_maps.to(device) if config.head_type.upper != 'WOO' else None
            gt_landmarks = gt_landmarks.to(device)
            
            # gt_offset = gt_landmarks - gt_landmarks.to(torch.int)
            # offset_resolution = torch.tensor(offset_maps.shape[-2:]).to(gt_offset)
            # gt_offset_location = gt_offset * offset_resolution

            pred_heatmap,pred_offset = model.inference(imgs,inference_indice)

            offset_location_decode_gt = decode_woo_head(offset_maps)
            offset_location_decode_pred = decode_woo_head(pred_offset)

            offset_losses = offset_criterion(pred_offset,offset_maps).view(-1,8,8).cpu().numpy()
            diffs = offset_location_decode_gt - offset_location_decode_pred
            diffs = diffs.cpu().numpy().reshape(-1,2)
            
            for i in range(len(diffs)):
                diff = diffs[i]
                loss = offset_losses[i]
                distance = np.linalg.norm(diff)
                mean_loss = np.mean(loss)

                if distance < threshold:
                    match_losses.append(mean_loss)
                else:
                    mismatch_losses.append(mean_loss)
    
    match_losses = np.array(match_losses)
    mismatch_losses = np.array(mismatch_losses)

    return match_losses, mismatch_losses
    