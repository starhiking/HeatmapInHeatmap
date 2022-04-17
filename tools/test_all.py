import imp
import os
import sys
import time
sys.path.append('.')
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from lib.utils.model import get_model
from lib.utils.parser import inference_parser_args as parser_args
from lib.utils.check import create_logger
from lib.utils.loss import get_loss,lr_repr,need_val
from lib.dataset.dataset import dataset_heads
from lib.utils.process import inference_model

label_300W = [
        "test",
        "valid",
        "valid_challenge",
        "valid_common"
]

label_WFLW = [
        "test",
        "test_occlusion",
        "test_makeup",
        "test_largepose",
        "test_illumination",
        "test_expression",
        "test_blur"
]

# COFW mat file contains information, thus don't need csv file

def main_test_all(config):

    assert config.resume_checkpoint,"Not find the checkpoint"
    labels = eval("label_"+config.data_type.upper())
    logger = create_logger(config,Is_test=True)
    logger.info("Test all subsets for {}".format(config.data_type))
    logger.info(config._print())
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    model = get_model(config)
    criterion, offset_criterion = get_loss(config)
    if config.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    model = model.to(device)
    criterion.to(device)
    if offset_criterion is not None:
        offset_criterion.to(device)

    if config.resume_checkpoint:
        try:
            model_checkpoint = torch.load(config.resume_checkpoint)
            model.load_state_dict(model_checkpoint['state_dict'],strict=True)
            best_nme = model_checkpoint['best_nme']
            logger.info("Restore from {}".format(config.resume_checkpoint))
        except:
            logger.info("Restore failed.")
            exit()
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    

    for label in labels:
        setattr(config,"test_folder",label)

        val_data = dataset_heads(config,read_mem=False,Is_train=False,transform=val_transform)

        val_loader = data.DataLoader(
                        val_data,
                        batch_size=16,
                        shuffle=False,
                        num_workers=4,
                        drop_last=False,
                        pin_memory=False
        )

        val_loss,val_nme = inference_model(config,device,model,val_loader,criterion,offset_criterion)
        logger.info("{}: [nme: {}] [loss: {}]".format(label,val_nme,val_loss))


if __name__ == "__main__":
    config = parser_args()
    main_test_all(config)