import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import numpy as np
import sys
sys.path.append('.')
from lib.dataset.augmentation import *
from lib.dataset.utils import flip_points,check_size
from lib.dataset.gen_heads import *
from lib.dataset.decode_heads import *
"""
    the landmarks output from all head methods should be correspond to the heatmap size. 
"""


class dataset_heads(data.Dataset):
    def __init__(self,config,read_mem=True,Is_train=True,transform=None):
        """
            config : experiments/**
            read_mem : bool, read dataset in memery or not.
            Is_train : bool, train or test
            transform: default is None, operations in torchvision
        """
        root_folder = os.path.join("data","benchmark")
        self.data_folder = config.data_folder if Is_train else config.test_folder
        self.data_path = os.path.join(root_folder,config.data_type,self.data_folder)
        self.points_flip = flip_points(config.data_type)
        self.is_train = Is_train
        self.transform = transform
        self.read_mem = read_mem
        self.config = config
        
        self.heat = gen_heat(config.heatmap_method,config.heatmap_sigma)
        self.gen_head = eval("gen_{}_head".format(config.head_type))

        label_path = os.path.join(root_folder,config.data_type,self.data_folder+".txt")
        with open(label_path,'r') as f:
            data_txt = f.readlines()
        data_info = np.array([x.strip().split() for x in data_txt])
        
        self.img_paths = data_info[:,0].copy()
        self.pts_array = data_info[:,1:].astype(np.float32).reshape(data_info.shape[0],-1,2).copy()
        self.imgs = [Image.open(os.path.join(self.data_path,img_path)).convert('RGB') 
                    for img_path in self.img_paths] if read_mem else []
        # check image size
        check_size(self.imgs,config)
        print("Finish READ and CHECK dataset, Success !")


    def __getitem__(self,index):
        
        img = self.imgs[index].copy() if self.read_mem \
              else Image.open(os.path.join(self.data_path,self.img_paths[index])).convert('RGB')
        target = self.pts_array[index].copy()

        if self.is_train:
            img, target = random_translate(img,target)
            img, target = random_flip(img, target, self.points_flip)
            img, target = random_rotate(img, target, angle_max=30)
            img = random_blur(img)
            img = random_occlusion(img)

        # ignore or pad crop,both are ok.
        img, target = pad_crop(img,target)
        # target = ignore_crop(target)  

        target_w_size = target * self.config.heatmap_size
        target_map, offset_map  = self.gen_head(target_w_size,self.heat,self.config)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        target_w_size = torch.from_numpy(target_w_size).float()
        target_map = torch.from_numpy(target_map).float()
        offset_map = torch.from_numpy(offset_map).float() if offset_map is not None else torch.zeros(1)

        return img,target_map,offset_map,target_w_size

    def __len__(self):
        return self.pts_array.shape[0]
