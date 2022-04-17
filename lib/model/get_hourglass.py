import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from functools import partial
import numpy as np
import time
import os
import sys
import math
from timm.models.vision_transformer import VisionTransformer, Attention
sys.path.append('.')

def attention_flops_counter_hook(module, input, output):
    input = input[0]
    B,N,C = input.shape
    qkv_linear_ops=C*3*B*N*C
    attention_ops=B*N*N*C*2
    proj_ops=B*N*C*C
    total_ops=qkv_linear_ops+proj_ops+attention_ops
    module.__flops__ += int(total_ops)

def count_attention(m:Attention,x:(torch.Tensor,),y:torch.Tensor):
    x = x[0]
    B, N, C = x.shape
    qkv_linear_ops=C*3*B*N*C
    attention_ops=B*N*N*C*2
    proj_ops=B*N*C*C
    total_ops=qkv_linear_ops+proj_ops+attention_ops
    m.total_ops += torch.DoubleTensor([int(total_ops)])

def forward_wo_cls(self,x):
    B = x.shape[0]
    x = self.patch_embed(x)

    # cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    # x = torch.cat((cls_tokens, x), dim=1)
    x = x + self.pos_embed[:,1:]
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)
    return x

VisionTransformer.forward_wo_cls = forward_wo_cls


class ConvBNReLu(nn.Module):
    def __init__(self,inp,oup,kernel=3,stride=1,bn=True,relu=True):
        super(ConvBNReLu,self).__init__()
        
        self.conv = nn.Conv2d(inp,oup,kernel,stride,padding=(kernel-1)//2,bias=False)
        self.bn = nn.BatchNorm2d(oup) if bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True) if relu else nn.Sequential()

    def forward(self,x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class Residual(nn.Module):
    def __init__(self,inp,oup):
        super(Residual,self).__init__()
        self.conv1 = ConvBNReLu(inp,int(oup/2),1)
        self.conv2 = ConvBNReLu(int(oup/2),int(oup/2),3)
        self.conv3 = ConvBNReLu(int(oup/2),oup,1,relu=False)
        self.skip_layer = ConvBNReLu(inp,oup,1,relu=False)
        self.need_skip = False if inp == oup else True
        self.last_relu = nn.ReLU(inplace=True)

    def forward(self,x):
        
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.last_relu(out)

        return out

class Down_Block(nn.Module):
    def __init__(self,inp,oup,stride=2):
        super(Down_Block,self).__init__()  
        self.conv1 = ConvBNReLu(inp,oup,3,stride=stride)
        self.conv2 = ConvBNReLu(oup,oup,3,relu=False)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            ConvBNReLu(inp,oup,1,stride=stride,relu=False)
        ) if stride!=1 or inp!=oup else nn.Sequential()

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.downsample(x)
        out = self.relu(out)
        return out

class AFC_head(nn.Module):
    def __init__(self,config):
        super(AFC_head,self).__init__()
        target_c = config.target_o
        
        self.conv1 = ConvBNReLu(config.num_feature,target_c,1)
        self.conv3x3_list = [
            ConvBNReLu(target_c,target_c,3)
            for i in range(config.num_o)
        ]
        self.convs = nn.Sequential(*self.conv3x3_list)
        self.conv2 = ConvBNReLu(target_c,target_c,1)
        self.shortcut = ConvBNReLu(config.num_feature,target_c,1) if config.num_feature != target_c else nn.Sequential()

        self.head = nn.Sequential(
            ConvBNReLu(target_c,target_c,1),
            nn.MaxPool2d(3,2),
            ConvBNReLu(target_c,target_c,3,2),
            ConvBNReLu(target_c,2*target_c,1),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(2*target_c,2*config.num_landmarks)

    def forward(self,x):

        conv1 = self.conv1(x)
        convs = self.convs(conv1)
        conv2 = self.conv2(convs)
        merge_feature = conv2 + self.shortcut(x)
        offset = self.head(merge_feature)
        offset = offset.view(offset.size(0),-1)
        offset = self.fc(offset)
        offset = offset.view(offset.size(0),-1,2)

        return offset  

    def _init_offset_params(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

class HIH_head(nn.Module):
    def __init__(self,config):
        super(HIH_head,self).__init__()
        target_c = config.target_o
        
        self.conv1 = ConvBNReLu(config.num_feature,target_c,1)
        self.pool = nn.MaxPool2d(3,2)
        downsample_stride = config.heatmap_size // config.offset_size
        down_step = int(math.log(downsample_stride,2))
        self.down_list = [
            Down_Block(target_c,target_c,2)
            for d_i in range(down_step - 1)
        ]
        
        self.down_ops = nn.Sequential(*self.down_list)
        self.head = nn.Sequential(
            ConvBNReLu(target_c,target_c,1),
            nn.Conv2d(target_c,config.num_landmarks,1)
        )
        
    def forward(self,x):
        conv1 = self.conv1(x)
        pool = self.pool(conv1)
        feature = self.down_ops(pool)
        offset = self.head(feature)

        return offset  

    def _init_offset_params(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

class OD_head(nn.Module):
    def __init__(self,config):
        super(OD_head,self).__init__()
        target_c = config.target_o
        
        self.conv1 = ConvBNReLu(config.num_feature,target_c,1)
        self.conv3x3_list = [
            ConvBNReLu(target_c,target_c,3)
            for i in range(config.num_o)
        ]
        self.convs = nn.Sequential(*self.conv3x3_list)
        self.conv2 = ConvBNReLu(target_c,target_c,1)
        self.shortcut = ConvBNReLu(config.num_feature,target_c,1) if config.num_feature != target_c else nn.Sequential()

        self.head = nn.Sequential(
            ConvBNReLu(target_c,target_c,1),
            nn.Conv2d(target_c,2,1)
        )

    def forward(self,x):
        conv1 = self.conv1(x)
        convs = self.convs(conv1)
        conv2 = self.conv2(convs)
        merge_feature = conv2 + self.shortcut(x)
        offset = self.head(merge_feature)

        return offset  

    def _init_offset_params(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    

class VIT_head(nn.Module):
    def __init__(self,config):
        super(VIT_head,self).__init__()
        self.config  = config
        self.vit = VisionTransformer(img_size=config.heatmap_size,in_chans=config.num_feature,
                                    patch_size=config.patch_size,embed_dim=config.embed_dim,depth=config.transformer_layer,num_heads=config.num_head,
                                    mlp_ratio=config.mlp_r, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.offset_lin = nn.Sequential(
                            ConvBNReLu(config.embed_dim, config.num_feature,1),
                            nn.Conv2d(config.num_feature,config.num_landmarks,1))

    def forward(self,x):
        
        vit_feature = self.vit.forward_wo_cls(x).transpose(1,2).contiguous().view([x.shape[0],self.config.embed_dim,x.shape[2]//self.config.patch_size,x.shape[3]//self.config.patch_size])
        vit_offset  = self.offset_lin(vit_feature)

        return vit_offset
    
    def _init_offset_params(self):
        # vit exist init weight in its code,which don't need.
        offset_modules = self.offset_lin.modules()
        for m in offset_modules:
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

class Hourglass(nn.Module):
    def __init__(self,config):
        super(Hourglass,self).__init__()
        self._n = config.num_layer
        self._f = config.num_feature
        self.config = config
        self._init_layers(self._n,self._f)

    def _init_layers(self,n,f):
        setattr(self, 'res' + str(n) + '_1', Residual(f, f))
        setattr(self, 'pool' + str(n) + '_1', nn.MaxPool2d(2, 2))
        setattr(self, 'res' + str(n) + '_2', Residual(f, f))
        if n > 1:
            self._init_layers(n - 1, f)
        else:
            self.res_center = Residual(f, f)
        setattr(self, 'res' + str(n) + '_3', Residual(f, f))

    def _forward(self, x, n, f):
        up1 = eval('self.res' + str(n) + '_1')(x)

        low1 = eval('self.pool' + str(n) + '_1')(x)
        low1 = eval('self.res' + str(n) + '_2')(low1)
        if n > 1:
            low2 = self._forward(low1, n - 1, f)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.' + 'res' + str(n) + '_3')(low3)
        up2 = nn.functional.interpolate(low3, scale_factor=2, mode='bilinear', align_corners=True)

        return up1 + up2

    def forward(self,x):
        return  self._forward(x,self._n,self._f)


class StackedHourGlass(nn.Module):
    def __init__(self,config):
        super(StackedHourGlass,self).__init__()
        self.config = config       
        self.stride = config.input_size // config.heatmap_size
        
        self.pre_conv = nn.Sequential(
            ConvBNReLu(3,config.num_feature//4,7,2),
            Residual(config.num_feature // 4, config.num_feature // 2),
            nn.MaxPool2d(2,2),
            Residual(config.num_feature // 2, config.num_feature // 2),
            Residual(config.num_feature // 2, config.num_feature))

        self._init_stacked_hourglass(config)
        self.init_weights(config.pretrained_path if config.pretrained and hasattr(config,"pretrained_path") else '')

        self.offset_func = eval(config.head_type.upper() + '_head') if config.head_type.upper() != 'WOO' else None      
        self.offset_head = self.offset_func(config) if self.offset_func is not None else None
        if self.offset_func is not None:
            self.offset_head._init_offset_params() 

    # init hourglass
    def _init_stacked_hourglass(self,config):
        for i in range(config.num_stack):
            setattr(self, 'hg' + str(i), Hourglass(config))
            setattr(self, 'hg' + str(i) + '_res1',
                    Residual(config.num_feature, config.num_feature))
            setattr(self, 'hg' + str(i) + '_lin1',
                    ConvBNReLu(config.num_feature, config.num_feature,1))
            setattr(self, 'hg' + str(i) + '_conv_pred',
                    nn.Conv2d(config.num_feature, config.num_landmarks, 1))
            # if self.offset_func is not None:
            #     setattr(self, 'offset_' + str(i),
            #         self.offset_func(config)) 
            
            setattr(self, 'hg' + str(i) + '_conv1',
                    ConvBNReLu(config.num_feature, config.num_feature, 1))
            setattr(self, 'hg' + str(i) + '_conv2',
                    ConvBNReLu(config.num_landmarks, config.num_feature, 1))

    def forward(self,x):
        x = self.pre_conv(x)
        out_preds = []
        # out_offsets = []

        for i in range(self.config.num_stack):
            hg = eval('self.hg'+str(i))(x)
            ll = eval('self.hg'+str(i)+'_res1')(hg)
            feature = eval('self.hg'+str(i)+'_lin1')(ll)
            preds = eval('self.hg'+str(i)+'_conv_pred')(feature)
            out_preds.append(preds)

            # if self.offset_func is not None:
            #     offset = eval('self.offset_' + str(i))(x,hg,preds)
            #     out_offsets.append(offset)

            # if i < self.config.num_stack - 1:
            merge_feature = eval('self.hg'+str(i)+'_conv1')(feature)
            merge_preds = eval('self.hg'+ str(i)+'_conv2')(preds)
            x = x+merge_feature+merge_preds
        
        if self.offset_func is not None:
            offset = self.offset_head(x)

        out_preds = torch.stack(out_preds,dim=1)
        # out_preds size: n_batch,n_stack,n_landmark,heatmap_size,heatmap_size

        if self.offset_func is not None:
            out_offsets = offset.unsqueeze(1)  # torch.stack(out_offsets,dim=1)
        else :
            out_offsets = None

        return out_preds,out_offsets

    def inference(self,x,inference_indice=-1):
        """
            inference_indice: the index of output seconde dimension, choose which one tensor of output.
            default is -1, the last one.
        """
        output,offset = self.forward(x)
        last_offset = offset[:,-1] if offset is not None else None
        last_output = output[:,inference_indice]

        return last_output,last_offset
        

    def init_weights(self,pretrained=''):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
        print("Finish init hourglass weights")

        if os.path.isfile(pretrained) or os.path.islink(pretrained):
            
            pretrained_dict = torch.load(pretrained)
            if not isinstance(pretrained_dict,collections.OrderedDict):    
                # suppose state_dict in pretrained_dict 
                if isinstance(pretrained_dict['state_dict'],collections.OrderedDict):
                    pretrained_dict = pretrained_dict['state_dict']
                else :
                    raise("cannot find the state_dict in {}".format(pretrained))

            print('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys() and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

