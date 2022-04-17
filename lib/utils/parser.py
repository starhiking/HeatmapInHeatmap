import argparse
from datetime import datetime
import importlib

def train_parser_args():
    """
        parser_args for training
    """
    parser = argparse.ArgumentParser(description='Training infos')

    parser.add_argument('--config_file',type=str,default='experiments/Data_WFLW/HIHC_64x8_hg_l2.py',help="relative path for experiments python file")
    parser.add_argument('--model_dir',type=str,default=None)
    parser.add_argument('--resume_checkpoint',type=str,default=None)
    parser.add_argument('--gpu_id',type=int,default=None)
    parser.add_argument('--local_rank',type=int,help='local current rank')
    parser.add_argument('--offset_fine_tune',action='store_true',default=False,help='fine tune offset')
    parser.add_argument('--transformer',action='store_true',default=False,help='use transformer or not')

    args, unparsed = parser.parse_known_args()
    
    # parsed args
    config_path = args.config_file[:-3].replace('/','.') # remove '.py'
    config = importlib.import_module(config_path).Config()
    setattr(config,'offset_fine_tune',args.offset_fine_tune)
    setattr(config,'transformer',args.transformer)
    if args.model_dir is not None:
        setattr(config,'model_dir',args.model_dir)
    else:
        setattr(config,'model_dir',datetime.now().strftime("%m_%d_%H_%M_%S"))
    if args.resume_checkpoint is not None:
        setattr(config,'resume_checkpoint',args.resume_checkpoint)
    else:
        setattr(config,'resume_checkpoint',False)
        
    if args.gpu_id is not None:
        setattr(config,'gpu_id',args.gpu_id)
    
    # unparsed args
    for up_kv in unparsed:
        up_kv = up_kv.replace('--','').split(':')
        up_kv = up_kv[0].split('=') if len(up_kv) == 1 else up_kv
        assert len(up_kv) == 2 , "unparsed item only support ':' or '=', like --batch_size=8"

        up_key = up_kv[0]
        up_val = up_kv[1]
        assert hasattr(config,up_key), "config file unsupport the item: {}".format(up_key)
        
        # judge type: bool, float, int
        if up_val.upper() == "TRUE":
            up_val = True
        elif up_val.upper() == "FALSE":
            up_val = False
        elif '.' in up_val or 'e' in up_val: # float
            try:
                up_val = float(up_val)
            except:
                pass
        else: # int
            try:
                up_val = int(up_val)
            except:
                print("WARN: Cannot regcongnize the data type of {} ! Process as string type".format(up_val))
        
        setattr(config,up_key,up_val)

    return config

def inference_parser_args():
    """
        parser_args for inference
    """
    parser = argparse.ArgumentParser(description='Inference infos')
    parser.add_argument('--config_file',type=str,required=True,help="same as train")
    parser.add_argument('--resume_checkpoint',type=str,required=True,help="the file path for resume checkpoint")
    parser.add_argument('--test_folder',type=str,default='test',help="Test which test folder, one subset or fullset")
    parser.add_argument('--gpu_id',type=int,default=None)
    parser.add_argument('--inference_indice',type=int,default=-1,help="The index of stacked output, default is last one")

    args, unparsed = parser.parse_known_args()

    config_path = args.config_file[:-3].replace('/','.') # remove '.py'
    config = importlib.import_module(config_path).Config()

    # setattr(config,'transformer',args.transformer)
    setattr(config,'resume_checkpoint',args.resume_checkpoint)
    setattr(config,'test_folder',args.test_folder)
    setattr(config,'model_dir',datetime.now().strftime("%m_%d_%H_%M_%S"))
    setattr(config,'inference_indice',args.inference_indice)

    if args.gpu_id is not None:
        setattr(config,'gpu_id',args.gpu_id)
    

    # unparsed args
    for up_kv in unparsed:
        up_kv = up_kv.replace('-','').split(':')
        up_kv = up_kv[0].split('=') if len(up_kv) == 1 else up_kv
        assert len(up_kv) == 2 , "unparsed item only support ':' or '=', like --batch_size=8"

        up_key = up_kv[0]
        up_val = up_kv[1]
        assert hasattr(config,up_key), "config file unsupport the item: {}".format(up_key)
        
        # judge type: bool, float, int
        if up_val.upper() == "TRUE":
            up_val = True
        elif up_val.upper() == "FALSE":
            up_val = False
        elif '.' in up_val: # float
            try:
                up_val = float(up_val)
            except:
                pass
        else: # int
            try:
                up_val = int(up_val)
            except:
                print("WARN: Cannot regcongnize the data type of {} ! Process as string type".format(up_val))
        
        setattr(config,up_key,up_val)

    return config