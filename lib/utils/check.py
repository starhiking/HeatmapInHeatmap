import os
from lib.utils.log import create_logger

def check_mkdir(str_path):
    paths = str_path.split('/')
    # temp_folder = paths[0]
    temp_folder = ""
    for i in range (len(paths)):
        temp_folder = os.path.join(temp_folder,paths[i])
        if not os.path.exists(temp_folder):
            print("INFO: {} not exist , created.".format(temp_folder))
            os.mkdir(temp_folder)
    
    assert os.path.exists(str_path) , "{} not created success.".format(str_path)
    
def check_pre(config):

    # check model dir
    checkpoint_dir = os.path.join('checkpoints',config.data_type,config.head_type,config.model_dir)
    check_mkdir(checkpoint_dir)
    
    # check log dir
    logger = create_logger(config)

    # check dataset
    dataset_path = os.path.join('data','benchmark',config.data_type.upper())
    assert os.path.exists(os.path.join(dataset_path,config.data_folder)),"Wrong image data folder,don't exist image file"
    assert os.path.exists(os.path.join(dataset_path,config.data_folder+'.txt')),"Wrong image data folder, don't exist label file."

    # check config
    if config.head_type.upper() != 'WOO':
        assert hasattr(config,'criterion_offset')
        assert hasattr(config,'loss_offset_weight')

    return checkpoint_dir,logger
