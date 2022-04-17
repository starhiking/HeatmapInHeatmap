import logging
import os
import sys
sys.path.append('.')
import numpy as np


def create_logger(config,Is_test=False):

    if not os.path.exists('logs'):
        os.mkdir('logs')
    
    log_file = "{}_{}_{}.log".format(config.data_type.upper(),config.head_type.upper(),config.model_dir)

    final_log_file = os.path.join('logs',log_file) if not Is_test else os.path.join('test_logs',log_file)
    if os.path.exists(final_log_file):
        print("Current log file is exist")
        raise("Log file alread exist")

    logging.basicConfig(
        format=
        '[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(final_log_file, mode='w'),
            logging.StreamHandler()
        ])                        
    logger = logging.getLogger()
    
    return logger