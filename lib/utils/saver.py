import os
import torch



def save_checkpoint(model_state_dict,best_nme,optimizer_state_dict,checkpoint_dir):
    
    epoch = best_nme['epoch']
    file_path = os.path.join(checkpoint_dir,"{}_checkpoint.pth".format(str(epoch)))
    best_path = os.path.join(checkpoint_dir,"best.pth")
    torch.save({
        "state_dict":model_state_dict,
        "best_nme":best_nme,
        "optimizer":optimizer_state_dict,
        "epoch":epoch
    },file_path)
    if os.path.islink(best_path):
        os.remove(best_path)
    
    # symlink is create a relative path file : a is exist file and relative path,b is link and absolute path
    os.symlink(os.path.join("./","{}_checkpoint.pth".format(str(epoch))),best_path)