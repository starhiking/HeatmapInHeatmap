import torch


def decode_woo_head(target_maps,offset_map=None):
    """
        Args:
            target_maps (n,98,64,64) tensor float32
            offset_map is None here.

        return : 
            preds (n,98,2)
    """
    max_v,idx = torch.max(target_maps.view(target_maps.size(0),target_maps.size(1),target_maps.size(2)*target_maps.size(3)), 2)
    preds = idx.view(idx.size(0),idx.size(1),1).repeat(1,1,2).float()
    max_v = max_v.view(idx.size(0),idx.size(1),1)
    pred_mask = max_v.gt(0).repeat(1, 1, 2).float()

    preds[..., 0].remainder_(target_maps.size(3))
    preds[..., 1].div_(target_maps.size(2)).floor_()

    preds.mul_(pred_mask)
    return preds

def decode_afc_head(target_maps,offset_float):
    """
        target_maps : (n,98,64,64) float32
        offset_float : (n,98,2) float32
    """
    
    preds = decode_woo_head(target_maps).float()
    preds.add_(offset_float)
     
    return preds

def decode_hih_head(target_maps,offset_map):
    """
        target_maps : (n,98,64,64) float32
        offset_map : (n,98,4,4) float32

        return preds (n,98,2)
    """
    preds = decode_woo_head(target_maps).float()
    offsets = decode_woo_head(offset_map) / torch.tensor([offset_map.size(3),offset_map.size(2)],dtype=torch.float32).cuda()
    preds.add_(offsets)
    
    return preds

def decode_od_head(target_maps,offset_map):
    """
        target_maps : (n,98,64,64) float32
        offset_map : (n,2,64,64) float32
    """
    preds = decode_woo_head(target_maps).float()

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            pX, pY = int(preds[i, j, 0]), int(preds[i, j, 1])
            offset_value = torch.FloatTensor([offset_map[i,0,pY,pX],offset_map[i,1,pY,pX]]).cuda()
            preds[i,j].add_(offset_value)

    return preds


