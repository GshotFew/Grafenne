import torch
import numpy as np
import os
import random
import sys
src_dir = os.path.dirname(os.path.dirname('__file__'))
sys.path.append(src_dir)
from seed import seed


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return 
seed_everything(seed)




def create_otf_edges_sample(node_features,feature_mask):
    assert node_features.shape[1] == feature_mask.shape[1]
    assert node_features.shape[0] == feature_mask.shape[0]
    otf_edge_index = feature_mask.nonzero(as_tuple=True)
    otf_edge_attr = node_features[otf_edge_index[0],otf_edge_index[1]].reshape(otf_edge_index[0].shape[0], -1)
    otf_edge_index = torch.cat((otf_edge_index[0].unsqueeze(0),otf_edge_index[1].unsqueeze(0)), dim=0)
    return otf_edge_index,otf_edge_attr

def create_otf_edges(node_features,feature_mask):
    assert node_features.shape[1] == feature_mask.shape[1]
    assert node_features.shape[0] == feature_mask.shape[0]
    otf_edge_index = feature_mask.nonzero(as_tuple=True)
    otf_edge_attr = node_features[otf_edge_index[0],otf_edge_index[1]].reshape(otf_edge_index[0].shape[0], -1)
    otf_edge_index = torch.cat((otf_edge_index[0].unsqueeze(0),otf_edge_index[1].unsqueeze(0)), dim=0)
    return otf_edge_index,otf_edge_attr

def get_feature_mask(rate, n_nodes, n_features, type="uniform"):
    """ Return mask of shape [n_nodes, n_features] indicating whether each feature is present or missing"""
    if type == "structural":  # either remove all of a nodes features or none
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes)).bool().unsqueeze(1).repeat(1, n_features)
    else:
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes, n_features)).bool()
    
def drop_negative_edges_from_feature_mask(node_features,feature_mask,drop_rate):
    assert node_features.shape[1] == feature_mask.shape[1]
    assert node_features.shape[0] == feature_mask.shape[0]
    otf_edge_index = feature_mask.nonzero()
    otf_edge_attr = node_features[otf_edge_index[:,0],otf_edge_index[:,1]] 
    otf_edge_index_pos = otf_edge_index[otf_edge_attr.bool()!=False]
    otf_edge_index_neg = otf_edge_index[otf_edge_attr.bool()==False]
    neg_indices_to_keep = otf_edge_index_neg[torch.bernoulli(torch.Tensor([1 - drop_rate]).repeat(otf_edge_index_neg.shape[0])).bool()]
    indices_to_keep = torch.cat((otf_edge_index_pos,neg_indices_to_keep),dim=0)
    new_feature_mask = torch.zeros_like(feature_mask).bool()
    new_feature_mask[indices_to_keep[:,0],indices_to_keep[:,1]] = True
    #print(otf_edge_index_pos.shape,neg_indices_to_keep.shape,new_feature_mask.sum())
    return new_feature_mask