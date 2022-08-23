import os
import sys
import torch
from torch_geometric.loader import NeighborSampler
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GINConv,SAGEConv, GATConv
import tqdm
import torch.nn.functional as F
import numpy as np
import random
import copy
import argparse
src_dir = os.path.dirname(os.path.dirname('__file__'))
sys.path.append(src_dir)
from utils.data_loader import load_data
from utils.utils import seed_everything,create_otf_edges,get_feature_mask
from models.fognn import ScalableFOGNN as FOGNN
import gc 
import argparse, parser

parser = argparse.ArgumentParser()


parser.add_argument("--data", help="name of the dataset",
                    type=str)
parser.add_argument("--gpu",help="GPU no. to use, -1 in case of no gpu", type=int)

args = parser.parse_args()

gpu = int(args.gpu)
dataset_name = args.data
categorical = True
verbose = True
num_layers = 2
bs_train_nbd = 512
bs_test_nbd = -1
drop_rate = 0.2

device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
data = load_data(dataset_name,train_ratio=0.4,val_ratio=0.3)
print("train dataset, val dataset and test dataset ", data.train_mask.sum(),data.val_mask.sum(),data.test_mask.sum())
    

#batch_size=256 is good
if bs_train_nbd == -1:
    bs_train_nbd = data.x.shape[0]

if bs_test_nbd == -1:
    bs_test_nbd = data.x.shape[0]
    

num_communities = len(set(data.y.numpy().tolist()))
print(f"Node Feature Matrix Info: # Nodes: {data.x.shape[0]}")
print(f"Node Feature Matrix Info: # Node Features: {data.x.shape[1]}")
print(f"Edge Index Shape: {data.edge_index.shape}")
print(f"Edge Weight: {data.edge_attr}")
print(f"# Labels/classes: {num_communities}")

def create_multiple_copies_data_feature_sampling_prob(data,feature_sample_size=100):
    num_features = data.x.shape[1]
    feature_list = list(range(num_features))
    print(len(feature_list))
    datax = [data.x.clone()]
    
    for i in range(0, 10):
        
        feats_available = random.sample(feature_list,feature_sample_size)
        missing_features = list(set(feature_list)-set(feats_available))
        tp = data.x.clone()
        tp[:,missing_features] = 0
        print(data.x.sum(),tp.sum(),tp[:,feats_available].sum(),tp[:,missing_features].sum())
        datax.append(tp)
    return datax

def create_multiple_copies_data_feature_sampling(data, num_time_steps=50, node_select_prob=0.1, feat_update_rate=0, feat_add_delete_rate = 0):
    datax = [(data.x.clone(),list(range(data.x.shape[0])))]
    tp = data.x.clone() #### initialized with all features
    for i in range(0, num_time_steps):
        expected_node_change = int(data.y.shape[0]*node_select_prob) # random.choices(list, k=3)#random.choice(sequence) 
        expected_feat_change = int(data.x.shape[1]*feat_update_rate)
        nodes_to_update = random.sample(list(range(0,data.x.shape[0])), expected_node_change)
        for node_id in nodes_to_update:
            feat_to_update = random.sample(list(range(0,data.x.shape[1])), expected_feat_change)
            for feat_id in feat_to_update:
                del_or_update = random.uniform(0,1)
                tp[node_id][feat_id] = 0
                if(del_or_update < feat_add_delete_rate): #### add this feature back
                    tp[node_id][feat_id] = data.x[node_id][feat_id]
                else:  ### Delete this feature please
                    tp[node_id][feat_id] = 0      
        print(data.x.sum(),tp.sum())
        datax.append((tp.clone(),nodes_to_update))
    return datax

 

    

steps = 20
nodes_prob = 0.03
del_feat_prob = 0.4

datax = create_multiple_copies_data_feature_sampling(data,steps, nodes_prob,del_feat_prob) #vary and play
num_samples = [20,15]
print("bs_train_nbd and test_nbd", bs_train_nbd,bs_test_nbd)
train_neigh_sampler_oracle = NeighborSampler(
        data.edge_index, node_idx= data.train_mask ,   
        sizes=num_samples, batch_size=bs_train_nbd, shuffle=True, num_workers=0)

subgraph_loader = NeighborSampler(
    data.edge_index, node_idx=None,
    sizes=[-1,-1], batch_size=bs_test_nbd, shuffle=False, num_workers=0)



Y = data.y.squeeze().to(device)
obs_features = torch.ones(data.x.shape[0],data.x.shape[1],dtype=torch.double).to(device) 
print(obs_features.shape)
feat_features = np.eye(data.x.shape[1])
feat_features = torch.tensor(feat_features,dtype=torch.double).to(device)
print(feat_features.shape)



def train(model,optimizer,train_neigh_sampler,obs_features,feature_mask,feat_features,X,Y,num_layers=2):
    numPosSamples =64
    model.train()
    total_loss =total_correct= total_computed=0
    for batch_size, n_id, adjs in train_neigh_sampler:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out,a,b = model(obs_features=obs_features[n_id],feature_mask = feature_mask[n_id],
            feat_features=feat_features,obs_adjs = adjs,data_x = X[n_id],num_layers=num_layers) 
        if bs_train_nbd == data.x.shape[0]:  #### whole batch is coming as out
            p = torch.Tensor([1/batch_size*1.0]*batch_size)
            sampledPosIndex = p.multinomial(num_samples=numPosSamples, replacement=False)
            newMask = torch.Tensor([False]*batch_size)
            newMask = newMask.to(torch.bool)
            newMask[sampledPosIndex]=True
            loss = F.nll_loss(out[newMask], Y[n_id[:batch_size]][newMask])
        else:
            loss = F.nll_loss(out, Y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(Y[n_id[:batch_size]]).sum())
        total_computed+= out.shape[0]

    loss = total_loss / len(train_neigh_sampler)
    approx_acc = total_correct / total_computed
    return model,optimizer,loss,approx_acc
    

def get_fisher(model,optimizer,train_importance_sampler,obs_features,feature_mask,feat_features,X,Y,num_layers=2, ):
    
    fisher_dict = {}
    optpar_dict = {}

    numPosSamples =64
    model.train()
    total_loss =total_correct= total_computed=0
    optimizer.zero_grad()
    
    for batch_size, n_id, adjs in train_importance_sampler:
        adjs = [adj.to(device) for adj in adjs]
        out,a,b = model(obs_features=obs_features[n_id],feature_mask = feature_mask[n_id],
            feat_features=feat_features,obs_adjs = adjs,data_x = X[n_id],num_layers=num_layers) 
        if bs_train_nbd == data.x.shape[0]:  #### whole batch is coming as out
            p = torch.Tensor([1/batch_size*1.0]*batch_size)
            sampledPosIndex = p.multinomial(num_samples=numPosSamples, replacement=False)
            newMask = torch.Tensor([False]*batch_size)
            newMask = newMask.to(torch.bool)
            newMask[sampledPosIndex]=True
            loss = F.nll_loss(out[newMask], Y[n_id[:batch_size]][newMask])
        else:
            loss = F.nll_loss(out, Y[n_id[:batch_size]])
        loss.backward()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(Y[n_id[:batch_size]]).sum())
        total_computed+= out.shape[0]

    loss = total_loss / len(train_importance_sampler)
    approx_acc = total_correct / total_computed
    
    for name, param in model.named_parameters():
       
        if(param.grad is not None):
            optpar_dict[name] = param.data.clone()
            fisher_dict[name] = param.grad.data.clone().pow(2)
        else:
            pass
            # print('none param ', name)
    
    
    return optpar_dict, fisher_dict #model,optimizer,loss,approx_acc
    
    
    
def train_mix(model,optimizer,train_neigh_sampler_ft, train_neigh_sampler_ct, lambda_cur, lambda_res,obs_features,feature_mask,feat_features,X,Y,num_layers=2,fisher_dict=None, optpar_dict = None, ewc_lambda = 10):
    numPosSamples =64
    model.train()
    total_loss =total_correct= total_computed=0
    optimizer.zero_grad()
    
    
    loss = 0 
    
    for batch_size, n_id, adjs in train_neigh_sampler_ft:
        adjs = [adj.to(device) for adj in adjs]
        
        out,a,b = model(obs_features=obs_features[n_id],feature_mask = feature_mask[n_id],
            feat_features=feat_features,obs_adjs = adjs,data_x = X[n_id],num_layers=num_layers) 

        loss += lambda_cur*F.nll_loss(out, Y[n_id[:batch_size]])
        
    for name, param in model.named_parameters():
        
        if name in fisher_dict:
            fisher = fisher_dict[name]
            optpar = optpar_dict[name]
            loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda
    
    loss.backward()
     
    optimizer.step()
    return model,optimizer,None,None


def test(model,optimizer,subgraph_loader,obs_features,feature_mask,feat_features,X,Y,num_layers=2):
    with torch.no_grad():
        model.eval()
        outs = []
        for batch_size, n_id, adjs in subgraph_loader:
            adjs = [adj.to(device) for adj in adjs]
            out,a,b = model(obs_features=obs_features[n_id],feature_mask = feature_mask[n_id],
                feat_features=feat_features,obs_adjs = adjs,data_x = X[n_id],num_layers=num_layers)
            outs.append(out)
            del a,b

        out = torch.cat(outs, dim=0)
        del outs
        train_acc = int(out.argmax(dim=-1).eq(Y)[data.train_mask].sum())*100.0/(data.train_mask.sum().item())
        val_acc = int(out.argmax(dim=-1).eq(Y)[data.val_mask].sum())*100.0/(data.val_mask.sum().item())
        test_acc = int(out.argmax(dim=-1).eq(Y)[data.test_mask].sum())*100.0/(data.test_mask.sum().item())
    return train_acc,val_acc,test_acc
    #int(data.train_mask.sum())
    

def add_element_in_buffer(memory,item,buffer_size,current_len):
    if item in memory:
        return memory
    if len(memory) < buffer_size:
        memory.append(item)
    else:
        index = random.randrange(current_len)
        if index < buffer_size:
            memory[index]=item
    return memory
    

    

model_ft = FOGNN(drop_rate=drop_rate, num_obs_node_features=data.num_node_features,
    num_feat_node_features=data.num_node_features,
    num_layers=2, hidden_size=256, out_channels=num_communities,heads=4,categorical=categorical,device=device)
model_ft = model_ft.to(device).double()
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.0001, weight_decay = 0.0001)

model_ct = FOGNN(drop_rate=drop_rate, num_obs_node_features=data.num_node_features,
    num_feat_node_features=data.num_node_features,
    num_layers=2, hidden_size=256, out_channels=num_communities,heads=4,categorical=categorical,device=device)
model_ct = model_ct.to(device).double()
optimizer_ct = torch.optim.Adam(model_ct.parameters(), lr=0.0001, weight_decay = 0.0001)

memory = []

current_stream_length = 0

mem_size = 50
memory_nodes_importance = []# datax[0][1]# []#  datax[0][1] #train_nodes
for node in  datax[0][1]:
        if data.train_mask[node]==True:
            memory_nodes_importance.append(node)
sample_initial_from_memory_node_importance = random.sample(memory_nodes_importance, mem_size)


current_stream_length = 0

mem_size = 50
memory_nodes_importance = []# datax[0][1]# []#  datax[0][1] #train_nodes

for node in  datax[0][1]:
        if data.train_mask[node]==True:
            memory_nodes_importance.append(node)
original_sample_initial_from_memory_node_importance = random.sample(memory_nodes_importance, mem_size)


fisher = {}
optpar = {}

ewc_lambda = 10000

for i in range(0, len(datax)):
    print("Training start at time ", i , "------------>")
    
    model_oracle = FOGNN(drop_rate=drop_rate, num_obs_node_features=data.num_node_features,
        num_feat_node_features=data.num_node_features,
        num_layers=2, hidden_size=256, out_channels=num_communities,heads=4,categorical=categorical,device=device)
    model_oracle = model_oracle.to(device).double()
    optimizer_oracle = torch.optim.Adam(model_oracle.parameters(), lr=0.0001, weight_decay = 0.0001)

    X = datax[i][0].to(device)
    nodes_changed = datax[i][1]
    feature_mask = X > 0   ### only in case values are categorical 
    actual_test_acc = 0
    best_val_acc = 0
    best_epoch = 0
    num_epochs_oracle = 300
    
    num_epochs_ft = 200
    num_epochs_ct = 200
    
    best_oracle_val = -1000000
    best_oracle_test = -1000000
    for epoch in range(0,num_epochs_oracle):
        model_oracle,optimizer_oracle,loss,approx_acc = train(model_oracle,optimizer_oracle,train_neigh_sampler_oracle,obs_features,feature_mask,feat_features,X,Y,num_layers=2)
        torch.cuda.empty_cache()
        gc.collect()
        train_acc,val_acc,test_acc = test(model_oracle,optimizer_oracle,subgraph_loader,obs_features,feature_mask,feat_features,X,Y,num_layers=2)
        if(val_acc> best_oracle_val):
            best_oracle_val = val_acc
            best_oracle_test = test_acc

    print(f't={i}:ORACLE_ACC={best_oracle_test}')
    
    gc.collect()
    
    model_oracle.zero_grad()
    model_ft.zero_grad()
    
        
    train_nodes = []
    for node in nodes_changed:
        if data.train_mask[node]==True:
            train_nodes.append(node)
    
    
    train_nodes_ft = [item for item in train_nodes]
    train_nodes_ct = list(set(memory))
    
    for node in train_nodes_ft:
        current_stream_length += 1
        memory = add_element_in_buffer(memory,node,100,current_stream_length)
        
    train_nodes_ft = torch.LongTensor(train_nodes_ft).to(device)     
    train_nodes_ct = torch.LongTensor(train_nodes_ct).to(device)  
    
    print('train_nodes_ft ', len(train_nodes_ft))
    
    train_neigh_sampler_ft = NeighborSampler(
        data.edge_index, node_idx= train_nodes_ft ,   
        sizes=num_samples, batch_size=bs_train_nbd, shuffle=True, num_workers=0)

    if i>0:
        train_neigh_sampler_ct = NeighborSampler(
        data.edge_index, node_idx= train_nodes_ct ,   
        sizes=num_samples, batch_size=bs_train_nbd, shuffle=True, num_workers=0)

    print("Fine tuning an online model")
    
    for epoch in range(num_epochs_ft):
        model_ft,optimizer_ft,loss,approx_acc = train(model_ft,optimizer_ft,train_neigh_sampler_ft,obs_features,feature_mask,feat_features,X,Y,num_layers=2)
        torch.cuda.empty_cache()
        gc.collect()
    train_acc,val_acc,test_acc = test(model_ft,optimizer_ft,subgraph_loader,obs_features,feature_mask,feat_features,X,Y,num_layers=2)
    gc.collect()
    print(f't={i}:FT_ACC={test_acc}')

    if i==0:
        model_ct =  copy.deepcopy(model_oracle)
        model_ft =  copy.deepcopy(model_oracle)
        optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.0001, weight_decay = 0.0001)
        optimizer_ct = torch.optim.Adam(model_ct.parameters(), lr=0.0001, weight_decay = 0.0001)

        
    if i >0:
        lambda_cur=1
        lambda_res=2
        
        
        sample_initial_from_memory_node_importance = list(set(original_sample_initial_from_memory_node_importance) - set(nodes_changed )  ) 
        
        train_nodes_importance = torch.LongTensor(sample_initial_from_memory_node_importance).to(device)  

        train_neigh_sampler_mem_imp = NeighborSampler(
        data.edge_index, node_idx= train_nodes_importance ,   
        sizes=num_samples, batch_size=bs_train_nbd, shuffle=True, num_workers=0)

        optpar_dict, fisher_dict = get_fisher(model_ct,optimizer_ct,train_neigh_sampler_mem_imp,obs_features,feature_mask,feat_features,X,Y,num_layers=2)        
    
    
        for epoch in range(num_epochs_ct):
            model_ct,optimizer_ct,_,_ = train_mix(model_ct,optimizer_ct,train_neigh_sampler_ft, train_neigh_sampler_ct, lambda_cur, lambda_res,obs_features,feature_mask,feat_features,X,Y,num_layers=2, fisher_dict=fisher_dict, optpar_dict = optpar_dict, ewc_lambda = ewc_lambda)
            
        train_acc,val_acc,test_acc = test(model_ct,optimizer_ct,subgraph_loader,obs_features,feature_mask,feat_features,X,Y,num_layers=2)
    
        print(f't={i}:CONT_ACC={test_acc}')
        
    
    
