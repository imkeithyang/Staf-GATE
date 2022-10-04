import numpy as np
import torch
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix

def find_subgraph_common(FC_model, sc, fc, common_edge=None,  k=20):
    if common_edge == None:
        common_edge = set([])
    
    mask = torch.ones(68,68)
    edge_count = 0
    
    # get predicted FC with unMasked SC
    _, test_sc_mu, test_sc_logvar, _ = FC_model(sc)
    test_pred_fc = FC_model.fc_predict(test_sc_mu)
    
    test_pred_fc_mean = test_pred_fc
    test_loss = np.corrcoef(test_pred_fc_mean.detach().numpy(), fc.flatten().detach().numpy())[0,1]
    
    ordered = []
    while edge_count != k:
        prev_loss = 0
        worst_mask_loss = test_loss
        worst_edge = None
        for i in range(len(torch.nonzero(torch.triu(sc.reshape(68,68))))):
            edge = torch.nonzero(torch.triu(sc.reshape(68,68)))[i]
            if tuple(edge.numpy()) not in common_edge:
                continue
            curr_mask = mask.clone()

            # set mask
            n1 = edge[0]
            n2 = edge[1]
            if curr_mask[n1,n2] == 0:
                continue
            curr_mask[n1, n2] = 0
            curr_mask[n2, n1] = 0

            # get Masked SC
            masked_sc = sc*curr_mask.flatten()
            
            # get predicted FC with Masked SC
            _, mask_sc_mu, mask_sc_logvar, _ = FC_model(masked_sc)
            mask_pred_fc = FC_model.fc_predict(mask_sc_mu)
            mask_pred_fc_mean = mask_pred_fc#torch.mean(mask_pred_fc, dim=0)
            net = mask_pred_fc_mean.detach().numpy().reshape(68,68)
            med = np.quantile(abs(net), 0.5)
            net[abs(net)<med] = 0
            g = from_numpy_matrix(net)
            mask_loss = np.corrcoef(mask_pred_fc_mean.detach().numpy(), fc.flatten().detach().numpy())[0,1]

            curr_loss = -(mask_loss - test_loss)
            if curr_loss > prev_loss:
                worst_mask_loss = mask_loss
                prev_loss = curr_loss
                worst_edge = edge
            del curr_mask
        
        # to ensure that removing one more edge is going to
        # remove even more information than the previous mask
        test_loss = worst_mask_loss
        edge_count += 1
        try:
            n1 = worst_edge[0]
            n2 = worst_edge[1]
            mask[n1,n2] = 0
            mask[n2,n1] = 0
            ordered.append(worst_edge.numpy())
        except Exception as e:
            print(e)
            break
            
    return mask, ordered




def find_subgraph_uncommon(FC_model, sc, fc, common_edge=None,  k=20):
    if common_edge == None:
        common_edge = set([])
    
    mask = torch.ones(68,68)
    edge_count = 0
    
    # get predicted FC with unMasked SC
    _, test_sc_mu, test_sc_logvar, _ = FC_model(sc)
    test_pred_fc = FC_model.fc_predict(test_sc_mu)
    
    test_pred_fc_mean = test_pred_fc
    test_loss = np.corrcoef(test_pred_fc_mean.detach().numpy(), fc.flatten().detach().numpy())[0,1]
    
    ordered = []
    while edge_count != k:
        prev_loss = 0
        worst_mask_loss = test_loss
        worst_edge = None
        for i in range(len(torch.nonzero(torch.triu(sc.reshape(68,68))))):
            edge = torch.nonzero(torch.triu(sc.reshape(68,68)))[i]
            if tuple(edge.numpy()) in common_edge:
                continue
            curr_mask = mask.clone()

            # set mask
            n1 = edge[0]
            n2 = edge[1]
            if curr_mask[n1,n2] == 0:
                continue
            curr_mask[n1, n2] = 0
            curr_mask[n2, n1] = 0

            # get Masked SC
            masked_sc = sc*curr_mask.flatten()
            
            # get predicted FC with Masked SC
            _, mask_sc_mu, mask_sc_logvar, _ = FC_model(masked_sc)
            mask_pred_fc = FC_model.fc_predict(mask_sc_mu)
            mask_pred_fc_mean = mask_pred_fc#torch.mean(mask_pred_fc, dim=0)
            net = mask_pred_fc_mean.detach().numpy().reshape(68,68)
            med = np.quantile(abs(net), 0.5)
            net[abs(net)<med] = 0
            g = from_numpy_matrix(net)
            mask_loss = np.corrcoef(mask_pred_fc_mean.detach().numpy(), fc.flatten().detach().numpy())[0,1]

            curr_loss = -(mask_loss - test_loss)
            if curr_loss > prev_loss:
                worst_mask_loss = mask_loss
                prev_loss = curr_loss
                worst_edge = edge
            del curr_mask
        
        # to ensure that removing one more edge is going to
        # remove even more information than the previous mask
        test_loss = worst_mask_loss
        edge_count += 1
        try:
            n1 = worst_edge[0]
            n2 = worst_edge[1]
            mask[n1,n2] = 0
            mask[n2,n1] = 0
            ordered.append(worst_edge.numpy())
        except Exception as e:
            break
            
    return mask, ordered