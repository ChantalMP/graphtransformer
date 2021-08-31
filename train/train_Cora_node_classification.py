"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl

from train.metrics import accuracy_SBM as accuracy

def train_epoch(model, optimizer, device, graph, epoch):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0

    if model.library == 'dgl':
        features = graph.ndata['feat'].to(device)
        labels = graph.ndata['label'].to(device)
        train_mask = graph.ndata['train_mask'].to(device)
    else:
        features = graph['x'].to(device)
        labels = graph['y'].to(device)
        train_mask = graph['train_mask'].to(device)

    optimizer.zero_grad()
    try:

        lap_pos_enc = graph.ndata['lap_pos_enc'].to(device) if model.library == 'dgl' else graph['ndata']['lap_pos_enc']
        sign_flip = torch.rand(lap_pos_enc.size(1)).to(device)
        sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
        lap_pos_enc = lap_pos_enc * sign_flip.unsqueeze(0)
    except:
        lap_pos_enc = None

    scores = model.forward(graph, features, None, lap_pos_enc)

    loss = model.loss(scores[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    epoch_loss += loss.detach().item()
    epoch_train_acc += accuracy(scores[train_mask], labels[train_mask])
    
    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, device, graph, epoch, split):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    features = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device)

    train_mask = graph.ndata['train_mask'].to(device)
    val_mask = graph.ndata['val_mask'].to(device)
    test_mask = graph.ndata['test_mask'].to(device)
    mask = val_mask if split == 'val' else (test_mask if split == 'test' else train_mask)

    nb_data = 0
    with torch.no_grad():
        try:
            batch_lap_pos_enc = graph.ndata['lap_pos_enc'].to(device)
        except:
            batch_lap_pos_enc = None

        scores = model.forward(graph, features, None, batch_lap_pos_enc)
        loss = model.loss(scores[mask], labels[mask])
        epoch_test_loss += loss.detach().item()
        epoch_test_acc += accuracy(scores[mask], labels[mask])
        
    return epoch_test_loss, epoch_test_acc


