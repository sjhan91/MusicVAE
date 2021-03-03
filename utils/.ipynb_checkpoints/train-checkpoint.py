import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from time import time
from utils.dl_utils import *
from utils.loss import *


def flat_train(device, loss_fn, train_loader, val_loader, model, optimizer, temp=1, epochs=100):
    history = {}
    history['train_loss'] = []
    history['train_acc'] = []
    history['val_loss'] = []
    history['val_acc'] = []
    
    encoder, decoder = model
    enc_optimizer, dec_optimizer = optimizer
    
    hidden_size = decoder.hidden_size
    num_hidden = decoder.num_hidden
    output_size = decoder.output_size
    
    enc_scheduler = optim.lr_scheduler.CosineAnnealingLR(enc_optimizer, epochs, eta_min=1e-6)
    dec_scheduler = optim.lr_scheduler.CosineAnnealingLR(dec_optimizer, epochs, eta_min=1e-6)
    
    for i in range(1, epochs+1):
        start_time = time()
        
        train_loss = 0
        train_acc = 0
        
        val_loss = 0
        val_acc = 0
        
        ### train
        encoder.train()
        decoder.train()
        for batch_idx, x_train in enumerate(train_loader):
            x_train = x_train.to(device)
            
            batch_size = x_train.shape[0]
            seq_len = x_train.shape[1]
            
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            
            # encoder
            z, x_train_mu, x_train_std = encoder(x_train)

            # initialize
            h = z.repeat(num_hidden, 1, int(hidden_size/z.shape[1]))
            c = z.repeat(num_hidden, 1, int(hidden_size/z.shape[1]))
            
            x_train_inputs = torch.zeros((batch_size, 1, x_train.shape[2]), device=device)
            x_train_inputs = torch.cat((x_train_inputs, z.unsqueeze(1)), 2)
            x_train_label = torch.zeros(x_train.shape[:-1], device=device) # argmax
            x_train_prob = torch.zeros(x_train.shape, device=device) # prob

            # forward
            for j in range(seq_len):
                label, prob, h, c = decoder(x_train_inputs, h, c, temp=1)

                x_train_label[:, j] = label.squeeze()
                x_train_prob[:, j, :] = prob.squeeze()
                
                # scheduled sampling
                if np.random.binomial(1, inverse_sigmoid(i)):
                    # teacher forcing
                    x_train_inputs = torch.cat((x_train[:, j, :], z), 1).unsqueeze(1)
                else:
                    # sampling
                    label = F.one_hot(label, num_classes=output_size)
                    x_train_inputs = torch.cat((label, z.unsqueeze(1)), 2)
            
            # loss
            beta = kl_annealing(i, 0, 0.2)
            loss = loss_fn(x_train_prob, x_train, x_train_mu, x_train_std, beta)
            
            # backward
            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            
            train_loss += loss.item()
            train_acc += accuracy(x_train, x_train_label).item()
            
        enc_scheduler.step()
        dec_scheduler.step()
        
        train_loss = train_loss / (batch_idx + 1)
        train_acc = train_acc / (batch_idx + 1)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        ### validation
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for batch_idx, x_val in enumerate(val_loader):
                x_val = x_val.to(device)
                
                batch_size = x_val.shape[0]
                seq_len = x_val.shape[1]
                
                # forward encoder
                z, x_val_mu, x_val_std = encoder(x_val)
                
                # initialize
                h = z.repeat(num_hidden, 1, int(hidden_size/z.shape[1]))
                c = z.repeat(num_hidden, 1, int(hidden_size/z.shape[1]))
                
                # full sampling
                x_val_inputs = torch.zeros((batch_size, 1, x_val.shape[2]), device=device)
                x_val_inputs = torch.cat((x_val_inputs, z.unsqueeze(1)), 2)
                x_val_label = torch.zeros(x_val.shape[:-1], device=device) # argmax
                x_val_prob = torch.zeros(x_val.shape, device=device) # prob
                
                # forward
                for j in range(seq_len):
                    label, prob, h, c = decoder(x_val_inputs, h, c, temp=1)
                    
                    x_val_label[:, j] = label.squeeze()
                    x_val_prob[:, j, :] = prob.squeeze()
                    
                    label = F.one_hot(label, num_classes=output_size)
                    x_val_inputs = torch.cat((label, z.unsqueeze(1)), 2)
                
                loss = loss_fn(x_val_prob, x_val, x_val_mu, x_val_std, beta)
                
                val_loss += loss.item()
                val_acc += accuracy(x_val, x_val_label).item()
                
        val_loss = val_loss / (batch_idx + 1)
        val_acc = val_acc / (batch_idx + 1)
        
        history['val_loss'].append(val_loss)
        history['val_acc'] .append(val_acc)
        
        print('Epoch %d (%0.2f sec) - train_loss: %0.3f, train_acc: %0.3f, val_loss: %0.3f, val_acc: %0.3f, lr: %0.6f' % \
             (i, time()-start_time, train_loss, train_acc, val_loss, val_acc, enc_scheduler.get_last_lr()[0]))
        
    return history


def hierarchical_train(device, loss_fn, train_loader, val_loader, model, optimizer, bar_units=16, epochs=100):
    history = {}
    history['train_loss'] = []
    history['train_acc'] = []
    history['val_loss'] = []
    history['val_acc'] = []
    
    encoder, conductor, decoder = model
    enc_optimizer, con_optimizer, dec_optimizer = optimizer
    
    hidden_size = decoder.hidden_size
    num_hidden = decoder.num_hidden
    output_size = decoder.output_size
    
    enc_scheduler = optim.lr_scheduler.CosineAnnealingLR(enc_optimizer, epochs, eta_min=1e-6)
    con_scheduler = optim.lr_scheduler.CosineAnnealingLR(con_optimizer, epochs, eta_min=1e-6)
    dec_scheduler = optim.lr_scheduler.CosineAnnealingLR(dec_optimizer, epochs, eta_min=1e-6)
    
    for i in range(1, epochs+1):
        start_time = time()
        
        train_loss = 0
        train_acc = 0
        
        val_loss = 0
        val_acc = 0
        
        encoder.train()
        conductor.train()
        decoder.train()
        for batch_idx, x_train in enumerate(train_loader):
            x_train = x_train.to(device)
            
            batch_size = x_train.shape[0]
            seq_len = x_train.shape[1]
            
            enc_optimizer.zero_grad()
            con_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            
            # forward
            x_train_z, x_train_mu, x_train_std = encoder(x_train)
            x_train_feat = conductor(x_train_z)
            
            # initialize    
            x_train_inputs = torch.zeros((batch_size, 1, x_train.shape[2]), device=device)
            x_train_label = torch.zeros(x_train.shape[:-1], device=device) # argmax
            x_train_prob = torch.zeros(x_train.shape, device=device) # prob
            
            # teacher forcing
            for j in range(seq_len):
                bar_idx = j // bar_units
                bar_change_idx = j % bar_units
                
                z = x_train_feat[:, bar_idx, :]
                
                # init state
                if bar_change_idx == 0:
                    h = z.repeat(num_hidden, 1, int(hidden_size/z.shape[1]))
                    c = z.repeat(num_hidden, 1, int(hidden_size/z.shape[1]))
                
                label, prob, h, c = decoder(x_train_inputs, h, c, z)
                
                x_train_label[:, j] = label.squeeze()
                x_train_prob[:, j, :] = prob.squeeze()
                
                # teacher forcing
                if np.random.binomial(1, inverse_sigmoid(i)):
                    x_train_inputs = x_train[:, j, :].unsqueeze(1)
                else:
                    x_train_inputs = F.one_hot(label, num_classes=output_size)
            
            beta = kl_annealing(i, 0, 0.2)
            loss = loss_fn(x_train_prob, x_train, x_train_mu, x_train_std, beta)
            
            # backward
            loss.backward()
            enc_optimizer.step()
            con_optimizer.step()
            dec_optimizer.step()
            
            train_loss += loss.item()
            train_acc += accuracy(x_train, x_train_label).item()
            
        enc_scheduler.step()
        con_scheduler.step()
        dec_scheduler.step()
        
        train_loss = train_loss / (batch_idx + 1)
        train_acc = train_acc / (batch_idx + 1)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        encoder.eval()
        conductor.eval()
        decoder.eval()
        with torch.no_grad():
            for batch_idx, x_val in enumerate(val_loader):
                x_val = x_val.to(device)
                
                batch_size = x_val.shape[0]
                seq_len = x_val.shape[1]
                
                # forward
                x_val_z, x_val_mu, x_val_std = encoder(x_val)
                x_val_feat = conductor(x_val_z)
                
                # initialize
                x_val_inputs = torch.zeros((batch_size, 1, x_val.shape[2]), device=device)  
                x_val_label = torch.zeros(x_val.shape[:-1], device=device) # argmax
                x_val_prob = torch.zeros(x_val.shape, device=device) # prob
                
                # full sampling
                for j in range(seq_len):
                    bar_idx = j // bar_units
                    bar_change_idx = j % bar_units
                    
                    z = x_val_feat[:, bar_idx, :]
                
                    # init state
                    if bar_change_idx == 0:
                        h = z.repeat(num_hidden, 1, int(hidden_size/z.shape[1]))
                        c = z.repeat(num_hidden, 1, int(hidden_size/z.shape[1]))
                    
                    label, prob, h, c = decoder(x_val_inputs, h, c, z)

                    x_val_label[:, j] = label.squeeze()
                    x_val_prob[:, j, :] = prob.squeeze()

                    # full sampling
                    x_val_inputs = F.one_hot(label, num_classes=output_size)
            
                loss = loss_fn(x_val_prob, x_val, x_val_mu, x_val_std)
                
                val_loss += loss.item()
                val_acc += accuracy(x_val, x_val_label).item()
                
        val_loss = val_loss / (batch_idx + 1)
        val_acc = val_acc / (batch_idx + 1)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print('Epoch %d (%0.2f sec) - train_loss: %0.3f, train_acc: %0.3f, val_loss: %0.3f, val_acc: %0.3f, lr: %0.6f' % \
             (i, time()-start_time, train_loss, train_acc, val_loss, val_acc, enc_scheduler.get_last_lr()[0]))
        
    return history