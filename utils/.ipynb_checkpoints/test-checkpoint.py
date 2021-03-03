import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from time import time
from utils.dl_utils import *
from utils.loss import *


def flat_test(device, loss_fn, test_loader, model, temp=1, options='teacher_forcing'):
    history = {}
    history['test_loss'] = []
    history['test_acc'] = []
    
    encoder, decoder = model
    
    start_time = time()

    test_loss = 0
    test_acc = 0
    
    y_true = []
    y_pred = []

    encoder.eval()
    decoder.eval()
    
    num_hidden = decoder.num_hidden
    hidden_size = decoder.hidden_size
    output_size = decoder.output_size
    
    with torch.no_grad():
        for batch_idx, x_test in enumerate(test_loader):
            x_test = x_test.to(device)
            
            batch_size = x_test.shape[0]
            seq_len = x_test.shape[1]

            # forward
            z, x_test_mu, x_test_std = encoder(x_test)
            
            h = z.repeat(num_hidden, 1, int(hidden_size/z.shape[1]))
            c = z.repeat(num_hidden, 1, int(hidden_size/z.shape[1]))
            
            x_test_inputs = torch.zeros((batch_size, 1, x_test.shape[2]), device=device)
            x_test_inputs = torch.cat((x_test_inputs, z.unsqueeze(1)), 2)
            x_test_label = torch.zeros(x_test.shape[:-1], device=device) # argmax
            x_test_prob = torch.zeros(x_test.shape, device=device) # prob
            
            for j in range(seq_len):
                label, prob, h, c = decoder(x_test_inputs, h, c, temp=temp)

                x_test_label[:, j] = label.squeeze()
                x_test_prob[:, j, :] = prob.squeeze()
                
                if options == 'teacher_forcing':
                    x_test_inputs = torch.cat((x_test[:, j, :], z), 1).unsqueeze(1)
                else:
                    label = F.one_hot(label, num_classes=output_size)
                    x_test_inputs = torch.cat((label, z.unsqueeze(1)), 2)

            loss = loss_fn(x_test_prob, x_test, x_test_mu, x_test_std)

            test_loss += loss.item()
            test_acc += accuracy(x_test, x_test_label).item()
            
            if batch_idx % 10000 == 0:
                y_true.append(x_test.data.cpu().numpy())
                y_pred.append(x_test_prob.data.cpu().numpy())

    test_loss = test_loss / (batch_idx + 1)
    test_acc = test_acc / (batch_idx + 1)

    history['test_loss'].append(test_loss)
    history['test_acc'] .append(test_acc)

    print('(%0.2f sec) - test_loss: %0.3f, test_acc: %0.3f' % (time()-start_time, test_loss, test_acc))
        
    return history, np.vstack(y_true), np.vstack(y_pred)


def hierarchical_test(device, loss_fn, test_loader, model, temp=1, bar_units=16, options='teacher_forcing'):
    history = {}
    history['test_loss'] = []
    history['test_acc'] = []
    
    encoder, conductor, decoder = model
    
    start_time = time()

    test_loss = 0
    test_acc = 0
    
    y_true = []
    y_pred = []

    encoder.eval()
    conductor.eval()
    decoder.eval()
    
    num_hidden = decoder.num_hidden
    hidden_size = decoder.hidden_size
    output_size = decoder.output_size

    with torch.no_grad():
        for batch_idx, x_test in enumerate(test_loader):
            x_test = x_test.to(device)
            batch_size = x_test.shape[0]
            seq_len = x_test.shape[1]

            # forward
            x_test_z, x_test_mu, x_test_std = encoder(x_test)
            x_test_feat = conductor(x_test_z)
            
            # initialize
            x_test_inputs = torch.zeros((batch_size, 1, x_test.shape[2]), device=device)
            x_test_label = torch.zeros(x_test.shape[:-1], device=device) # argmax
            x_test_prob = torch.zeros(x_test.shape, device=device) # prob
            
            for j in range(seq_len):
                bar_idx = j // bar_units
                bar_change_idx = j % bar_units

                z = x_test_feat[:, bar_idx, :]

                # init state
                if bar_change_idx == 0:
                    h = z.repeat(num_hidden, 1, int(hidden_size/z.shape[1]))
                    c = z.repeat(num_hidden, 1, int(hidden_size/z.shape[1]))

                label, prob, h, c = decoder(x_test_inputs, h, c, z)

                x_test_label[:, j] = label.squeeze()
                x_test_prob[:, j, :] = prob.squeeze()
                
                if options == 'teacher_forcing':
                    x_test_inputs = x_test[:, j, :].unsqueeze(1)
                else:
                    x_test_inputs = F.one_hot(label, num_classes=output_size)

            loss = loss_fn(x_test_prob, x_test, x_test_mu, x_test_std)

            test_loss += loss.item()
            test_acc += accuracy(x_test, x_test_label).item()
            
            if batch_idx % 10000 == 0:
                y_true.append(x_test.data.cpu().numpy())
                y_pred.append(x_test_prob.data.cpu().numpy())

    test_loss = test_loss / (batch_idx + 1)
    test_acc = test_acc / (batch_idx + 1)

    history['test_loss'].append(test_loss)
    history['test_acc'] .append(test_acc)

    print('(%0.2f sec) - test_loss: %0.3f, test_acc: %0.3f' % (time()-start_time, test_loss, test_acc))
        
    return history, np.vstack(y_true), np.vstack(y_pred)