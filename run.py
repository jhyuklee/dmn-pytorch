import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
from utils import progress 
from torch.autograd import Variable
from datetime import datetime


def run_epoch(m, d, ep, mode='tr', is_train=True):
    total_metrics = np.zeros(2)
    total_step = 0.0
    print_step = m.config.print_step
    start_time = datetime.now()
    # d.shuffle_data(seed=None, mode='tr') 

    while True:
        m.optimizer.zero_grad()
        inputs, targets = d.get_next_batch(mode=mode, pad=True)
        targets = Variable(torch.LongTensor(np.array(targets)).cuda()) 

        if is_train: m.train()
        else: m.eval()
        outputs = m(inputs)
        loss = m.criterion(outputs, targets)
        metrics = m.get_metrics(outputs, targets)

        if is_train:
            loss.backward()
            nn.utils.clip_grad_norm(m.parameters(), m.config.grad_max_norm)
            m.optimizer.step()

        total_metrics[0] += loss.data[0]
        total_metrics[1] += metrics
        total_step += 1.0
        
        # print step
        if d.get_batch_ptr(mode) % print_step == 0 or total_step == 1:
            et = int((datetime.now() - start_time).total_seconds())
            _progress = progress(d.get_batch_ptr(mode) / d.get_dataset_len(mode))
            if d.get_batch_ptr(mode) == 0:
                _progress = progress(1)
            _progress += '[%s] time: %s' % (
                    '\t'.join(['{:.2f}'.format(k) 
                    for k in total_metrics / total_step]),
                    '{:2d}:{:2d}:{:2d}'.format(et//3600, et%3600//60, et%60))
            sys.stdout.write(_progress)
            sys.stdout.flush()

            # end of an epoch
            if d.get_batch_ptr(mode) == 0:
                et = (datetime.now() - start_time).total_seconds()
                print('\n\ttotal metrics:\t%s' % ('\t'.join(['{:.2f}'.format(k)
                    for k in total_metrics / total_step]))) 
                break

    return total_metrics / total_step

