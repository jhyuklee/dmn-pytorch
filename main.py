import torch
import argparse
import pickle
import pprint
import numpy as np
import os
from dataset import Dataset, Config
from model import DMN
from run import run_epoch


argparser = argparse.ArgumentParser()
# run settings
argparser.add_argument('--data_path', type=str, default='./data/babi(tmp).pkl')
argparser.add_argument('--model_name', type=str, default='m')
argparser.add_argument('--checkpoint_dir', type=str, default='./results/')
argparser.add_argument('--batch_size', type=int, default=32)
argparser.add_argument('--epoch', type=int, default=100)
argparser.add_argument('--train', type=int, default=1)
argparser.add_argument('--valid', type=int, default=1)
argparser.add_argument('--test', type=int, default=1)
argparser.add_argument('--early_stop', type=int, default=0)
argparser.add_argument('--resume', action='store_true', default=False)
argparser.add_argument('--save', action='store_true', default=False)
argparser.add_argument('--print_step', type=float, default=128)

# model hyperparameters
argparser.add_argument('--lr', type=float, default=0.0003)
argparser.add_argument('--lr_decay', type=float, default=1.0)
argparser.add_argument('--wd', type=float, default=0)
argparser.add_argument('--grad_max_norm', type=int, default=5)
argparser.add_argument('--s_rnn_hdim', type=int, default=100)
argparser.add_argument('--s_rnn_ln', type=int, default=1)
argparser.add_argument('--s_rnn_dr', type=float, default=0.0)
argparser.add_argument('--q_rnn_hdim', type=int, default=100)
argparser.add_argument('--q_rnn_ln', type=int, default=1)
argparser.add_argument('--q_rnn_dr', type=float, default=0.0)
argparser.add_argument('--e_cell_hdim', type=int, default=100)
argparser.add_argument('--m_cell_hdim', type=int, default=100)
argparser.add_argument('--a_cell_hdim', type=int, default=100)
argparser.add_argument('--word_dr', type=float, default=0.2)
argparser.add_argument('--g1_dim', type=int, default=500)
argparser.add_argument('--max_episode', type=int, default=10)
argparser.add_argument('--beta_cnt', type=int, default=10)
argparser.add_argument('--set_num', type=int, default=1)
argparser.add_argument('--max_alen', type=int, default=2)
args = argparser.parse_args()


def run_experiment(model, dataset, set_num):
    best_metric = np.zeros(2)
    early_stop = False
    if model.config.train:
        if model.config.resume:
            model.load_checkpoint()

        for ep in range(model.config.epoch):
            if early_stop:
                break
            print('- Training Epoch %d' % (ep+1))
            run_epoch(model, dataset, ep, 'tr', set_num)

            if model.config.valid:
                print('- Validation')
                met = run_epoch(model, dataset, ep, 'va', set_num, False)
                if best_metric[1] < met[1]:
                    best_metric = met
                    model.save_checkpoint({
                        'config': model.config,
                        'state_dict': model.state_dict(),
                        'optimizer': model.optimizer.state_dict()})
                    if best_metric[1] == 100:
                        break
                else:
                    # model.decay_lr()
                    if model.config.early_stop:
                        early_stop = True
                        print('\tearly stop applied')
                print('\tbest metrics:\t%s' % ('\t'.join(['{:.2f}'.format(k)
                    for k in best_metric])))

            if model.config.test:
                print('- Testing')
                run_epoch(model, dataset, ep, 'te', set_num, False)
            print()
    
    if model.config.test:
        print('- Load Validation/Testing')
        if model.config.resume or model.config.train:
            model.load_checkpoint()
        run_epoch(model, dataset, 0, 'va', set_num, False)
        run_epoch(model, dataset, 0, 'te', set_num, False)
        print()

    return best_metric


def main():
    if not os.path.exists('./results'):
        os.makedirs('./results')

    print('### load dataset')
    dataset = pickle.load(open(args.data_path, 'rb'))
    
    # update args
    dataset.config.__dict__.update(args.__dict__)
    args.__dict__.update(dataset.config.__dict__)
    pp = lambda x: pprint.PrettyPrinter().pprint(x)
    pp(args.__dict__)

    # new model experiment
    for set_num in range(args.set_num, args.set_num+1):
        print('\n[QA set %d]' % (set_num))
        model = DMN(args, dataset.idx2vec, set_num).cuda()
        results = run_experiment(model, dataset, set_num)

    print('### end of experiment')

if __name__ == '__main__':
    main()

