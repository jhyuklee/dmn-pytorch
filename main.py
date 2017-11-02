import torch
import argparse
import pickle
import pprint
import numpy as np
from dataset import Dataset, Config
from model import DMN
from run import run_epoch


argparser = argparse.ArgumentParser()
# run settings
argparser.add_argument('--data_path', type=str, default='./data/en/babi.pkl')
argparser.add_argument('--model_name', type=str, default='')
argparser.add_argument('--checkpoint_dir', type=str, default='./results/')
argparser.add_argument('--batch_size', type=int, default=16)
argparser.add_argument('--epoch', type=int, default=10)
argparser.add_argument('--train', type=int, default=1)
argparser.add_argument('--valid', type=int, default=1)
argparser.add_argument('--test', type=int, default=1)
argparser.add_argument('--early_stop', type=int, default=0)
argparser.add_argument('--random_search', type=int, default=0)
argparser.add_argument('--valid_iter', type=int, default=5)
argparser.add_argument('--resume', action='store_true', default=False)
argparser.add_argument('--save', action='store_true', default=False)
argparser.add_argument('--print_step', type=float, default=2000)

# model hyperparameters
argparser.add_argument('--lr', type=float, default=0.00062)
argparser.add_argument('--lr_decay', type=float, default=1.0)
argparser.add_argument('--wd', type=float, default=0)
argparser.add_argument('--rnn_hdim', type=int, default=300)
argparser.add_argument('--rnn_ln', type=int, default=1)
argparser.add_argument('--rnn_dr', type=float, default=0.5)
argparser.add_argument('--word_dr', type=float, default=0.0)
argparser.add_argument('--fc1_dim', type=int, default=400)
argparser.add_argument('--fc1_dr', type=float, default=0.0)
argparser.add_argument('--grad_max_norm', type=int, default=5)
args = argparser.parse_args()


def random_search(config):
    if config.random_search:
        config.lr = np.random.uniform(5e-4, 5e-3) 
        # config.rnn_hdim = 50 * np.random.randint(1, 10)
        pass

    new_params = {
        'lr': '%.5f' % config.lr,
        # 'rnn_hdim': config.t_rnn_hdim,
    }
    pp = lambda x: pprint.PrettyPrinter().pprint(x)
    pp(new_params)
    return new_params


def run_experiment(model, dataset, iter):
    best_metric = np.zeros(2)
    early_stop = False
    if model.config.train:
        if model.config.resume:
            model.load_checkpoint()

        for ep in range(model.config.epoch):
            if early_stop:
                break
            print('- Training Epoch %d' % (ep+1))
            run_epoch(model, dataset, ep, 'tr')

            if model.config.valid:
                print('- Validation')
                met = run_epoch(model, dataset, ep, 'va', False)
                if best_metric[0] < met[0]: # compare MRR
                    best_metric = met
                    model.save_checkpoint({
                        'config': model.config,
                        'state_dict': model.state_dict(),
                        'optimizer': model.optimizer.state_dict()})
                else:
                    # model.decay_lr()
                    if model.config.early_stop:
                        early_stop = True
                        print('\tearly stop applied')
                print('\tbest metrics:\t%s' % ('\t'.join(['{:.2f}'.format(k)
                    for k in best_metric])))

            if model.config.test:
                print('- Testing')
                run_epoch(model, dataset, ep, 'te', False)
            print()
    
    if model.config.test:
        print('- Load Validation/Testing')
        if model.config.resume or (model.config.save and model.config.train):
            model.load_checkpoint()
        run_epoch(model, dataset, 0, 'va', False)
        run_epoch(model, dataset, 0, 'te', False)
        print()

    return best_metric


def main():
    print('### load dataset')
    dataset = pickle.load(open(args.data_path, 'rb'))
    
    # update args
    dataset.config.__dict__.update(args.__dict__)
    args.__dict__.update(dataset.config.__dict__)
    pp = lambda x: pprint.PrettyPrinter().pprint(x)
    pp(args.config.__dict__)

    # new model experiment
    for valid_idx in range(args.valid_iter):
        print('### validation step %d' % (valid_idx+1))
        new_p = random_search(args)

        # build/run model
        model = DMN(args, dataset.widx2vec, valid_idx).cuda()
        results = run_experiment(model, dataset, valid_idx)

        # record results
        record_results(new_p, results)

    print('### end of experiment')

if __name__ == '__main__':
    main()

