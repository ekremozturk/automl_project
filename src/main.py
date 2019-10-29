import os
import argparse
import logging
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary

from datasets import K49
from datasets import KMNIST

from cnn import torchModel, dartsModel
import darts_train_search as darts
from bohb_worker import run_BOHB, eval_BOHB
import genotypes
import utils
import visualize

import pickle
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

def main(config,
         genotype,
         data_dir,
         num_epochs=10,
         batch_size=50,
         data_augmentations=None,
         save_model_str=None,
         exp_dir = None):
    """
    Training loop for configurableNet.
    :param model_config: network config (dict)
    :param data_dir: dataset path (str)
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param train_criterion: Which loss to use during training (torch.nn._Loss)
    :param model_optimizer: Which model optimizer to use during trainnig (torch.optim.Optimizer)
    :param data_augmentations: List of data augmentations to apply such as rescaling.
        (list[transformations], transforms.Composition[list[transformations]], None)
        If none only ToTensor is used
    :return:
    """

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if data_augmentations is None:
        # You can add any preprocessing/data augmentation you want here
        data_augmentations = transforms.ToTensor()
    elif isinstance(type(data_augmentations), list):
        data_augmentations = transforms.Compose(data_augmentations)
    elif not isinstance(data_augmentations, transforms.Compose):
        raise NotImplementedError

    train_dataset = K49(data_dir, True, data_augmentations)
    #train_dataset = KMNIST(data_dir, True, data_augmentations)
    
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.9 * num_train))

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train])

    # Make data batch iterable
    # Could modify the sampler to not uniformly random sample

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    validation_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=validation_sampler, num_workers=2)

    ########################################################################################################################
    model_config = {
        'n_cells': config['n_cells'],
        'init_channels': config['init_channels'],
        'drop_path_prob': config['drop_path_prob'],
        'n_hidden_layers': 1,#config['n_hidden_layers'],
        'n_hidden_units': 256,#config['n_hidden_units'],
        'dropout_ratio': 0.2#config['dropout_ratio'],
        }

    
    model = dartsModel(genotype,
                       model_config,
                       input_shape=(train_dataset.channels, train_dataset.img_rows, train_dataset.img_cols),
                       num_classes=train_dataset.n_classes).to(device)
    '''
    model = torchModel(model_config,
                       input_shape=(train_dataset.channels, train_dataset.img_rows, train_dataset.img_cols),
                       num_classes=train_dataset.n_classes).to(device)
    '''
    ########################################################################################################################

    total_model_params = sum(p.numel() for p in model.parameters())
    
    # instantiate optimizer
    optimizer = None

    weight_decay = 0.0
    if(config['weight_decay_bool']):
        weight_decay = config['weight_decay']

    lr = config['lr']

    if(config['optimizer'] == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
    if(config['optimizer'] == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum = config['sgd_momentum'], weight_decay = weight_decay)
    
    # instantiate training criterion
    _, occurences = np.unique(train_dataset.labels, return_counts = True)
    class_weights = torch.FloatTensor(1/occurences).to(device)
    train_criterion = torch.nn.CrossEntropyLoss().to(device)

    logging.info('Generated Network:')
    summary(model, (train_dataset.channels,
                    train_dataset.img_rows,
                    train_dataset.img_cols),
            device='cuda' if torch.cuda.is_available() else 'cpu')

    history = {'training': {'loss': list(), 'acc': list()}, 
               'validation': {'loss': list(), 'acc': list()}}

    # Train the model
    for epoch in range(num_epochs):
        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        model.drop_path_prob = model_config['drop_path_prob'] * epoch / num_epochs

        score, loss = model.train_fn(optimizer, train_criterion, train_loader, device)
        logging.info('Training finished | loss: %f | acc: %f \n' % (loss, score))
        history['training']['loss'].append(loss)
        history['training']['acc'].append(score)

        score, loss = model.eval_fn(validation_loader, device, criterion = train_criterion)
        logging.info('Validation finished | loss: %f | acc: %f \n' % (loss, score))
        history['validation']['loss'].append(loss)
        history['validation']['acc'].append(score)

    if save_model_str:
        # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
        if os.path.exists(save_model_str):
            save_model_str += '_'.join(time.ctime())
        torch.save(model.state_dict(), save_model_str)

    history_res = exp_dir+'/history'
    model_res = exp_dir+'/model'

    with open(model_res, 'wb') as f:
        pickle.dump(model, f)

    with open(history_res, 'wb') as f:
        pickle.dump(history, f)
        

def test(model_res, batch_size = 96):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(model_res, 'rb') as f:
            model = pickle.load(f)
    data_augmentations = transforms.ToTensor()
    test_dataset = K49('../data', False, data_augmentations)
    #test_dataset = KMNIST('../data', False, data_augmentations)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    score, loss = model.eval_fn(test_loader, device, criterion = criterion)
    plt.show()


if __name__ == '__main__':

    cmdline_parser = argparse.ArgumentParser('AutoML SS19 final project')

    cmdline_parser.add_argument('-e', '--epochs',
                                default=10,
                                help='Number of epochs',
                                type=int)
    cmdline_parser.add_argument('-b', '--batch_size',
                                default=96,
                                help='Batch size',
                                type=int)
    cmdline_parser.add_argument('-D', '--data_dir',
                                default='../data',
                                help='Directory in which the data is stored (can be downloaded)')
    cmdline_parser.add_argument('-m', '--model_path',
                                default=None,
                                help='Path to store model',
                                type=str)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    cmdline_parser.add_argument('-M', '--methods',
                                default='BOTH',
                                choices=['BOHB', 'DARTS', 'BOTH', 'NONE'],
                                help='Method to use')
    cmdline_parser.add_argument('-g', '--genotype',
                                default='DEFAULT',
                                help='If not using DARTS, specify genotype to use')
    cmdline_parser.add_argument('-t', '--test',
                                action='store_true',
                                default=False,
                                help='Whether to run final test')
    cmdline_parser.add_argument('-c', '--config',
                                default='default',
                                choices=['default', 'final'],
                                help='config')

    # Args for BOHB

    cmdline_parser.add_argument('-bw', '--b_warmstart',
                                action='store_true',
                                default=False,
                                help='Whether to run BOHB with warmstarting')
    cmdline_parser.add_argument('-bni', '--b_n_iters',
                                default=6,
                                help='Number of configurations to try out',
                                type=int)
    cmdline_parser.add_argument('-bd', '--b_dataset',
                                default='K49',
                                choices=['K49', 'KMNIST'],
                                help='Dataset to use if you are using without warmstarting')    

    args, unknowns = cmdline_parser.parse_known_args()

    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl, stream=sys.stdout)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')
    
    exp_dir = 'experiment-{}-{}'.format(args.methods, datetime.now().strftime("%Y%m%d-%H%M%S%f"))
    utils.create_exp_dir(exp_dir)

    genotype = config = None

    if(args.methods == 'DARTS' or args.methods == 'BOTH'):
        logging.info('\n###### NAS w/ DARTS ######\n')
        start = time.time()
        darts.main(exp_dir) 
        architecture_res = exp_dir+'/arch'
        with open(architecture_res, 'rb') as f:
            genotype = pickle.load(f)
        end = time.time()
        logging.info('\nTime elapsed for DARTS: %.0f sec\n', (end-start))
    else:
        genotype = eval(str("genotypes."+args.genotype))

    visualize.plot(genotype.normal, exp_dir+"/normal")
    visualize.plot(genotype.reduce, exp_dir+"/reduction")

    if(args.methods == 'BOHB' or args.methods == 'BOTH'):
        logging.info('\n###### HPO w/ BOHB ######\n')
        n_iter = args.b_n_iters
        if(args.b_warmstart):
            start = time.time()
            bohb_result_file = os.path.join(exp_dir, 'bohb_result_low_budget.pkl')
            run_BOHB(exp_dir, bohb_result_file, 2*n_iter, 1, 9, genotype)
            end = time.time()
            logging.info('Time elapsed for low budget: %.0f sec\n', (end-start))

            time.sleep(5)

            # WARMSTARTING
            bohb_result_file = os.path.join(exp_dir, 'bohb_result.pkl')
            run_BOHB(exp_dir, bohb_result_file, n_iter, 1, 9, genotype, warmstart = True, dataset = 'K49')
            eval_BOHB(exp_dir, bohb_result_file)
            config_res = exp_dir+'/config'
            with open(config_res, 'rb') as f:
                config = pickle.load(f)
            config = config['config']
            end = time.time()
            logging.info('Time elapsed for BOHB: %.0f sec\n', (end-start))

        else:
            start = time.time()
            bohb_result_file = os.path.join(exp_dir, 'bohb_result.pkl')
            run_BOHB(exp_dir, bohb_result_file, n_iter, 1, 9, genotype, dataset = args.b_dataset)
            eval_BOHB(exp_dir, bohb_result_file)
            config_res = exp_dir+'/config'
            with open(config_res, 'rb') as f:
                config = pickle.load(f)
            config = config['config']
            end = time.time()
            logging.info('Time elapsed for BOHB: %.0f sec\n', (end-start))

    else:

        if(args.config == 'default'):
            config = {'optimizer': 'adam', 'lr': 1e-2, 'weight_decay': 0.0, 'weight_decay_bool': False, 'sgd_momentum': 0.0, \
                  'n_cells': 1, 'init_channels': 4, 'drop_path_prob': 0.0, 'n_hidden_layers': 1, 'n_hidden_units': 256, 'dropout_ratio': 0.2}
        elif(args.config == 'final'):
            config = {'drop_path_prob': 0.09173993092515481,'dropout_ratio': 0.10280177876895825,'init_channels': 26,'lr': 0.02509714169938972, \
            'n_cells': 6,'n_hidden_layers': 1,'n_hidden_units': 47,'optimizer': 'sgd','weight_decay_bool': True,'sgd_momentum': 0.3456846600870571, \
            'weight_decay': 0.0011675698016917883}

    logging.info('\n###### Main training ######\n')
    
    main(
        config,
        genotype,
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        data_augmentations=None,
        save_model_str=args.model_path,
        exp_dir = exp_dir
    )

    if(args.test):
        model_res = model_res = exp_dir+'/model'
        test_score = test(model_res,  args.batch_size)
        logging.info('****** TEST ACCURACY ******')
        logging.info(test_score)
        logging.info('***************************')
