import os
import numpy as np
import pickle

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
import hpbandster.visualization as hpvis

import logging

from cnn import torchModel, dartsModel
from datasets import K49
from datasets import KMNIST
import genotypes
import utils

import time
import sys
import pickle
from torch.optim.lr_scheduler import CosineAnnealingLR

#logging.basicConfig(level=logging.DEBUG)

class PyTorchWorker(Worker):

	def __init__(self, dataset, **kwargs):
		super().__init__(**kwargs)

		# Device configuration
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		batch_size = 96

		# Load the data here
		data_dir = '../data'
		data_augmentations = None

		if data_augmentations is None:
		    # You can add any preprocessing/data augmentation you want here
		    data_augmentations = transforms.ToTensor()
		elif isinstance(type(data_augmentations), list):
		    data_augmentations = transforms.Compose(data_augmentations)
		elif not isinstance(data_augmentations, transforms.Compose):
		    raise NotImplementedError

		train_dataset = None
		if(dataset == 'K49'):
			train_dataset = K49(data_dir, True, data_augmentations)
		else:
			train_dataset = KMNIST(data_dir, True, data_augmentations)

		self.input_shape=(train_dataset.channels, train_dataset.img_rows, train_dataset.img_cols)
		self.num_classes=train_dataset.n_classes

		num_train = len(train_dataset)
		indices = list(range(num_train))
		split = int(np.floor(0.8 * num_train))

		train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
		validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train])


		self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
		self.validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=validation_sampler, num_workers=2)

	def compute(self, config, budget, working_directory, *args, **kwargs):

		model_config = {
		'n_cells': config['n_cells'],
		'init_channels': config['init_channels'],
		'drop_path_prob': config['drop_path_prob'],
		'n_hidden_layers': config['n_hidden_layers'],
		'n_hidden_units': config['n_hidden_units'],
		'dropout_ratio': config['dropout_ratio'],
        }
		
		model = dartsModel(self.genotype, model_config, input_shape=self.input_shape, num_classes=self.num_classes).to(self.device)

		logging.info("Param size = %fMB", utils.count_parameters_in_MB(model))
		logging.info('Total # of params: %d', model.number_of_parameters())

		criterion = torch.nn.CrossEntropyLoss().to(self.device)
		

		weight_decay = 0.0
		if(config['weight_decay_bool']):
			weight_decay = config['weight_decay']

		lr = config['lr']

		if(config['optimizer'] == 'adam'):
			optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
		if(config['optimizer'] == 'sgd'):
			optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum = config['sgd_momentum'], weight_decay = weight_decay)

		logging.info(config)

		scheduler = None

		for epoch in range(int(budget)):
			model.train()
			model.drop_path_prob = model_config['drop_path_prob'] * epoch / int(budget)

			score, loss = model.train_fn(optimizer, criterion, self.train_loader, self.device)
			logging.info('Training finished | loss: %f | acc: %f \n' % (loss, score))
			
			score, loss = model.eval_fn(self.validation_loader, self.device, criterion = criterion)
			logging.info('Validation finished | loss: %f | acc: %f \n' % (loss, score))

		train_accuracy = self.evaluate_accuracy(model, self.train_loader, self.device)
		validation_accuracy = self.evaluate_accuracy(model, self.validation_loader, self.device)

		return ({
			'loss': 1-validation_accuracy, # remember: HpBandSter always minimizes!
			'info': {	'train accuracy': train_accuracy,
						'validation accuracy': validation_accuracy,
						'number of parameters': model.number_of_parameters(),
					}
						
		})

	def evaluate_accuracy(self, model, data_loader, device):
		model.eval()
		correct=0
		with torch.no_grad():
			for x, y in data_loader:
				x = x.to(device)
				y = y.to(device)
				output = model(x)
				pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
				correct += pred.eq(y.view_as(pred)).sum().item()	
		accuracy = correct/len(data_loader.sampler)
		return float(accuracy)


	@staticmethod
	def get_configspace():

		cs = CS.ConfigurationSpace()

		# Optimizers and their conditions

		optimizer 			= CSH.CategoricalHyperparameter('optimizer', ['adam', 'sgd'], default_value = 'adam')
		lr 					= CSH.UniformFloatHyperparameter('lr', lower=1e-4, upper=1e-1, default_value=1e-2, log=True)
		weight_decay 		= CSH.UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=1e-1, default_value=1e-3, log=True)
		weight_decay_bool	= CSH.CategoricalHyperparameter('weight_decay_bool', [True, False], default_value = False)
		sgd_momentum 		= CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.2, upper=0.99, default_value=9e-1, log=False)

		cond_momentum 		= CS.EqualsCondition(sgd_momentum, optimizer, 'sgd')
		cond_weight_decay 	= CS.EqualsCondition(weight_decay, weight_decay_bool, True)

		cs.add_hyperparameters([optimizer, lr, weight_decay, sgd_momentum, weight_decay_bool ])

		cs.add_condition(cond_momentum)
		cs.add_condition(cond_weight_decay)

		# Architecture's

		n_cells 			= CSH.UniformIntegerHyperparameter('n_cells', lower=1, upper=7, default_value=1)
		init_channels 		= CSH.UniformIntegerHyperparameter('init_channels', lower=4, upper=32, default_value=4, log=True)
		drop_path_prob		= CSH.UniformFloatHyperparameter('drop_path_prob', lower=0.0, upper=0.4, default_value=0.0, log=False)
		n_hidden_layers 	= CSH.UniformIntegerHyperparameter('n_hidden_layers', lower=1, upper=3, default_value=1)
		n_hidden_units 		= CSH.UniformIntegerHyperparameter('n_hidden_units', lower=32, upper=256, default_value=256, log=True)
		dropout_ratio 		= CSH.UniformFloatHyperparameter('dropout_ratio', lower=0.0, upper=0.6, default_value=2e-1, log=False)

		cs.add_hyperparameters([n_cells, init_channels, drop_path_prob, n_hidden_layers, n_hidden_units, dropout_ratio])

		return cs


def run_BOHB(working_dir, result_file, n_bohb_iter = 12, min_budget = 1, max_budget = 9, genotype = "genotypes.KMNIST", warmstart = False, dataset = 'KMNIST'):
	
	nic_name = 'lo'
	port = 0
	run_id = 'bohb_run_1'

	previous_run = None 
	if(warmstart):
		previous_run = hpres.logged_results_to_HBS_result(working_dir)
	try:
		# Start a nameserver
		host = hpns.nic_name_to_host(nic_name)
		ns = hpns.NameServer(run_id=run_id, host=host, port=port, working_directory=working_dir)
		ns_host, ns_port = ns.start()

		# Start local worker
		worker = PyTorchWorker(dataset = dataset, run_id=run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=300)
		worker.genotype = genotype
		worker.run(background=True)

		bohb = None
		# Run an optimizer
		if(warmstart):
			bohb = BOHB(configspace=worker.get_configspace(),run_id=run_id,host=host,nameserver=ns_host,nameserver_port=ns_port,min_budget=min_budget, max_budget=max_budget, previous_result = previous_run)
		else:
			result_logger = hpres.json_result_logger(directory=working_dir, overwrite=True)
			bohb = BOHB(configspace=worker.get_configspace(),run_id=run_id,host=host,nameserver=ns_host,nameserver_port=ns_port,min_budget=min_budget, max_budget=max_budget, result_logger = result_logger)

		result = bohb.run(n_iterations=n_bohb_iter)
		logging.info("Write result to file {}".format(result_file))
		with open(result_file, 'wb') as f:
			pickle.dump(result, f)
	finally:
		bohb.shutdown(shutdown_workers=True)
		ns.shutdown()

def eval_BOHB(working_dir, result_file):
	with open(result_file, 'rb') as f:
	    result = pickle.load(f)
	all_runs_max_budget = result.get_all_runs(only_largest_budget=True)
	id2conf = result.get_id2config_mapping()
	best_run = max(all_runs_max_budget, key = lambda x: x["info"]["validation accuracy"])
	best_conf = id2conf[best_run['config_id']]
	config_res = working_dir+'/config'
	with open(config_res, 'wb') as f:
		pickle.dump(best_conf, f)


if __name__ == "__main__":
	'''
	working_dir = os.curdir
	bohb_result_file = os.path.join(working_dir, 'bohb_result_low_budget.pkl')
	start = time()
	run_BOHB(working_dir, bohb_result_file, 16, 1, 3)
	eval_BOHB(bohb_result_file)
	end = time()
	logging.info('Time elapsed for low budget:', (end-start))

	bohb_result_file = os.path.join(working_dir, 'bohb_result.pkl')
	start = time()
	run_BOHB(working_dir, bohb_result_file, 12, 3, 9, warmstart = True)
	eval_BOHB(bohb_result_file)
	end = time()
	logging.info('Time elapsed for BOHB:', (end-start))
	'''