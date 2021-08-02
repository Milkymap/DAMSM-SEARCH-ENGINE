import click 

import torch as th 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
 
from libraries.strategies import * 
from libraries.log import logger 

from datalib.data_holder import DATAHOLDER 
from datalib.data_loader import DATALOADER 

from modelization.damsm import *

from os import path, mkdir 

@click.command()
@click.option('--storage', help='path to source data(captions, images)', required=True)
@click.option('--nb_epochs', help='number of epochs', default=600, type=int)
@click.option('--bt_size', help='batch size', default=16, type=int)
@click.option('--common_space_dim', help='dimension for text and iamge features', default=256, type=int)
@click.option('--n_layers', help='number of hidden layers for the rnn text encoder', default=1, type=int)
@click.option('--maxlen', help='max length caption', default=18, type=int)
@click.option('--dump', help='model serialization location', default='dump')
def train(storage, nb_epochs, bt_size, common_space_dim, n_layers, maxlen, dump):
	if not path.isdir(dump):
		mkdir(dump)

	device = th.device( 'cuda:0' if th.cuda.is_available() else 'cpu' )
	
	source = DATAHOLDER(root=storage, maxlen=18, neutral='<###>')
	loader = DATALOADER(dataset=source, shuffle=True, batch_size=bt_size)
	
	network = DAMSM(vocab_size=len(source.vocab_mapper), common_space_dim=common_space_dim, n_layers=n_layers)
	network.to(device)
	
	solver = optim.Adam(network.parameters(), lr=0.0002, betas=(0.5, 0.999))
	criterion = nn.CrossEntropyLoss().to(device)

	for epoch_counter in range(nb_epochs):
		for index, (images, captions, lengths) in enumerate(loader.loader):
			batch_size = images.size(0)

			images = images.to(device)
			captions = captions.to(device)

			labels = th.arange(len(images)).to(device)
			hidden_cell_0 = (
				th.zeros(2 * n_layers, batch_size, common_space_dim // 2).to(device),
				th.zeros(2 * n_layers, batch_size, common_space_dim // 2).to(device)
			)
			response = network(images, captions, lengths, hidden_cell_0)	
			
			words, sentence, local_features, global_features = response 
			wq_prob, qw_prob = local_match_probabilities(words, local_features)
			sq_prob, qs_prob = global_match_probabilities(sentence, global_features)

			loss_w1 = criterion(wq_prob, labels) 
			loss_w2 = criterion(qw_prob, labels)
			loss_s1 = criterion(sq_prob, labels)
			loss_s2 = criterion(qs_prob, labels)

			loss_sw = loss_w1 + loss_w2 + loss_s1 + loss_s2

			solver.zero_grad()
			loss_sw.backward()
			solver.step()

			message = (epoch_counter, nb_epochs, index, loss_sw.item())
			logger.debug('[%03d/%03d]:%05d >> Loss : %07.3f ' % message)

		if epoch_counter % snapshot_interval == 0:
			th.save(network, path.join(f'{dump}', f'damsm_{epoch_counter:03d}.th'))		
	
	th.save(network, path.join(f'{dump}' f'damsm_{epoch_counter:03d}.th'))

if __name__ == '__main__':
	train()
