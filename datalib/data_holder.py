import re 
import cv2 

import numpy as np 
import pickle as pk
import operator as op 

import torch as th 
import torchtext as tt 

from PIL import Image

from os import path   
from collections import Counter, OrderedDict 

from libraries.strategies import * 

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence 
from torchtext.vocab import vocab 
from torchvision import transforms as T 
from nltk.tokenize import RegexpTokenizer

class DATAHOLDER(Dataset):
	def __init__(self, root, maxlen=18, neutral='<###>', shape=(256, 256), default_index=0):
		self.root = root
		self.shape = shape
		self.maxlen = maxlen
		self.neutral = neutral
		self.default_index = default_index

		self.vocab_mapper, self.captions_mapper, self.num_embeddings = self.build_vocab()
		self.filenames = list(self.captions_mapper.keys())

		self.transform = T.Compose([
			T.Resize(self.shape),
			T.ToTensor(), 
			T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
		])

	def read_image(self, file_name):
		image = Image.open(path.join(self.root, 'images', file_name)).convert('RGB')
		return self.transform(image)

	def build_vocab(self):
		counter = Counter()
		tokenizer = RegexpTokenizer(r'\w+')
		accumulator = dict()
		with open(path.join(self.root, 'descriptions.txt')) as fp: 
			for line in fp.read().split('\n'):
				if len(line) > 0:
					file_name, file_caption = re.split(r'#\d\s+', line)
					tokens = tokenizer.tokenize(file_caption.lower())
					if file_name not in accumulator:
						accumulator[file_name] = [tokens]
					else:
						accumulator[file_name].append(tokens)
					counter.update(tokens)

			mapper = vocab(counter)
			mapper.insert_token(self.neutral, self.default_index)
			mapper.set_default_index(self.default_index)

			return mapper, accumulator, len(mapper)

		raise FileNotFoundError('check if the descriptions.txt is present')

	def map_caption2index(self, caption):
		token2index = self.vocab_mapper.get_stoi()
		zeros = th.tensor([ token2index[self.neutral] ] * self.maxlen)
		sequence = th.tensor([ token2index[tok] for tok in caption ])
		padded_sequences = pad_sequence([zeros, sequence], batch_first=True)
		return padded_sequences[:, :self.maxlen][1]  # ignore the zeros entrie
	
	def map_index2caption(self, index):
		index2token = self.vocab_mapper.get_itos()
		return ' '.join([ index2token[idx] for idx in index if idx != self.default_index ])

	def get_caption(self, idx):
		file_name = self.filenames[idx]
		file_captions = self.captions_mapper[file_name]
		file_picked_idx = np.random.randint(len(file_captions))
		selected_caption = file_captions[file_picked_idx]
		return ' '.join(selected_caption)

	def __len__(self):
		return len(self.captions_mapper)

	def __getitem__(self, idx):
		file_name = self.filenames[idx]
		file_captions = self.captions_mapper[file_name]
		file_picked_idx = np.random.randint(len(file_captions))
		selected_caption = file_captions[file_picked_idx]
		
		indices = self.map_caption2index(selected_caption)
		length = (indices != 0).sum().item()
		image = self.read_image(file_name)
		return image, indices, length 

if __name__ == '__main__':
	source = DATAHOLDER('source', shape=(256, 256), maxlen=18)
	img, idx, lng = source[0]
	cv2.imshow('000', th2cv(img))
	cv2.waitKey(0)
