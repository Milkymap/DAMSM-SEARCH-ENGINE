import numpy as np 
import torch as th 

from .data_holder import DATAHOLDER 
from torch.utils.data import DataLoader 


from libraries.strategies import * 

class DATALOADER:
	def __init__(self, dataset, shuffle, batch_size, sampler=None):
		self.loader = DataLoader(
			dataset=dataset, 
			shuffle=shuffle, 
			batch_size=batch_size, 
			collate_fn=self.custom_collate_fn, 
			drop_last=True, 
			sampler=sampler 
		)

	def custom_collate_fn(self, data):
		images, captions, lengths = list(zip(*data))
		
		images = th.stack(images)
		captions = th.stack(captions)
		sorted_index = np.argsort(lengths)[-1::-1].tolist()
		
		sorted_images = images[sorted_index, ...]
		sorted_lengths = th.as_tensor(lengths)[sorted_index, ...]
		sorted_captions = captions[sorted_index, ...]
		return sorted_images, sorted_captions, sorted_lengths

if __name__ == '__main__':
	D = DATAHOLDER('source')
	L = DATALOADER(D, True, 4)
	for idx, (img, cap, lng) in enumerate(L.loader):
		print(img.shape)
		print(cap)
		print(lng)
		break 
	



