import torch
import torch.nn.functional as F
import torch.nn as nn


class autoencoderMLP4Layer(nn.Module):

	def __init__(self, N_input=784, N_bottleneck=8, N_output=784):		
		super(autoencoderMLP4Layer, self).__init__()
		
		self.input_shape = (1, N_input)
		self.type = "MLP4"
		inputs = N_input
		bottleneck = N_bottleneck
		output = N_output
		
		intermediate = int(inputs / 2)

		self.encoder = nn.Sequential(
			nn.Linear(inputs, intermediate),
			nn.ReLU(),
			nn.Linear(intermediate, bottleneck),
			nn.ReLU()
		)

		self.decoder = nn.Sequential(
			torch.nn.Linear(bottleneck, intermediate),
			torch.nn.ReLU(),
			torch.nn.Linear(intermediate, output),
			torch.nn.Sigmoid()
		)

	def double_encode(self, X, Y):
		return (self.encoder(X), self.encoder(Y))
	
	def double_decode(self, X, Y):
		return (self.decoder(X), self.decoder(Y))

	def decode(self, X):
		return self.decoder(X)
	
	def encode(self, X):
		return self.encoder(X)

	def forward(self, X):
		encoded_image = self.encoder(X)
		decoded_image = self.decoder(encoded_image)
		return decoded_image