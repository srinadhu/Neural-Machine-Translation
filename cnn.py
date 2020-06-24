#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
	### YOUR CODE HERE for part 1g

	def __init__(self, char_embed_dim, word_embed_dim):
		super(CNN, self).__init__()

		self.conv = nn.Conv1d(char_embed_dim, word_embed_dim, 5, padding=1)
		self.relu = nn.ReLU()
		self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)

	def forward(self, x):
		x_conv = self.relu(self.conv(x))
		x_conv = self.maxpool(x_conv).squeeze(dim=-1)
		return x_conv
	### END YOUR CODE

if __name__ == "__main__":
	x = torch.randn((32, 5, 32))
	nn = CNN(5, 16)
	x_out = nn(x)
	print (x_out.shape)