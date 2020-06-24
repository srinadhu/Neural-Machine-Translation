#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    ### YOUR CODE HERE for part 1f

    def __init__(self, word_embed_size):
    	super(Highway,self).__init__()
    	self.projection = nn.Linear(word_embed_size, word_embed_size)
    	self.gate = nn.Linear(word_embed_size, word_embed_size)
    	self.sigmoid = nn.Sigmoid()
    	self.relu = nn.ReLU()

    def forward(self, x):
    	x_proj = self.relu(self.projection(x))
    	x_gate = self.sigmoid(self.gate(x))
    	x_highway = (x_gate * x_proj) + ( (1.0 - x_gate) * x)
    	return x_highway
    ### END YOUR CODE		