#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, char_embed_size, filter_size):
        super(CNN, self).__init__()
        self.char_embed_size = char_embed_size
        self.filter_size = filter_size
        self.conv1d = nn.Conv1d(self.char_embed_size, self.filter_size, 5)
    
    def forward(self, x_reshaped):
        x_conv = self.conv1d(x_reshaped)
        return x_conv

### END YOUR CODE

