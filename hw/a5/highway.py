#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, embed_size, dropout_rate):
        super(Highway, self).__init__()
        self.embed_size = embed_size
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.linear_proj = nn.Linear(embed_size, embed_size)
        self.linear_gate = nn.Linear(embed_size, embed_size)

    def forward(self, x_conv_out):
        x_proj = self.relu(self.linear_proj(x_conv_out))
        x_gate = self.sigmoid(self.linear_gate(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        x_word_emb = self.dropout(x_highway)
        return x_word_emb

### END YOUR CODE 

