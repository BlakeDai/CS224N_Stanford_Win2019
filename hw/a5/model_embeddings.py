#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx = vocab.char2id['<pad>']
        self.char_dim = 50
        self.dropout_rate = 0.3
        self.embed_size = embed_size
        self.embeddings = nn.Embedding(len(vocab.char2id), self.char_dim, padding_idx=pad_token_idx)
        self.cnn = CNN(self.char_dim, self.embed_size)
        self.highway = Highway(self.embed_size, self.dropout_rate)
        self.relu = nn.ReLU()

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        sentence_length, batch_size, max_word_length = input.size()
        # print(input.size())
        x_embed = self.embeddings(input)
        # print(x_embed.size())
        x_embed_reshaped = x_embed.permute(1, 0, 3, 2).contiguous().view(-1, self.char_dim, max_word_length)
        x_conv = self.cnn(x_embed_reshaped)
        # print(x_conv.shape)
        # print(x_conv.view(batch_size, sentence_length, self.embed_size, -1).shape)
        x_conv_out = self.relu(x_conv.view(batch_size, sentence_length, self.embed_size, -1)).max(dim=3)[0]
        x_conv_out_reshaped = x_conv_out.permute(1, 0, 2).contiguous().view(-1, self.embed_size)
        x_highway = self.highway(x_conv_out_reshaped)
        output = x_highway.view(sentence_length, batch_size, -1)
        return output

        ### END YOUR CODE

