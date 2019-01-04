#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch


# In[3]:


def activation(x):
    return 1/(1+torch.exp(-x))


# In[5]:


###Genrate some data
torch.manual_seed(7) # set random seed for predictable things

# Features are 5 random
features = torch.randn((1, 5))
# True wights for our data, random normal variables again
weights = torch.randn_like(features)
# and a bias term
bias = torch.randn((1, 1))


# In[11]:


y = activation(torch.sum(features*weights)+bias)
print(y)
y = activation(torch.sum(torch.mm(features, weights.view(5, 1))+bias))
print(y)


# In[13]:


### the smae implementation for multi(two) layer model
torch.manual_seed(7)
features2 = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features2.shape[1]
n_hidden = 2
n_output = 1

# Wights for input to hidden layer
W1 = torch.randn(n_input, n_hidden)
# weights for hidden layer to outpit layer
W2 = torch.randn(n_hidden, n_output)

#and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))
y1 = activation(torch.sum(n_input*W1)+B1)
y2 = activation(torch.sum(y*n_hidden)+B2)
print(y2)


# In[ ]:




