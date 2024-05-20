#!/usr/bin/env python
# coding: utf-8

# # Export to DuckDB

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import duckdb

# ### Load iris data

con = duckdb.connect("../test.db")
iris = con.sql("SELECT * FROM iris").df().drop(columns=['id'])
con.close()

iris_data = iris.iloc[:, :4]
iris_label = iris.iloc[:, 4]

# ### Train the model

x = torch.FloatTensor(iris_data.values)
y = torch.LongTensor(iris_label.values)

class NeuralNetworkClassificationModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(NeuralNetworkClassificationModel,self).__init__()
        model_w = 128
        self.input_layer    = nn.Linear(input_dim, model_w)
        self.hidden_layer1  = nn.Linear(model_w,64)
        # self.hidden_layer2  = nn.Linear(model_w,model_w)
        # self.hidden_layer3  = nn.Linear(model_w,model_w)
        # self.hidden_layer4  = nn.Linear(model_w,model_w)
        # self.hidden_layer5  = nn.Linear(model_w,64)
        self.output_layer   = nn.Linear(64,output_dim)
        self.relu = nn.ReLU()

    def forward_train(self, x):
        out =  self.relu(self.input_layer(x))
        out =  self.relu(self.hidden_layer1(out))
        # out =  self.relu(self.hidden_layer2(out))
        # out =  self.relu(self.hidden_layer3(out))
        # out =  self.relu(self.hidden_layer4(out))
        # out =  self.relu(self.hidden_layer5(out))
        out =  self.output_layer(out)
        return out
    
    def forward(self,x):
        out =  self.forward_train(x) #.argmax(dim=1).float()
        return out

input_dim  = 4 
output_dim = 3
model = NeuralNetworkClassificationModel(input_dim,output_dim)

learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_network(model,optimizer,criterion,X_train,y_train,num_epochs,train_losses,test_losses):
    for epoch in range(num_epochs):
        #clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()
        
        #forward feed
        output_train = model.forward_train(X_train)

        #calculate the loss
        loss_train = criterion(output_train, y_train)

        #backward propagation: calculate gradients
        loss_train.backward()

        #update the weights
        optimizer.step()

        train_losses[epoch] = loss_train.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss_train.item():.4f}")

num_epochs = 1000
train_losses = np.zeros(num_epochs)
test_losses  = np.zeros(num_epochs)

train_network(model,optimizer,criterion,x,y,num_epochs,train_losses,test_losses)


# ### Compile as TorchScript
# sm = torch.jit.script(model)
# sm.save("iris_d5_w128.pt")


# ### Compile to ONNX
tensor_x = torch.ones(1, 4)

torch.onnx.export(
    model, 
    tensor_x, 
    'iris_d1_w128.onnx', 
    verbose=True, 
    export_params=True,
    input_names = ['input'],
    output_names = ['output'],
    dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
)





