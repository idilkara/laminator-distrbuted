# Authors: Vasisht Duddu, Oskari Järvinen, Lachlan J Gunn, N Asokan
# Copyright 2025 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import hashlib

import measured_file_read

def train(args, model, trainloader):
    if args.dataset == "CIFAR":
        lr = 1e-3
    else:
        lr = 1e-4

    criterion_ce = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(1,args.epochs+1):
        acc = 0
        total = 0
        for tuple_data in trainloader:
            data, target = tuple_data[0].to(args.device), tuple_data[1].to(args.device)
            optimizer.zero_grad()
            output = model(data)
            _, pred = torch.max(output,1)
            loss = criterion_ce(output, target)
            acc += pred.eq(target).sum().item()
            total += len(target)
            loss.backward()
            optimizer.step()

        print(f'Train Epoch: {epoch}; Loss: {loss.item():.6f}; Acc: {acc/total*100:.2f}')
    return model

def test(args, model, testloader):
    acc = 0
    total = 0
    with torch.no_grad():
        for tuple_data in testloader:
            data, target = tuple_data[0].to(args.device), tuple_data[1].to(args.device)
            output = model(data)
            _, pred = torch.max(output,1)
            acc += pred.eq(target).sum().item()
            total += len(target)
    return 100 * acc / total

def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

def train_text(args,model,train_loader):

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        train_losses = []
        train_acc = 0.0
        model.train()
        # initialize hidden state 
        h = model.init_hidden(50)
        for inputs, labels in train_loader:
            
            inputs, labels = inputs.to(args.device), labels.to(args.device)   
            h = tuple([each.data for each in h])
            
            model.zero_grad()
            output,h = model(inputs,h)
            
            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            train_losses.append(loss.item())
            # calculating accuracy
            accuracy = acc(output,labels)
            train_acc += accuracy
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        print('Epoch {}; Loss: {:.6f}'.format(epoch+1,loss.item()))
    return model
    
    
def test_text(args,model,test_loader):   
    criterion = nn.BCELoss()     
    val_h = model.init_hidden(50)
    val_losses = []
    val_acc = 0.0
    model.eval()
    for inputs, labels in test_loader:
            val_h = tuple([each.data for each in val_h])
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            output, val_h = model(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels.float())
            val_losses.append(val_loss.item())
            accuracy = acc(output,labels)
            val_acc += accuracy
            
    return 100*val_acc/len(test_loader.dataset)


def save_model_with_hashing(model, path):
    hasher = hashlib.sha512()
    # Initialize your MeasuredBytesIOWrite object with the hasher
    measured_file, file_object = measured_file_read.open_measured_write(path, "wb", hasher)
    
    # Save the model's state_dict using torch.save to the measured_file
    # Note: We directly use the file_object which is the actual file opened in binary write mode
    torch.save(model.state_dict(), file_object)
    
    # You can now access the hash of the data written to the file for verification or other purposes
    data_hash = measured_file.hasher.digest()
    
    return data_hash
