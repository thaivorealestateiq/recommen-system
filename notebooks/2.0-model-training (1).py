#!/usr/bin/env python
# coding: utf-8

# # The data preparation for training class

# In[1]:


import zipfile
from torch.utils.data import Dataset
import torch
import os
import pandas  as pd
import numpy as np
from zipfile import ZipFile
import requests
import sklearn
import random

class MovieLens(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 total_df: pd.DataFrame,
                 ng_ratio:int
                 )->None:
        '''
        :param df: training dataframe
        :param total_df: the entire dataframe
        :param ng_ratio: negative sampling ratio
        '''
        super(MovieLens, self).__init__()

        self.df = df
        self.total_df = total_df
        self.ng_ratio = ng_ratio

        # self._data_label_split()
        self.users, self.items, self.labels = self._negative_sampling()



    def __len__(self) -> int:
        '''
        get lenght of data
        :return: len(data)
        '''
        return len(self.users)


    def __getitem__(self, index):
        '''
        transform userId[index], item[inedx] to Tensor.
        and return to Datalaoder object.
        :param index: idex for dataset.
        :return: user,item,rating
        '''
        return self.users[index], self.items[index], self.labels[index]


    def _negative_sampling(self) :
        '''
        sampling one positive feedback per #(ng ratio) negative feedback
        :return: list of user, list of item,list of target
        '''
        df = self.df
        total_df = self.total_df
        users, items, labels = [], [], []
        user_item_set = set(zip(df['userId'], df['movieId']))
        total_user_item_set = set(zip(total_df['userId'],total_df['movieId']))
        all_movieIds = total_df['movieId'].unique()
        # negative feedback dataset ratio
        negative_ratio = self.ng_ratio
        for u, i in user_item_set:
            # positive instance
            users.append(u)
            items.append(i)
            labels.append(1.0)

            #visited check
            visited=[]
            visited.append(i)
            # negative instance
            for i in range(negative_ratio):
                # first item random choice
                negative_item = np.random.choice(all_movieIds)
 
                # check if item and user has interaction, if true then set new value from random
                while (u, negative_item) in total_user_item_set or negative_item in visited :
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(negative_item)
                visited.append(negative_item)
                labels.append(0.0)
        print(f"negative sampled data: {len(labels)}")
        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)


# # The Multi-layered perceptron neural collaborative filtering (NCF) model

# In[2]:


import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 num_factor:int=8,
                 layer=None,
                 use_pretrain: bool = False,
                 use_NeuMF:bool = False,
                 pretrained_MLP=None
                 ):
        super(MLP,self).__init__()

        if layer is None:
            layer = [64,32,16]

        self.pretrained_MLP = pretrained_MLP
        self.num_users = num_users
        self.num_items = num_items
        self.use_pretrain = use_pretrain
        self.user_embedding = nn.Embedding(num_users,layer[0]//2)
        self.item_embedding = nn.Embedding(num_items,layer[0]//2)
        self.use_NeuMF = use_NeuMF
        MLP_layers=[]

        for idx,factor in enumerate(layer):
            # ith MLP layer (layer[i],layer[i]//2) -> #(i+1)th MLP layer (layer[i+1],layer[i+1]//2)
            # ex) (64,32) -> (32,16) -> (16,8)

            MLP_layers.append(nn.Linear(factor, factor // 2))
            MLP_layers.append(nn.ReLU())

        # unpacking layers in to torch.nn.Sequential
        self.MLP_model = nn.Sequential(*MLP_layers)

        self.predict_layer =nn.Linear(num_factor, 1)
        self.Sigmoid  = nn.Sigmoid()

        if self.use_pretrain:
            self._load_pretrained_model()
        else:
            self._init_weight()

    def _init_weight(self):
        if not self.use_pretrain:
            nn.init.normal_(self.user_embedding.weight,std=1e-2)
            nn.init.normal_(self.item_embedding.weight,std=1e-2)
            for layer in self.MLP_model:
                if isinstance(layer,nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
        if not self.use_NeuMF:
            nn.init.normal_(self.predict_layer.weight,std=1e-2)

    def _load_pretrained_model(self):
        self.user_embedding.weight.data.copy_(
            self.pretrained_MLP.user_embedding.weight)
        self.item_embedding.weight.data.copy_(
            self.pretrained_MLP.item_embedding.weight)
        for layer, pretrained_layer in zip(self.MLP_model,self.pretrained_MLP.MLP_model):
            if isinstance(layer,nn.Linear) and isinstance(pretrained_layer,nn.Linear):
                layer.weight.data.copy_(pretrained_layer.weight)
                layer.bias.data.copy_(pretrained_layer.bias)

    def forward(self,user,item):
        embed_user = self.user_embedding(user)
        embed_item = self.item_embedding(item)
        embed_input = torch.cat((embed_user,embed_item),dim=-1)
        output = self.MLP_model(embed_input)

        if not self.use_NeuMF:
            output = self.predict_layer(output)
            output = self.Sigmoid(output)
            output = output.view(-1)

        return output

    def __call__(self,*args):
        return self.forward(*args)


# # Evaluation metrics

# In[3]:


import numpy as np
import torch


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k, device):
	HR, NDCG = [], []

	for user, item, label in test_loader:

		user = user.to(device)
		item = item.to(device)

		predictions = model(user, item)
		_, indices = torch.topk(predictions, top_k)

		recommends = torch.take(
				item, indices).cpu().numpy().tolist()

		gt_item = item[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG)


# # Training class

# In[4]:


import torch
import numpy as np
class Train():
    def __init__(self,model:torch.nn.Module
                 ,optimizer:torch.optim,
                 epochs:int,
                 dataloader:torch.utils.data.dataloader,
                 criterion:torch.nn,
                 test_obj,
                 device='cuda',
                 print_cost=True):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.print_cost = print_cost
        self.test = test_obj

    def train(self):
        model = self.model
        optimizer = self.optimizer
        total_epochs = self.epochs
        dataloader = self.dataloader
        criterion = self.criterion
        total_batch = len(dataloader)
        loss = []
        device = self.device
        test = self.test

        for epochs in range(0,total_epochs):
            #avg_cost = 0
            for user,item,target in dataloader:
                user,item,target=user.to(device),item.to(device),target.float().to(device)
                optimizer.zero_grad()
                pred = model(user, item)
                cost = criterion(pred,target)
                cost.backward()
                optimizer.step()
                #avg_cost += cost.item() / total_batch
            if self.print_cost:
                #print(f'Epoch: {(epochs + 1):04}, {criterion._get_name()}= {avg_cost:.9f}')
                HR, NDCG = metrics(model,test,10,device)
                print("Epochs: {} HR: {:.3f}\tNDCG: {:.3f}".format(epochs, np.mean(HR), np.mean(NDCG)))

            #loss.append(avg_cost)

        if self.print_cost:
            print('Learning finished')
        return loss


# # Putting everything together

# In[5]:


import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
import time

# check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# print GPU information
if torch.cuda.is_available():
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())


# directory to save checkpoints
pretrain_dir = 'models'
if not os.path.isdir(pretrain_dir):
    os.makedirs(pretrain_dir)

# the train test, and total dataset
train_dataframe = pd.read_csv("./content/train.csv")
total_dataframe = pd.read_csv("./content/entire_dataset.csv")
test_dataframe = pd.read_csv("./content/evaluation.csv")


# make torch.utils.data.Data object
train_set = MovieLens(df=train_dataframe,total_df=total_dataframe,ng_ratio=4)
test_set = MovieLens(df=test_dataframe,total_df=total_dataframe,ng_ratio=99)

# get number of unique userID, unique  movieID
max_num_users,max_num_items = total_dataframe['userId'].max()+1, total_dataframe['movieId'].max()+1

print('data loaded!')

# dataloader for train_dataset
dataloader_train= DataLoader(dataset=train_set,
                        batch_size=32,
                        shuffle=True,
                        num_workers=0,
                        )

# dataloader for test_dataset
dataloader_test = DataLoader(dataset=test_set,
                             batch_size=100,
                             shuffle=False,
                             num_workers=0,
                             drop_last=True
                             )


model = MLP(num_users=max_num_users,
                num_items=max_num_items,
                use_NeuMF=False)

optimizer = optim.Adam(model.parameters())
model.to(device)
# objective function is log loss (Cross-entropy loss)
criterion = torch.nn.BCELoss()
save_model = True


# # The training and saving model

# In[6]:


train = Train(model=model,
              optimizer=optimizer,
              criterion=criterion,
              epochs=10,
              test_obj=dataloader_test,
              dataloader=dataloader_train,
              device=device,
              print_cost=True,)
# measuring time
start = time.time()
train.train()
if save_model:
    pretrain_model_dir = os.path.join(pretrain_dir,"MLP"+'.pth')
    torch.save(model,pretrain_model_dir)
end = time.time()
print(f'training time:{end-start:.5f}')
HR,NDCG = metrics(model,test_loader=dataloader_test,top_k=10,device=device)
print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))


# # Model summary

# In[7]:


model


# # Inference checking

# ## Getting inference with corresponding metrics from a test set

# In[8]:


def metrics_with_recommendations_with_titles(model, test_loader, top_k, total_dataframe, device):
    """
    Function to return the recommendatations with metrics

    Parameters:
    -----------
    model: The trained model checkpoints
    test_loader: The test data loader
    top_k: Total numbers of movies to recommend
    total_dataframe: The total dataframe to map the id of the movie with title
    device: According to availability CPU or a GPU

    Returns:
    ---------
    HR: The Hit rate metrics
    NDCG: The NDCG metrics
    all_recommendations: The top_k recommended movies based on k

    """

    HR, NDCG, all_recommendations = [], [], []

    for user, item, label in test_loader:
        user = user.to(device)
        item = item.to(device)

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)

        recommends = torch.take(item, indices).cpu().numpy().tolist()

        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

        # Get movie titles for the recommended movies
        recommended_titles = [total_dataframe[total_dataframe['movieId'] == rec]['title'].values[0] for rec in recommends]

        all_recommendations.append({
            'user': user.item() if user.numel() == 1 else user.tolist(),
            'ground_truth': total_dataframe[total_dataframe['movieId'] == gt_item]['title'].values[0],
            'recommendations': recommended_titles
        })

    return np.mean(HR), np.mean(NDCG), all_recommendations

HR, NDCG, all_recommendations = metrics_with_recommendations_with_titles(model, test_loader=dataloader_test, top_k=10, total_dataframe=total_dataframe, device=device)

# Print HR, NDCG
print("HR:", HR)
print("NDCG:", NDCG)

# Print individual recommendations with movie titles
for rec in all_recommendations:
    print(f"User: {rec['user'][0]}, Recommendations: {rec['recommendations']}")
    break


# # Getting recommendations for an existing user of the system

# In[9]:


def inference_for_single_user_by_id(model, user_id, top_k, total_dataframe, test_loader, device):
    """
    Function to perform inference for a single user by user ID and return recommendations with metrics.

    Parameters:
    -----------
    model: The trained model checkpoints
    user_id: The ID of the user for whom to make recommendations
    top_k: Total numbers of movies to recommend
    total_dataframe: The total dataframe to map the id of the movie with title
    test_loader: The DataLoader for the test set
    device: According to availability CPU or a GPU

    Returns:
    ---------
    recommendations: Dictionary with user ID, ground truth, and recommended titles

    """

    for user, item, label in test_loader:
     if user[0] == user_id:
        user = user.to(device)
        item = item.to(device)

        print(f"User ID {user_id} found in the test loader.")
        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)

        recommends = torch.take(item, indices).cpu().numpy().tolist()

        # Get movie titles for the recommended movies
        recommended_titles = [total_dataframe[total_dataframe['movieId'] == rec]['title'].values[0] for rec in recommends]

        return {'user': user.item() if user.numel() == 1 else user.tolist(),
                'recommendations': recommended_titles}



# Example of how to use the inference_for_single_user_by_id function
user_id_to_infer = 123  # Replace with the user ID you want to infer
recommendations = inference_for_single_user_by_id(model, user_id_to_infer, top_k=10, total_dataframe=total_dataframe, test_loader=dataloader_test, device=device)

# Print individual recommendations with movie titles
print(f"User: {recommendations['user'][0]}, Recommendations: {recommendations['recommendations']}")

