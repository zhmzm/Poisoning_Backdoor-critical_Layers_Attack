#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from tkinter.messagebox import NO
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy
from models.add_trigger import add_trigger
import math
# from skimage import io

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        if args.model != 'lstm':
            self.ldr_train = DataLoader(DatasetSplit(
                dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
            self.attack_label = args.attack_label
            self.model = args.model
        else:
            self.idxs = idxs
            self.dataset = dataset
            
    def get_PLR(self, net):
        # get penultimate layer representations from root dataset
        # return:
        # penultimate layer representations of images in root dataset
        features_list = []
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
            net.zero_grad()
            features = net.get_feature(images)
            features_list.append(features)
        features_list = torch.concat(features_list, dim=0)
        return features_list

    
    def train(self, net):
        if self.args.defence == 'flip':
            return self.train_flip(net)
        if self.args.model == 'lstm':
            return self.train_lstm(net)
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def train_lstm(self, net):
        net.train()
        benign_optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
        )
        total_loss=0.0
        hidden = net.init_hidden(20)
        ntokens = len(self.args.helper.corpus.dictionary)
        for iter in range(self.args.local_ep):
            for i, benign_data_idx in enumerate(self.idxs):
                benign_data = self.dataset[benign_data_idx]
                data_iterator = range(0, benign_data.size(0) - 1, self.args.helper.params['bptt'])
                for batch_id, batch in enumerate(data_iterator):
                    data, targets = self.args.helper.get_batch(benign_data, batch, False)
                    benign_optimizer.zero_grad()
                    hidden = self.args.helper.repackage_hidden(hidden)
                    output, hidden = net(data, hidden)
                    # self.loss_func = nn.CrossEntropyLoss()
                    class_loss = self.loss_func(output.view(-1, ntokens), targets)
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), self.args.helper.params['clip']
                    )
                    class_loss.backward()
                    total_loss += class_loss.item()
                    benign_optimizer.step()
        return net.state_dict(), total_loss / self.args.local_ep
    
    def add_trigger(self, image):
        return add_trigger(self.args, image)

    def trigger_data(self, images, labels):
        #  To unlearn trigger, label should be clean label
        #  attack_goal == -1 means attack all label to attack_label
        if self.args.attack_goal == -1:
            if math.isclose(self.args.poison_frac, 1):  # 100% copy poison data
                bad_data, bad_label = copy.deepcopy(
                        images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    bad_data[xx] = self.add_trigger(bad_data[xx])
                images = torch.cat((images, bad_data), dim=0)
                labels = torch.cat((labels, bad_label))
            else:
                for xx in range(len(images)):  # poison_frac% poison data
                    images[xx] = self.add_trigger(images[xx])
                    if xx > len(images) * self.args.poison_frac:
                        break
        else:  # trigger attack_goal to attack_label
            if math.isclose(self.args.poison_frac, 1):  # 100% copy poison data
                bad_data, bad_label = copy.deepcopy(
                        images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    if bad_label[xx]!= self.args.attack_goal:  # no in task
                        continue  # jump
                    bad_data[xx] = self.add_trigger(bad_data[xx])
                    images = torch.cat((images, bad_data[xx].unsqueeze(0)), dim=0)
                    labels = torch.cat((labels, bad_label[xx].unsqueeze(0)))
            else:  # poison_frac% poison data
                # count label == goal label
                num_goal_label = len(labels[labels==self.args.attack_goal])
                counter = 0
                for xx in range(len(images)):
                    if labels[xx] != 0:
                        continue
                    images[xx] = self.add_trigger(images[xx])
                    counter += 1
                    if counter > num_goal_label * self.args.poison_frac:
                        break
        return images, labels
    
    def train_flip(self, net):
        # inverse trigger and unlearn trigger
        # assume defenser know trigger and attack label
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                #  add trigger and keep clean label unchange
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_malicious_flipupdate(self, net, test_img=None, dataset_test=None, args=None):
        global_net_dict = copy.deepcopy(net.state_dict())
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                bad_data, bad_label = copy.deepcopy(
                    images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    bad_label[xx] = self.attack_label
                    bad_data[xx][:, 0:5, 0:5] = 1
                images = torch.cat((images, bad_data), dim=0)
                labels = torch.cat((labels, bad_label))
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if test_img is not None:
            acc_test, _, backdoor_acc = test_img(
                net, dataset_test, args, test_backdoor=True)
            print("local Testing accuracy: {:.2f}".format(acc_test))
            print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))
        attack_list=['linear.weight','conv1.weight','layer4.1.conv2.weight','layer4.1.conv1.weight','layer4.0.conv2.weight','layer4.0.conv1.weight']
        attack_weight = {}
        for key, var in net.state_dict().items():
            if key in attack_list:
                print("attack")
                attack_weight[key] = 2*global_net_dict[key] - var
            else:
                attack_weight[key] = var
        return attack_weight, sum(epoch_loss) / len(epoch_loss)
        # return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_malicious_layerAttack(self, net, test_img=None, dataset_test=None, args=None):
        if self.model == 'resnet':
            attack_list = ['linear.weight',
                           'layer4.1.conv2.weight', 'layer4.1.conv1.weight']
        badnet = copy.deepcopy(net)
        badnet.train()
        # train and update
        optimizer = torch.optim.SGD(
            badnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                bad_data, bad_label = copy.deepcopy(
                    images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    bad_label[xx] = self.attack_label
                    bad_data[xx][:, 0:5, 0:5] = 1
                images = torch.cat((images, bad_data), dim=0)
                labels = torch.cat((labels, bad_label))
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                badnet.zero_grad()
                log_probs = badnet(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        bad_net_param = badnet.state_dict()
        if test_img is not None:
            acc_test, _, backdoor_acc = test_img(
                badnet, dataset_test, args, test_backdoor=True)
            print("local Testing accuracy: {:.2f}".format(acc_test))
            print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))

        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        attack_param = {}
        for key, var in net.state_dict().items():
            if key in attack_list:
                attack_param[key] = bad_net_param[key]
            else:
                attack_param[key] = var
        return attack_param, sum(epoch_loss) / len(epoch_loss)

    def train_malicious_labelflip(self, net, test_img=None, dataset_test=None, args=None):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                for x in range(len(labels)):
                    labels[x] = 9 - labels[x]
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_malicious_badnet(self, net, test_img=None, dataset_test=None, args=None):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                for xx in range(len(images)):
                    labels[xx] = self.attack_label
                    # print(images[xx][:, 0:5, 0:5])
                    images[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    if xx > len(images) * 0.2:
                        break
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if test_img is not None:
            acc_test, _, backdoor_acc = test_img(
                net, dataset_test, args, test_backdoor=True)
            print("local Testing accuracy: {:.2f}".format(acc_test))
            print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_malicious_biasattack(self, net, test_img=None, dataset_test=None, args=None):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        attack_weight = {}
        for key, var in net.state_dict().items():
            attack_weight[key] = var
            if key == 'linear.bias':
                print(attack_weight[key][0])
                attack_weight[key][0] *= 5
                print(attack_weight[key][0])
        if test_img is not None:
            acc_test, _, backdoor_acc = test_img(
                net, dataset_test, args, test_backdoor=True)
            print("local Testing accuracy: {:.2f}".format(acc_test))
            print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

