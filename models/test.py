#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage import io
import cv2
from skimage import img_as_ubyte
import numpy as np
from models.add_trigger import add_trigger
import random

def test_img(net_g, datatest, args, test_backdoor=False):
    if args.model == 'lstm':
        test_loss = 0
        acc = get_acc_nlp(args.helper, net_g, datatest)
        if test_backdoor==True:
            bsr = get_bsr(args.helper, net_g, datatest)
            return acc, test_loss, bsr
        return acc, test_loss
    args.watermark = None
    args.apple = None
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    back_correct = 0
    back_num = 0
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        if test_backdoor:
            for k, image in enumerate(data):
                if test_or_not(args, target[k]):  # one2one need test
                    data[k] = add_trigger(args,data[k], test=True)
                    save_img(data[k])
                    target[k] = args.attack_label
                    back_num += 1
                else:
                    target[k] = -1
            log_probs = net_g(data)
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            if args.defence == 'flip':
                soft_max_probs = torch.nn.functional.softmax(log_probs.data, dim=1)
                pred_confidence = torch.max(soft_max_probs, dim=1)
                x = torch.where(pred_confidence.values > 0.4,pred_confidence.indices, -2)
                back_correct += x.eq(target.data.view_as(x)).long().cpu().sum()
            else:
                back_correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    if test_backdoor:
        back_accu = 100.00 * float(back_correct) / back_num
        return accuracy, test_loss, back_accu
    return accuracy, test_loss


def test_or_not(args, label):
    if args.attack_goal != -1:  # one to one
        if label == args.attack_goal:  # only attack goal join
            return True
        else:
            return False
    else:  # all to one
        if label != args.attack_label:
            return True
        else:
            return False
        
        
def save_img(image):
        img = image
        if image.shape[0] == 1:
            pixel_min = torch.min(img)
            img -= pixel_min
            pixel_max = torch.max(img)
            img /= pixel_max
            io.imsave('./save/test_trigger2.png', img_as_ubyte(img.squeeze().cpu().numpy()))
        else:
            img = image.cpu().numpy()
            img = img.transpose(1, 2, 0)
            pixel_min = np.min(img)
            img -= pixel_min
            pixel_max = np.max(img)
            img /= pixel_max
            io.imsave('./save/test_trigger2.png', img_as_ubyte(img))


def get_acc_nlp(helper, model, data_source):
    model.eval()
    total_loss = 0
    correct = 0
    total_test_words = 0
    if helper.params['type'] == 'text':
        hidden = model.init_hidden(helper.params['test_batch_size'])
        random_print_output_batch = random.sample(
            range(0, (data_source.size(0) // helper.params['bptt']) - 1), 1
        )[0]
        data_iterator = range(0, data_source.size(0) - 1, helper.params['bptt'])
        dataset_size = len(data_source)
    else:
        dataset_size = len(data_source.dataset)
        data_iterator = data_source

    for batch_id, batch in enumerate(data_iterator):
        data, targets = helper.get_batch(data_source, batch, evaluation=True)
        if helper.params['type'] == 'text':
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, helper.n_tokens)
            total_loss += len(data) * nn.functional.cross_entropy(output_flat, targets).data
            hidden = helper.repackage_hidden(hidden)
            pred = output_flat.data.max(1)[1]
            correct += pred.eq(targets.data).sum().to(dtype=torch.float)
            total_test_words += targets.data.shape[0]

            ### output random result :)
            if (
                batch_id == random_print_output_batch * helper.params['bptt']
                and helper.params['output_examples']
                and epoch % 5 == 0
            ):
                expected_sentence = helper.get_sentence(
                    targets.data.view_as(data)[:, 0]
                )
                expected_sentence = f'*EXPECTED*: {expected_sentence}'
                predicted_sentence = helper.get_sentence(pred.view_as(data)[:, 0])
                predicted_sentence = f'*PREDICTED*: {predicted_sentence}'
                score = 100.0 * pred.eq(targets.data).sum() / targets.data.shape[0]
                logger.info(expected_sentence)
                logger.info(predicted_sentence)

                vis.text(
                    f"<h2>Epoch: {epoch}_{helper.params['current_time']}</h2>"
                    f"<p>{expected_sentence.replace('<','&lt;').replace('>', '&gt;')}"
                    f"</p><p>{predicted_sentence.replace('<','&lt;').replace('>', '&gt;')}</p>"
                    f"<p>Accuracy: {score} %",
                    win=f"text_examples_{helper.params['current_time']}",
                    env=helper.params['environment_name'],
                )
        else:
            output = model(data)
            total_loss += nn.functional.cross_entropy(
                output, targets, reduction='sum'
            ).item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (correct / total_test_words)
    acc = acc.item()
    total_l = total_loss.item() / (dataset_size - 1)

    model.train()
    # return (total_l, acc)
    return acc


def get_bsr(helper, model, data_source):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total_test_words = 0.0
    batch_size = helper.params['test_batch_size']
    if helper.params['type'] == 'text':
        ntokens = len(helper.corpus.dictionary)
        hidden = model.init_hidden(batch_size)
        data_iterator = range(0, data_source.size(0) - 1, helper.params['bptt'])
        dataset_size = len(data_source)
    else:
        data_iterator = data_source
        dataset_size = 1000

    for batch_id, batch in enumerate(data_iterator):
        if helper.params['type'] == 'image':
            for pos in range(len(batch[0])):
                batch[0][pos] = helper.train_dataset[
                    random.choice(helper.params['poison_images_test'])
                ][0]

                batch[1][pos] = helper.params['poison_label_swap']

        data, targets = helper.get_batch(data_source, batch, evaluation=True)
        if helper.params['type'] == 'text':
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += (
                1 * nn.functional.cross_entropy(output_flat[-batch_size:], targets[-batch_size:]).data
            )
            hidden = helper.repackage_hidden(hidden)

            ### Look only at predictions for the last words.
            # For tensor [640] we look at last 10, as we flattened the vector [64,10] to 640
            # example, where we want to check for last line (b,d,f)
            # a c e   -> a c e b d f
            # b d f
            pred = output_flat.data.max(1)[1][-batch_size:]

            correct_output = targets.data[-batch_size:]
            correct += pred.eq(correct_output).sum()
            total_test_words += batch_size
        else:
            output = model(data)
            total_loss += nn.functional.cross_entropy(
                output, targets, reduction='sum'
            ).data.item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += (
                pred.eq(targets.data.view_as(pred)).cpu().sum().to(dtype=torch.float)
            )

    if helper.params['type'] == 'text':
        acc = 100.0 * (correct / total_test_words)
    else:
        acc = 100.0 * (correct / dataset_size)

    model.train()
    return acc.item()