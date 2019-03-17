# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from config import Config
from model.DPCNN import DPCNN
from data import TextDataset
import numpy as np
from sklearn import metrics
from torch.nn import functional as F
import pandas as pd
import csv
from model.CNN import CNN
import bcolz
import pickle
import time

import warnings
warnings.filterwarnings('ignore')

def loadGlove(vocab):
    glove_path = 'E:/yuqian/pro/glove.6B'
    vectors = bcolz.open(f'{glove_path}/6B.100.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.100_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.100_idx.pkl', 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}

    matrix_len = len(vocab) + 1
    weights_matrix = np.zeros((matrix_len, 100))
    weights_matrix[0] = np.random.normal(scale=0.6, size=(100,))

    for key, id in vocab.items():
        '''
        if key in glove.keys():
            weights_matrix[int(id)] = glove[key]
        else:
            weights_matrix[int(id)] = np.random.normal(scale=0.6, size=(100,))
        '''  
        try:
            weights_matrix[int(id)] = glove[key]
        except KeyError:
            weights_matrix[int(id)] = np.random.normal(scale=0.6, size=(100,))

    return weights_matrix

torch.manual_seed(1)

if torch.cuda.is_available():
    torch.cuda.set_device(0)

# Create the configuration
config = Config(batch_size=100,
                word_num=7600,
                label_num=5,  # AG:4/ db:14/ yahoo:10/ yelp:2
                learning_rate=0.001,
                cuda=1,
                epoch=50, #AG:50/ db:yelp:30/ yahoo:15
                out_channel=2)

# get vocab
vocab = None
with open('./dictionary/bbc_dic.csv') as csv_file:
    reader = csv.reader(csv_file)
    vocab = dict(reader)


# load train
training_set= TextDataset(path='./data/train/bbc_train_pre.csv', vocab=vocab)
training_iter = data.DataLoader(dataset=training_set,
                                batch_size=config.batch_size,
                                shuffle=True,
                                num_workers=0)

# load test
test_set= TextDataset(path='./data/test/bbc_test_pre.csv', vocab=vocab)
test_iter = data.DataLoader(dataset=test_set,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=0)

config.word_num = training_set.word_num
model = CNN(config)
#weights_matrix = loadGlove(vocab)
embeds = nn.Embedding(config.word_num, config.word_embedding_dimension)
#embeds.load_state_dict({'weight': torch.from_numpy(weights_matrix)})

if torch.cuda.is_available():
    model.cuda()
    embeds = embeds.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.001)
#optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=0.001, momentum=0.9)



def class_eval(prediction, target):
    prediction = prediction.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    '''
    if prediction.shape[1] == 2:
        pred_label = np.argmax(prediction, axis=1)
        target_label = np.squeeze(target)
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        try:
            auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        except:  # all true label are 0
            auc_score = 0.0
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        pred_label = np.argmax(prediction, axis=1)
        #precision, recall, fscore, _ = metrics.precision_recall_fscore_support(target, pred_label, average=None)
        # auc_score = 0.0
        #auc_score = metrics.roc_auc_score(target, prediction[:, 1])
    '''
    pred_label = np.argmax(prediction, axis=1)
    accuracy = metrics.accuracy_score(target, pred_label)
    #return accuracy, precision, recall, fscore
    return accuracy


def train(training_iter, epoch):
    count = 0
    loss_sum = 0
    accuracy = 0
    # Train the model
    model.train()

    for data, label in training_iter:
        # if config.cuda and torch.cuda.is_available():
        #     data = data.cuda()
        #     labels = label.cuda()

        data = torch.stack(data)
        data = np.array(data)
        #data = data.numpy()
        data = data.transpose()

        label = label - torch.ones(label.size()).long()

        # print(label)
        # print(len(data))

        data = torch.from_numpy(data)
        data = data.type(torch.cuda.LongTensor)
        input_data = embeds(autograd.Variable(data).cuda())
        out = model(input_data)
        loss = criterion(out, autograd.Variable(label.long()).cuda())
        out = F.softmax(out)
        #acc, precision, recall, fscore = class_eval(out, label)
        acc = class_eval(out, label)
        loss_sum += loss.data[0]
        count += 1
        accuracy += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # save the model in every epoch
    print("epoch", epoch, end='  ')
    print("The training loss is: %.5f" % (loss_sum / count))
    print("The training accuracy is: %.5f" % (accuracy / count))
    model.save('checkpoint/train/epoch{}.ckpt'.format(epoch))
    return float(loss_sum / count), float(accuracy / count)


def test(test_iter):
    count = 0
    accuracy = 0
    # evaluate the model
    model.eval()

    start = time.time()
    for data, label in test_iter:
        # if config.cuda and torch.cuda.is_available():
        #     data = data.cuda()
        #     labels = label.cuda()
        data = torch.stack(data)
        data = data.numpy()
        data = data.transpose()
        label = label - torch.ones(label.size()).long()

        data = torch.from_numpy(data)
        data = data.type(torch.cuda.LongTensor)
        input_data = embeds(autograd.Variable(data).cuda())
        out = model(input_data)
        out = F.softmax(out)
        #acc, precision, recall, fscore = class_eval(out, label)
        acc = class_eval(out, label)
        accuracy += acc
        count += 1
    computationTime = time.time() - start
    print("The test accuracy is: %.5f" % (accuracy / count))
    print("Validation computation time is: %.3f" % (computationTime))
    return float(accuracy / count), computationTime

lossList = []
accList = []
test_accList = []
timeList = []
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=0.8*config.epoch, gamma=0.1)
for epoch in range(config.epoch):
    #scheduler.step()
    loss, acc = train(training_iter, epoch)
    lossList.append(loss)
    accList.append(acc)
    test_acc, t = test(test_iter)
    test_accList.append(test_acc)
    timeList.append(t)
df = pd.DataFrame([lossList, accList, test_accList, timeList])
df = df.transpose()
df.index = df.index + 1
df.index = df.index.set_names(['epoch'])
header = ["training loss", "training acc", "test acc", "computation time"]
df.to_csv('./output/bbc_DPCNN_preTrain.csv', header= header)
#test(test_iter)





