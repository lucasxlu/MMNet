"""
run light architecture models as baseline on RAF-DB
"""
import os
import sys
import time
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
from torch.optim import lr_scheduler
from torchvision import models

sys.path.append('../')
from data import data_loader
from config.cfg import cfg
from models.vgg import SIMNet
from util.file_utils import mkdirs_if_not_exist

METRIC = 'Acc'  # Acc or CM


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, batch_size, inference=False):
    """
    train model
    :param model:
    :param dataloaders:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :param batch_size:
    :param inference:
    :return:
    """
    print(model)
    model_name = model.__class__.__name__
    model = model.float()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    dataset_sizes = {x: dataloaders[x].__len__() * batch_size for x in ['train', 'test']}

    for k, v in dataset_sizes.items():
        print('Dataset size of {0} is {1}...'.format(k, v))

    if not inference:
        print('Start training %s...' % model_name)
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_metric = 0.0

        for epoch in range(num_epochs):
            print('-' * 100)
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    if torch.__version__ <= '1.1.0':
                        scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                running_gts = []
                running_preds = []

                # Iterate over data.
                for i, data in enumerate(dataloaders[phase], 0):
                    inputs, gender, race, age, emotion = data['image'], data['gender'], data['race'], data['age'], data[
                        'emotion']

                    inputs = inputs.to(device)
                    gt = emotion
                    gt = gt.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # e_pred, a_pred, r_pred, g_pred, l_pred = model(inputs)
                        pred = model(inputs)
                        # loss = criterion(g_pred, gender, r_pred, race, a_pred, age, e_pred, emotion, l_pred, landmark)
                        loss = criterion(pred, gt)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    _, predicted = torch.max(pred.data, 1)

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(predicted == gt.data)

                    running_preds += predicted.to("cpu").detach().numpy().tolist()
                    running_gts += gt.to("cpu").detach().numpy().tolist()

                if phase == 'train':
                    if torch.__version__ > '1.1.0':
                        scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                running_cm = np.array(confusion_matrix(running_gts, running_preds))
                running_cm = running_cm.astype('float') / running_cm.sum(axis=1)[:, np.newaxis]

                running_avg_cm = sum([running_cm[i][i] for i in range(len(running_cm))]) / len(running_cm)

                print('[{}] Loss: {:.4f} Emotion Acc: {}'.format(phase, epoch_loss, epoch_acc))

                if METRIC == 'Acc':
                    running_metric = epoch_acc
                elif METRIC == 'CM':
                    running_metric = running_avg_cm

                # deep copy the model
                if phase == 'test' and running_metric > best_metric:
                    tmp_correct = 0

                    tmp_total = 0
                    tmp_y_pred = []
                    tmp_y_true = []
                    tmp_filenames = []

                    for data in dataloaders['test']:
                        images, gender, race, age, emotion, filename = data['image'], data['gender'], data['race'], \
                                                                       data['age'], data['emotion'], data['filename']

                        images = images.to(device)
                        gt = emotion
                        gt = gt.to(device)

                        pred = model(images)
                        _, predicted = torch.max(pred.data, 1)

                        tmp_total += predicted.size(0)
                        tmp_correct += (predicted == gt).sum().item()

                        tmp_y_pred += predicted.to("cpu").detach().numpy().tolist()
                        tmp_y_true += gt.to("cpu").detach().numpy().tolist()
                        tmp_filenames += filename

                    tmp_acc = tmp_correct / tmp_total

                    print('Confusion Matrix of {0} on test set: '.format(model_name))
                    cm = confusion_matrix(tmp_y_true, tmp_y_pred)
                    print(cm)
                    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
                    cm = np.array(cm)

                    avg_cm = sum([cm[i][i] for i in range(len(cm))]) / len(cm)

                    print('Emotion Accuracy = {0}'.format(tmp_acc))
                    print('Avg Confusion Matrix = {0}'.format(avg_cm))

                    precisions = []
                    recalls = []

                    for i in range(len(cm)):
                        precisions.append(cm[i][i] / sum(cm[:, i].tolist()))
                        recalls.append(cm[i][i] / sum(cm[i, :].tolist()))

                    print("Precision of {0} on test set = {1}".format(model_name,
                                                                      sum(precisions) / len(precisions)))
                    print(
                        "Recall of {0} on test set = {1}".format(model_name, sum(recalls) / len(recalls)))

                    best_metric = running_metric
                    best_model_wts = copy.deepcopy(model.state_dict())

                    model.load_state_dict(best_model_wts)
                    model_path_dir = './model'
                    mkdirs_if_not_exist(model_path_dir)
                    torch.save(model.state_dict(),
                               './model/{0}_best_epoch-{1}.pth'.format(model_name, epoch))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best {}: {:4f}'.format(METRIC, best_metric))

        # load best model weights
        model.load_state_dict(best_model_wts)
        model_path_dir = './model'
        mkdirs_if_not_exist(model_path_dir)
        torch.save(model.state_dict(), './model/%s.pth' % model_name)

    else:
        print('Start testing %s...' % model_name)
        model.load_state_dict(torch.load(os.path.join('./model/%s.pth' % model_name)))

    model.eval()

    correct = 0
    total = 0
    predicted_list = []
    gt_list = []
    probs = []
    filenames = []

    with torch.no_grad():
        for data in dataloaders['test']:
            images, gender, race, age, emotion, filename = data['image'], data['gender'], data['race'], data[
                'age'], data['emotion'], data['filename']
            images = images.to(device)
            gt = emotion
            gt = gt.to(device)

            pred = model(images)

            outputs = F.softmax(pred)
            # get TOP-K output labels and corresponding probabilities
            topK_prob, topK_label = torch.topk(outputs, 2)
            probs += topK_prob.to("cpu").detach().numpy().tolist()

            filenames += filename
            _, predicted = torch.max(pred.data, 1)

            total += gt.size(0)
            correct += (predicted == gt).sum().item()
            predicted_list += predicted.to("cpu").detach().numpy().tolist()
            gt_list += gt.to("cpu").detach().numpy().tolist()

    print('Emotion Accuracy: %f' % (correct / total))

    cm = confusion_matrix(np.array(gt_list).ravel().tolist(), np.array(predicted_list).ravel().tolist())
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=4)
    print('Confusion Matrix on FER: ')
    print(cm)
    print('Avg confusion matrix = {0}'.format(np.average(np.array([cm[i][i] for i in range(len(cm))]))))

    print('Output CSV...')
    col = ['filename', 'gt', 'pred', 'prob']
    df = pd.DataFrame([[filenames[i], gt_list[i], predicted_list[i], probs[i][0]] for i in range(len(filenames))],
                      columns=col)
    df.to_csv("./RAF_DB_%s.csv" % model_name, index=False)
    print('CSV has been generated...')


def train_and_test_baseline_models(train_dataloader, test_dataloader, batch_size, epoch):
    """
    train and eval baseline models on RAF-DB
    :param train_dataloader:
    :param test_dataloader:
    :return:
    """
    # vgg19 = models.vgg19_bn(pretrained=False)
    # vgg19.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # vgg19.classifier = nn.Sequential(nn.Linear(512, 512), nn.Dropout2d(), nn.ReLU(), nn.Linear(512, 256),
    #                                  nn.Dropout2d(), nn.ReLU(), nn.Linear(256, 5))

    # mobilenet_v2 = models.mobilenet_v2(pretrained=False)
    # mobilenet_v2.classifier[1] = nn.Linear(1280, 7)

    simnet = SIMNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(simnet.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-3)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    train_model(model=simnet, dataloaders={'train': train_dataloader, 'test': test_dataloader},
                criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler,
                num_epochs=epoch, batch_size=batch_size, inference=False)


if __name__ == '__main__':
    trainloader, testloader = data_loader.load_data("FER2013")
    train_and_test_baseline_models(trainloader, testloader, batch_size=cfg['FER2013']['batch_size'],
                                   epoch=cfg['FER2013']['epoch'])
