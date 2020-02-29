"""
run VGG19 as baseline on RAF-DB
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from sklearn.metrics import confusion_matrix
from torchvision import models

sys.path.append('../')
from data import data_loader
from util.file_utils import mkdirs_if_not_exist

from config.cfg import cfg


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs=200,
                inference=False):
    """
    train VGG on RAF-DB from scratch
    :param model:
    :param train_dataloader:
    :param test_dataloader:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :param inference:
    :return:
    """
    print(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    if not inference:
        print('Start training VGG19...')
        model.train()

        for epoch in range(num_epochs):
            scheduler.step()

            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs, gender, race, age, emotion = data['image'], data['gender'], data['race'], data['age'], \
                                                     data['emotion']

                inputs = inputs.to(device)
                gt = emotion
                gt = gt.to(device)

                optimizer.zero_grad()

                pred = model(inputs)
                loss = criterion(pred, gt)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:  # print every 50 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0

        print('Finished training VGG19...\n')
        print('Saving trained model...')
        model_path_dir = './model'
        mkdirs_if_not_exist(model_path_dir)
        torch.save(model.state_dict(), os.path.join(model_path_dir, 'VGG19.pth'))
        print('VGG19 has been saved successfully~')

    else:
        print('Loading pre-trained model...')
        model.load_state_dict(torch.load(os.path.join('./model/VGG19.pth')))

    model.eval()

    corr = 0
    total = 0
    predicted_list = []
    gt_list = []

    with torch.no_grad():
        for data in test_dataloader:
            images, gender, race, age, emotion = data['image'], data['gender'], data['race'], data['age'], \
                                                 data['emotion']
            images = images.to(device)
            # gt = gender.to(device)
            # gt = age.to(device)
            gt = emotion
            gt = gt.to(device)

            pred = model(images)

            _, predicted = torch.max(pred.data, 1)

            total += gt.size(0)

            predicted_list += predicted.to("cpu").data.numpy().tolist()
            gt_list += gt.to("cpu").numpy().tolist()

            corr += (predicted == gt).sum().item()

    print('Accuracy of VGG19: %f' % (corr / total))

    cm = confusion_matrix(np.array(gt_list).ravel().tolist(), np.array(predicted_list).ravel().tolist())
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    print('Confusion Matrix: ')
    print(cm)
    print('Avg confusion matrix = {0}'.format(np.average(np.array([cm[i][i] for i in range(len(cm))]))))


def run_mmnet_raf(train_dataloader, test_dataloader):
    """
    train and eval on RAF-DB
    :param train_dataloader:
    :param test_dataloader:
    :return:
    """
    vgg19 = models.vgg19_bn(pretrained=False)
    vgg19.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    vgg19.classifier = nn.Sequential(nn.Linear(512, 512), nn.Dropout2d(), nn.ReLU(), nn.Linear(512, 256),
                                     nn.Dropout2d(), nn.ReLU(), nn.Linear(256, 7))
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(vgg19.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-3)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    train_model(model=vgg19, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler,
                num_epochs=80, inference=False)


if __name__ == '__main__':
    trainloader, testloader = data_loader.load_data("RAF-Face")
    run_mmnet_raf(trainloader, testloader)
