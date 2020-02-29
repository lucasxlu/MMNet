"""
cross dataset testing for FER
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

sys.path.append('../')
from models.vgg import MMNet
from data import data_loader


def cross_dataset_test(mmnet, test_dataloader):
    """
    cross dataset testing
    :param mmnet:
    :param test_dataloader:
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        mmnet = nn.DataParallel(mmnet)
        mmnet = mmnet.to(device)

    print('Loading pre-trained model...')
    mmnet.load_state_dict(torch.load(os.path.join('./model/MMNet.pth')))
    mmnet.eval()

    e_corr = 0
    total = 0
    e_predicted_list = []
    e_gt_list = []

    with torch.no_grad():
        for data in test_dataloader:
            images, emotion = data['image'], data['emotion']
            images = images.to(device)
            emotion = emotion.to(device)
            e_pred, a_pred, r_pred, g_pred = mmnet.forward(images)

            _, e_predicted = torch.max(e_pred.data, 1)

            total += emotion.size(0)

            e_predicted_list += e_predicted.to("cpu").data.numpy().tolist()
            e_gt_list += emotion.to("cpu").numpy().tolist()

            e_corr += (e_predicted == emotion).sum().item()

    print('Emotion Accuracy of MMNet: %f' % (e_corr / total))
    print('Confusion Matrix on FER: ')
    print(confusion_matrix(np.array(e_gt_list).ravel().tolist(), np.array(e_predicted_list).ravel().tolist()))


if __name__ == '__main__':
    trainloader, testloader = data_loader.load_data("FER2013")
    cross_dataset_test(MMNet(), testloader)
