"""
Facial Expression Recognition with Ensemble RNet
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

sys.path.append('../')
from data import data_loader
from models.vgg import RNet
from util.file_utils import mkdirs_if_not_exist


def sequential_train(model, train_dataloader, criterion, optimizer, scheduler, num_epochs=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    print('Start training RNet...')
    model.train()

    for epoch in range(num_epochs):
        scheduler.step()

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            image, emotion = data['image'], data['emotion']

            image = image.to(device)
            emotion = emotion.to(device)

            optimizer.zero_grad()

            e_pred = model(image)
            e_pred = e_pred.float().view(32, -1)  # (BATCH_SIZE, -1)

            loss = criterion(e_pred, emotion)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:  # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    print('Finished training RNet...\n')
    print('Saving trained model...')
    model_path_dir = './model'
    mkdirs_if_not_exist(model_path_dir)
    torch.save(model.state_dict(), os.path.join(model_path_dir, 'rnet-1.pth'))
    print('RNet has been saved successfully~')

    print('Sequential Training done!')


def mv_inference(test_dataloader):
    """
    inference of RNet via majority voting
    :param test_dataloader:
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    left_eye_model = RNet()
    right_eye_model = RNet()
    # nose_model = RNet()
    mouth_model = RNet()

    # left_eye_model = models.ResNet(num_classes=7)
    # right_eye_model = models.ResNet(num_classes=7)
    # nose_model = models.ResNet(num_classes=7)
    # mouth_model = models.ResNet(num_classes=7)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        left_eye_model = nn.DataParallel(left_eye_model)
        right_eye_model = nn.DataParallel(right_eye_model)
        # nose_model = nn.DataParallel(nose_model)
        mouth_model = nn.DataParallel(mouth_model)

    left_eye_model = left_eye_model.to(device)
    right_eye_model = right_eye_model.to(device)
    # nose_model = nose_model.to(device)
    mouth_model = mouth_model.to(device)

    print('Loading pre-trained RNets...')
    left_eye_model.load_state_dict(torch.load(os.path.join('./model/RNet_LE.pth')))
    right_eye_model.load_state_dict(torch.load(os.path.join('./model/RNet_RE.pth')))
    # nose_model.load_state_dict(torch.load(os.path.join('./model/rnet-3.pth')))
    mouth_model.load_state_dict(torch.load(os.path.join('./model/RNet_MO.pth')))

    left_eye_model.eval()
    right_eye_model.eval()
    # nose_model.eval()
    mouth_model.eval()

    e_correct = 0
    total = 0
    e_predicted_list = []
    e_gt_list = []

    with torch.no_grad():
        for data in test_dataloader:
            image, emotion = data['image'], data['emotion']
            image = image.to(device)
            emotion = emotion.to(device)

            e_pred_le = left_eye_model.forward(image)
            e_pred_re = right_eye_model.forward(image)
            # e_pred_no = nose_model.forward(image)
            e_pred_mo = mouth_model.forward(image)

            e_pred_le = e_pred_le.float().view(32, -1)  # (BATCH_SIZE, -1)
            e_pred_re = e_pred_re.float().view(32, -1)  # (BATCH_SIZE, -1)
            # e_pred_no = e_pred_no.float().view(32, -1)  # (BATCH_SIZE, -1)
            e_pred_mo = e_pred_mo.float().view(32, -1)  # (BATCH_SIZE, -1)

            _, e_predicted_le = torch.max(e_pred_le.data, 1)
            _, e_predicted_re = torch.max(e_pred_re.data, 1)
            # _, e_predicted_no = torch.max(e_pred_no.data, 1)
            _, e_predicted_mo = torch.max(e_pred_mo.data, 1)

            # majority voting
            ensemble_predicted = []

            a = np.array([e_predicted_le.to("cpu").detach().numpy(), e_predicted_re.to("cpu").detach().numpy(),
                          e_predicted_mo.to("cpu").detach().numpy()])

            for i in range(len(a[0])):
                counts = np.bincount(a[:, i])
                e_predicted = np.argmax(counts)
                ensemble_predicted.append(e_predicted.tolist())

            ensemble_predicted = torch.Tensor(ensemble_predicted).to(device).long()
            print(ensemble_predicted)
            total += emotion.size(0)

            e_predicted_list += ensemble_predicted.to("cpu").data.numpy().tolist()
            e_gt_list += emotion.to("cpu").numpy().tolist()

            e_correct += (ensemble_predicted == emotion).sum().item()

    print('Emotion Accuracy of RegionNet: %f' % (e_correct / total))
    print('Confusion Matrix on FER: ')
    print(confusion_matrix(np.array(e_gt_list).ravel().tolist(), np.array(e_predicted_list).ravel().tolist()))


if __name__ == '__main__':
    trainloader, testloader = data_loader.load_data("RAF-Part")
    mv_inference(test_dataloader=testloader)
