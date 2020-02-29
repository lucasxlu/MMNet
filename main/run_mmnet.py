import os
import sys
import time
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

from sklearn.metrics import confusion_matrix, mean_absolute_error

sys.path.append('../')
from data import data_loader
from models.losses import MTLoss
from models.vgg import MMNet
from util.file_utils import mkdirs_if_not_exist
from config.cfg import cfg


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, inference=False):
    """
    train model
    :param model:
    :param dataloaders:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :param inference:
    :return:
    """
    print(model)
    model_name = model.__class__.__name__
    model = model.float()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    bs = cfg["RAF-Face"]['batch_size']

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    dataset_sizes = {x: dataloaders[x].__len__() * bs for x in ['train', 'test']}

    for k, v in dataset_sizes.items():
        print('Dataset size of {0} is {1}...'.format(k, v))

    if not inference:
        print('Start training %s...' % model_name)
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        # best_acc = 0.0
        best_avg_cm = 0.0

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
                e_running_corrects = 0
                a_running_corrects = 0
                r_running_corrects = 0
                g_running_corrects = 0

                e_running_gts = []
                e_running_preds = []
                # ldmk_running_gts = []
                # ldmk_running_preds = []

                # Iterate over data.
                for i, data in enumerate(dataloaders[phase], 0):

                    # inputs, gender, race, age, emotion, landmark = data['image'], data['gender'], data['race'], data[
                    #     'age'], data['emotion'], data['landmark']

                    inputs, gender, race, age, emotion = data['image'], data['gender'], data['race'], data['age'], data[
                        'emotion']

                    inputs = inputs.to(device)
                    gender = gender.to(device)
                    race = race.to(device)
                    age = age.to(device)
                    emotion = emotion.to(device)
                    # landmark = landmark.float().to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # e_pred, a_pred, r_pred, g_pred, l_pred = model(inputs)
                        e_pred, a_pred, r_pred, g_pred = model(inputs)
                        # loss = criterion(g_pred, gender, r_pred, race, a_pred, age, e_pred, emotion, l_pred, landmark)
                        loss = criterion(g_pred, gender, r_pred, race, a_pred, age, e_pred, emotion)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    _, g_predicted = torch.max(g_pred.data, 1)
                    _, r_predicted = torch.max(r_pred.data, 1)
                    _, a_predicted = torch.max(a_pred.data, 1)
                    _, e_predicted = torch.max(e_pred.data, 1)

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    e_running_corrects += torch.sum(e_predicted == emotion.data)
                    a_running_corrects += torch.sum(a_predicted == age.data)
                    g_running_corrects += torch.sum(g_predicted == gender.data)
                    r_running_corrects += torch.sum(r_predicted == race.data)

                    e_running_preds += e_predicted.to("cpu").detach().numpy().tolist()
                    e_running_gts += emotion.to("cpu").detach().numpy().tolist()
                    # ldmk_running_preds += l_pred.to("cpu").detach().numpy().tolist()
                    # ldmk_running_gts += landmark.to("cpu").detach().numpy().tolist()

                if phase == 'train':
                    if torch.__version__ > '1.1.0':
                        scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                e_epoch_acc = e_running_corrects.double() / dataset_sizes[phase]
                a_epoch_acc = a_running_corrects.double() / dataset_sizes[phase]
                r_epoch_acc = r_running_corrects.double() / dataset_sizes[phase]
                g_epoch_acc = g_running_corrects.double() / dataset_sizes[phase]

                running_cm = np.array(confusion_matrix(e_running_gts, e_running_preds))
                running_cm = running_cm.astype('float') / running_cm.sum(axis=1)[:, np.newaxis]

                running_avg_cm = sum([running_cm[i][i] for i in range(len(running_cm))]) / len(running_cm)

                # ldmk_mae = mean_absolute_error(ldmk_running_gts, ldmk_running_preds)

                print(
                    '[{}] Loss: {:.4f} Emotion Acc: {:.4f} Age Acc: {:.4f} Race Acc: {:.4f} Gender Acc: {:.4f}'.format(
                        phase, epoch_loss, e_epoch_acc, a_epoch_acc, r_epoch_acc, g_epoch_acc))

                # deep copy the model
                if phase == 'test' and running_avg_cm > best_avg_cm:
                    tmp_e_correct = 0
                    tmp_a_correct = 0
                    tmp_g_correct = 0
                    tmp_r_correct = 0
                    # tmp_ldmk_gts = []
                    # tmp_ldmk_preds = []

                    tmp_total = 0
                    tmp_y_pred = []
                    tmp_y_true = []
                    tmp_filenames = []

                    for data in dataloaders['test']:
                        images, gender, race, age, emotion, filename = data['image'], data['gender'], data['race'], \
                                                                       data['age'], data['emotion'], data['filename']

                        images = images.to(device)
                        gender = gender.to(device)
                        race = race.to(device)
                        age = age.to(device)
                        emotion = emotion.to(device)
                        # landmark = landmark.to(device)

                        # e_pred, a_pred, r_pred, g_pred, l_pred = model(images)
                        e_pred, a_pred, r_pred, g_pred = model(images)
                        _, g_predicted = torch.max(g_pred.data, 1)
                        _, r_predicted = torch.max(r_pred.data, 1)
                        _, a_predicted = torch.max(a_pred.data, 1)
                        _, e_predicted = torch.max(e_pred.data, 1)

                        tmp_total += e_predicted.size(0)
                        tmp_e_correct += (e_predicted == emotion).sum().item()
                        tmp_a_correct += (a_predicted == age).sum().item()
                        tmp_r_correct += (r_predicted == race).sum().item()
                        tmp_g_correct += (g_predicted == gender).sum().item()

                        tmp_y_pred += e_predicted.to("cpu").detach().numpy().tolist()
                        tmp_y_true += emotion.to("cpu").detach().numpy().tolist()
                        tmp_filenames += filename

                        # tmp_ldmk_gts += landmark.to('cpu').detach().numpy().flatten().tolist()
                        # tmp_ldmk_preds += l_pred.to('cpu').detach().numpy().flatten().tolist()

                    tmp_e_acc = tmp_e_correct / tmp_total
                    tmp_a_acc = tmp_a_correct / tmp_total
                    tmp_g_acc = tmp_g_correct / tmp_total
                    tmp_r_acc = tmp_r_correct / tmp_total

                    print('Confusion Matrix of {0} on test set: '.format(model_name))
                    cm = confusion_matrix(tmp_y_true, tmp_y_pred)
                    print(cm)
                    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
                    cm = np.array(cm)

                    avg_cm = sum([cm[i][i] for i in range(len(cm))]) / len(cm)
                    # ldmk_mae = mean_absolute_error(tmp_ldmk_gts, tmp_ldmk_preds)

                    print('Emotion Accuracy = {0}'.format(tmp_e_acc))
                    print('Age Accuracy = {0}'.format(tmp_a_acc))
                    print('Gender Accuracy = {0}'.format(tmp_g_acc))
                    print('Race Accuracy = {0}'.format(tmp_r_acc))
                    print('Avg Confusion Matrix = {0}'.format(avg_cm))
                    # print('Landmark MAE = {0}'.format(ldmk_mae))

                    precisions = []
                    recalls = []

                    for i in range(len(cm)):
                        precisions.append(cm[i][i] / sum(cm[:, i].tolist()))
                        recalls.append(cm[i][i] / sum(cm[i, :].tolist()))

                    print("Precision of {0} on test set = {1}".format(model_name,
                                                                      sum(precisions) / len(precisions)))
                    print(
                        "Recall of {0} on test set = {1}".format(model_name, sum(recalls) / len(recalls)))

                    # best_acc = e_epoch_acc
                    best_avg_cm = running_avg_cm
                    best_model_wts = copy.deepcopy(model.state_dict())

                    model.load_state_dict(best_model_wts)
                    model_path_dir = './model'
                    mkdirs_if_not_exist(model_path_dir)
                    torch.save(model.state_dict(),
                               './model/{0}_best_epoch-{1}.pth'.format(model_name, epoch))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best test Avg CM: {:4f}'.format(best_avg_cm))

        # load best model weights
        model.load_state_dict(best_model_wts)
        model_path_dir = './model'
        mkdirs_if_not_exist(model_path_dir)
        torch.save(model.state_dict(), './model/%s.pth' % model_name)

    else:
        print('Start testing %s...' % model_name)
        model.load_state_dict(torch.load(os.path.join('./model/%s.pth' % model_name)))

    model.eval()

    g_correct = 0
    r_correct = 0
    a_correct = 0
    e_correct = 0
    total = 0
    e_predicted_list = []
    e_gt_list = []
    # ldmk_gts = []
    # ldmk_preds = []

    probs = []
    filenames = []

    with torch.no_grad():
        for data in dataloaders['test']:
            images, gender, race, age, emotion, filename = data['image'], data['gender'], data['race'], data[
                'age'], data['emotion'], data['filename']
            images = images.to(device)
            gender = gender.to(device)
            race = race.to(device)
            age = age.to(device)
            emotion = emotion.to(device)
            # landmark = landmark.to(device)

            e_pred, a_pred, r_pred, g_pred = model(images)

            outputs = F.softmax(e_pred)
            # get TOP-K output labels and corresponding probabilities
            topK_prob, topK_label = torch.topk(outputs, 2)
            probs += topK_prob.to("cpu").detach().numpy().tolist()

            filenames += filename

            _, g_predicted = torch.max(g_pred.data, 1)
            _, r_predicted = torch.max(r_pred.data, 1)
            _, a_predicted = torch.max(a_pred.data, 1)
            _, e_predicted = torch.max(e_pred.data, 1)

            total += emotion.size(0)

            g_correct += (g_predicted == gender).sum().item()
            r_correct += (r_predicted == race).sum().item()
            a_correct += (a_predicted == age).sum().item()
            e_correct += (e_predicted == emotion).sum().item()

            e_predicted_list += e_predicted.to("cpu").detach().numpy().tolist()
            e_gt_list += emotion.to("cpu").detach().numpy().tolist()

            # ldmk_gts += landmark.to('cpu').detach().numpy().flatten().tolist()
            # ldmk_preds += l_pred.to('cpu').detach().numpy().flatten().tolist()

    print('Race Accuracy of TreeCNN: %f' % (r_correct / total))
    print('Gender Accuracy of TreeCNN: %f' % (g_correct / total))
    print('Age Accuracy of TreeCNN: %f' % (a_correct / total))
    print('Emotion Accuracy of TreeCNN: %f' % (e_correct / total))

    cm = confusion_matrix(np.array(e_gt_list).ravel().tolist(), np.array(e_predicted_list).ravel().tolist())
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=4)
    print('Confusion Matrix on FER: ')
    print(cm)
    print('Avg confusion matrix = {0}'.format(np.average(np.array([cm[i][i] for i in range(len(cm))]))))
    # ldmk_mae = mean_absolute_error(ldmk_gts, ldmk_preds)
    # print('Landmark MAE = {0}'.format(ldmk_mae))

    print('Output CSV...')
    col = ['filename', 'gt', 'pred', 'prob']
    df = pd.DataFrame([[filenames[i], e_gt_list[i], e_predicted_list[i], probs[i][0]] for i in range(len(filenames))],
                      columns=col)
    df.to_csv("./RAF_DB_%s.csv" % model_name, index=False)
    print('CSV has been generated...')


def main(train_dataloader, test_dataloader):
    """
    train and eval on RAF-DB
    :param train_dataloader:
    :param test_dataloader:
    :return:
    """
    mmnet = MMNet()
    criterion = MTLoss()

    optimizer = optim.SGD(mmnet.parameters(), lr=cfg['RAF-Face']['init_lr'], momentum=0.9, weight_decay=cfg[
        'RAF-Face']['weight_decay'])

    # optimizer = optim.RMSprop(treecnn.parameters(), lr=cfg['RAF-Face']['init_lr'], weight_decay=cfg[
    #     'RAF-Face']['weight_decay'])

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg['RAF-Face']['lr_decay_step'], gamma=0.1)

    train_model(model=mmnet, dataloaders={'train': train_dataloader, 'test': test_dataloader},
                criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler, num_epochs=
                cfg['RAF-Face']['epoch'], inference=False)


if __name__ == '__main__':
    trainloader, testloader = data_loader.load_data("RAF-Face")
    main(trainloader, testloader)
