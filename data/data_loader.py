import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.append('../')
from config.cfg import cfg
from data.datasets import RafFaceDataset, UTKFaceDataset, RafPartDataset, FER2013Dataset


def load_data(dataset_name):
    """
    load dataset
    :param dataset_name:
    :return:
    """
    batch_size = cfg[dataset_name]['batch_size']

    if dataset_name == 'RAF-Face':
        print('loading %s dataset...' % dataset_name)
        train_dataset = RafFaceDataset(train=True, type='basic',
                                       transform=transforms.Compose([
                                           transforms.Resize(224),
                                           transforms.ColorJitter(),
                                           transforms.RandomRotation(30),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                                           )
                                       ]))

        weights = []
        for sample in train_dataset:
            label = sample['emotion']
            if label == 0:
                weights.append(3.68)
            elif label == 1:
                weights.append(16.78)
            elif label == 2:
                weights.append(6.8)
            elif label == 3:
                weights.append(1)
            elif label == 4:
                weights.append(2.42)
            elif label == 5:
                weights.append(6.87)
            elif label == 6:
                weights.append(1.86)
            else:
                print('label error')

        weighted_random_sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)

        trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=50, pin_memory=True,
                                 sampler=weighted_random_sampler)

        test_dataset = RafFaceDataset(train=False, type='basic',
                                      transform=transforms.Compose([
                                          transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ]))
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=50, pin_memory=True)

        return trainloader, testloader
    elif dataset_name == 'UTKFace':
        print('loading %s dataset...' % dataset_name)
        train_dataset = UTKFaceDataset(train=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(224),
                                           transforms.ColorJitter(),
                                           transforms.RandomRotation(30),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               mean=[0.5, 0.5, 0.5],
                                               std=[0.5, 0.5, 0.5])
                                       ]))
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=50, pin_memory=True)

        test_dataset = UTKFaceDataset(train=False,
                                      transform=transforms.Compose([
                                          transforms.Resize(224),
                                          transforms.ColorJitter(),
                                          transforms.RandomRotation(30),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              mean=[0.5, 0.5, 0.5],
                                              std=[0.5, 0.5, 0.5])
                                      ]))
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=50, pin_memory=True)

        return trainloader, testloader
    elif dataset_name == 'RAF-Part':
        print('loading %s dataset...' % dataset_name)
        train_dataset = RafPartDataset(train=True, type='basic',
                                       transform=transforms.Compose([
                                           transforms.Resize((48, 64)),
                                           transforms.CenterCrop(48),
                                           transforms.ColorJitter(),
                                           transforms.RandomRotation(15),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                                           )
                                       ]))
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=50,
                                 pin_memory=True)
        test_dataset = RafPartDataset(train=False, type='basic',
                                      transform=transforms.Compose([
                                          transforms.Resize((48, 64)),
                                          transforms.CenterCrop(48),
                                          transforms.ColorJitter(),
                                          transforms.RandomRotation(15),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                                          )
                                      ]))
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=50,
                                pin_memory=True)

        return trainloader, testloader
    elif dataset_name == 'FER2013':
        print('loading %s dataset...' % dataset_name)
        train_dataset = FER2013Dataset(train=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(227),
                                           transforms.CenterCrop(224),
                                           transforms.ColorJitter(),
                                           transforms.RandomRotation(15),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                                           )
                                       ]))

        weights = []
        for sample in train_dataset:
            label = sample['emotion']
            if label == 0:
                weights.append(1.81)
            elif label == 1:
                weights.append(16.55)
            elif label == 2:
                weights.append(1.76)
            elif label == 3:
                weights.append(1)
            elif label == 4:
                weights.append(1.49)
            elif label == 5:
                weights.append(2.28)
            elif label == 6:
                weights.append(1.45)
            else:
                print('label error')

        weighted_random_sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)

        trainloader = DataLoader(train_dataset, batch_size=batch_size, sampler=weighted_random_sampler,
                                 num_workers=50, pin_memory=True)

        test_dataset = FER2013Dataset(train=False,
                                      transform=transforms.Compose([
                                          transforms.Resize(224),
                                          transforms.ColorJitter(),
                                          transforms.RandomRotation(15),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                                          )
                                      ]))
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=50,
                                pin_memory=True)

        return trainloader, testloader
    else:
        print('Error! Invalid dataset name~')
        sys.exit(0)
