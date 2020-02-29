"""
inference code
"""
import sys
import time
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage import io
from torchvision.transforms import transforms

sys.path.append('../')
from models.vgg import MMNet


class MMNetRecognizer:
    """
    MMNet Recognizer Class Wrapper
    """

    def __init__(self, pretrained_model_path='MMNet.pth'):
        model = MMNet()
        model = model.float()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # model.load_state_dict(torch.load(pretrained_model_path))

        if torch.cuda.device_count() > 1:
            print("We are running on", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(pretrained_model_path))
        else:
            state_dict = torch.load(pretrained_model_path)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        model.to(device)
        model.eval()

        self.device = device
        self.model = model

    def infer(self, img_file):
        tik = time.time()
        img = io.imread(img_file)
        img = Image.fromarray(img.astype(np.uint8))

        preprocess = transforms.Compose([
            transforms.Resize(227),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = preprocess(img)
        img.unsqueeze_(0)

        img = img.to(self.device)

        e_pred, a_pred, r_pred, g_pred = self.model.forward(img)
        tok = time.time()

        _, e_predicted = torch.max(e_pred.data, 1)
        _, a_predicted = torch.max(a_pred.data, 1)
        _, r_predicted = torch.max(r_pred.data, 1)
        _, g_predicted = torch.max(g_pred.data, 1)

        if int(g_predicted.to("cpu")) == 0:
            g_pred = 'male'
        elif int(g_predicted.to("cpu")) == 1:
            g_pred = 'female'
        elif int(g_predicted.to("cpu")) == 2:
            g_pred = 'unsure'

        if int(r_predicted.to("cpu")) == 0:
            r_pred = 'Caucasian'
        elif int(r_predicted.to("cpu")) == 1:
            r_pred = 'African-American'
        elif int(r_predicted.to("cpu")) == 2:
            r_pred = 'Asian'

        if int(a_predicted.to("cpu")) == 0:
            a_pred = '0-3'
        elif int(a_predicted.to("cpu")) == 1:
            a_pred = '4-19'
        elif int(a_predicted.to("cpu")) == 2:
            a_pred = '20-39'
        elif int(a_predicted.to("cpu")) == 3:
            a_pred = '40-69'
        elif int(a_predicted.to("cpu")) == 4:
            a_pred = '70+'

        if int(e_predicted.to("cpu")) == 0:
            e_pred = 'Surprise'
        elif int(e_predicted.to("cpu")) == 1:
            e_pred = 'Fear'
        elif int(e_predicted.to("cpu")) == 2:
            e_pred = 'Disgust'
        elif int(e_predicted.to("cpu")) == 3:
            e_pred = 'Happiness'
        elif int(e_predicted.to("cpu")) == 4:
            e_pred = 'Sadness'
        elif int(e_predicted.to("cpu")) == 5:
            e_pred = 'Anger'
        elif int(e_predicted.to("cpu")) == 6:
            e_pred = 'Neutral'

        return {
            'status': 0,
            'message': 'success',
            'elapse': tok - tik,
            'results': {
                'gender': g_pred,
                'emotion': e_pred,
                'race': r_pred,
                'age': a_pred,
                'elapse': tok - tik
            }
        }


if __name__ == '__main__':
    mmnet_recognizer = MMNetRecognizer()
    pprint(mmnet_recognizer.infer('test.jpg'))
