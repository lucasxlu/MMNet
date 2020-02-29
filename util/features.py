import os
import sys

import numpy as np
import torch
import torch.nn as nn
from skimage import io
from skimage.transform import resize

sys.path.append('../')
from util.file_utils import mkdirs_if_not_exist
from config.cfg import cfg
from models.vgg import MMNet


def ext_feat(eval_model, img_path):
    """
    extract low-level features from pretrained TreeCNN on GlobalNet
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = io.imread(img_path)
    x = resize(x, (224, 224), anti_aliasing=True)

    x[:, :, 0] -= 131.45376586914062
    x[:, :, 1] -= 103.98748016357422
    x[:, :, 2] -= 91.46234893798828

    x = np.transpose(x, [2, 0, 1])
    x = torch.from_numpy(x).unsqueeze(0).float()

    print(x.shape)
    x = x.to(device)

    for name, module in eval_model.named_children():
        for na, mod in module.named_children():
            if na != 'relu1':
                x = mod.forward(x)
            else:
                return x


if __name__ == '__main__':
    treecnn = TreeCNN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        treecnn = nn.DataParallel(treecnn)
    treecnn = treecnn.to(device)

    print('Loading pre-trained model...')
    treecnn.load_state_dict(torch.load(os.path.join('../main/model/treecnn.pth')))
    treecnn.eval()

    part_name = 'Mouth'

    mkdirs_if_not_exist('./LocalPartFeatures/%s' % part_name)
    base_dir = os.path.join(cfg['root'], 'RAF-Face', 'basic', 'LocalParts', part_name)
    for _ in os.listdir(base_dir):
        ft = ext_feat(treecnn, os.path.join(base_dir, _))
        np.savetxt('./LocalPartFeatures/{0}/{1}.txt'.format(part_name, _), ft.to("cpu").detach().numpy().flatten(),
                   fmt='%f')
