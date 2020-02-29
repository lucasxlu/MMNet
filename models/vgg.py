import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.activation import Swish

FUSE_TYPE = 'Add'
MT = True  # whether to adopt Multi-task mode


class MMNet(nn.Module):
    """
    NN definition of MMNet
    """

    def __init__(self, fuse_type=FUSE_TYPE, init_weights=True):
        super(MMNet, self).__init__()
        assert fuse_type in ['None', 'Avg', 'Add', 'Max', 'Min']
        self.fuse_type = fuse_type

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)  # 224*224
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)  # 224*224
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU()
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112*112

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)  # 112*112
        self.bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)  # 112*112
        self.bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU()
        self.mpool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56*56

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)  # 56*56
        self.bn5 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)  # 56*56
        self.bn6 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)  # 56*56
        self.bn7 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)  # 56*56
        self.bn8 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu8 = nn.ReLU()
        self.mpool8 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28*28

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)  # 28*28
        self.bn9 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)  # 28*28
        self.bn10 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu10 = nn.ReLU()
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)  # 28*28
        self.bn11 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)  # 28*28
        self.bn12 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu12 = nn.ReLU()
        self.mpool12 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14*14

        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)  # 14*14
        self.bn13 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu13 = nn.ReLU()
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)  # 14*14
        self.bn14 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu14 = nn.ReLU()
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)  # 14*14
        self.bn15 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu15 = nn.ReLU()
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)  # 14*14
        self.bn16 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu16 = nn.ReLU()
        self.mpool16 = nn.MaxPool2d(kernel_size=2, stride=2)  # 7*7

        self.conv17 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)  # 7*7
        self.bn17 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu17 = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.emotion_branch = EmotionBranch()
        self.age_branch = AgeBranch()
        self.race_branch = RaceBranch()
        self.gender_branch = GenderBranch()

        if init_weights:
            self._init_imagenet_weights_on_features()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu1(x2)
        x4 = self.conv2(x3)

        if self.fuse_type == 'Avg':
            x4 = (x1 + x4) / 2
        elif self.fuse_type == 'Add':
            x4 = x1 + x4
        elif self.fuse_type == 'Max':
            x4 = torch.max(x1, x4)
        elif self.fuse_type == 'Min':
            x4 = torch.min(x1, x4)
        elif self.fuse_type == 'None':
            x4 = x4

        x5 = self.bn2(x4)
        x6 = self.relu2(x5)
        x7 = self.mpool2(x6)

        x8 = self.conv3(x7)
        x9 = self.bn3(x8)
        x10 = self.relu3(x9)
        x11 = self.conv4(x10)

        if self.fuse_type == 'Avg':
            x11 = (x8 + x11) / 2
        elif self.fuse_type == 'Add':
            x11 = x8 + x11
        elif self.fuse_type == 'Max':
            x11 = torch.max(x8, x11)
        elif self.fuse_type == 'Min':
            x11 = torch.min(x8, x11)
        elif self.fuse_type == 'None':
            x11 = x11

        x12 = self.bn4(x11)
        x13 = self.relu4(x12)
        x14 = self.mpool4(x13)

        x15 = self.conv5(x14)
        x16 = self.bn5(x15)
        x17 = self.relu5(x16)
        x18 = self.conv6(x17)
        x19 = self.bn6(x18)
        x20 = self.relu6(x19)
        x21 = self.conv7(x20)
        x22 = self.bn7(x21)
        x23 = self.relu7(x22)
        x24 = self.conv8(x23)  # 56*56*256

        if self.fuse_type == 'Avg':
            x24 = (x15 + x18 + x21 + x24) / 4
        elif self.fuse_type == 'Add':
            x24 = x15 + x18 + x21 + x24
        elif self.fuse_type == 'Max':
            x24 = torch.max(torch.max(x15, x18), torch.max(x21, x24))
        elif self.fuse_type == 'Min':
            x24 = torch.max(torch.min(x15, x18), torch.min(x21, x24))
        elif self.fuse_type == 'None':
            x24 = x24

        x25 = self.bn8(x24)
        x26 = self.relu8(x25)
        x27 = self.mpool8(x26)

        x28 = self.conv9(x27)
        x29 = self.bn9(x28)
        x30 = self.relu9(x29)
        x31 = self.conv10(x30)
        x32 = self.bn10(x31)
        x33 = self.relu10(x32)
        x34 = self.conv11(x33)
        x35 = self.bn11(x34)
        x36 = self.relu11(x35)
        x37 = self.conv12(x36)  # # 28*28*512

        if self.fuse_type == 'Avg':
            x37 = (x28 + x31 + x34 + x37) / 4
        elif self.fuse_type == 'Add':
            x37 = x28 + x31 + x34 + x37
        elif self.fuse_type == 'Max':
            x37 = torch.max(torch.max(x28, x31), torch.max(x34, x37))
        elif self.fuse_type == 'Min':
            x37 = torch.max(torch.min(x28, x31), torch.min(x34, x37))
        elif self.fuse_type == 'None':
            x37 = x37

        x38 = self.bn12(x37)
        x39 = self.relu12(x38)
        x40 = self.mpool12(x39)

        x41 = self.conv13(x40)
        x42 = self.bn13(x41)
        x43 = self.relu13(x42)
        x44 = self.conv14(x43)
        x45 = self.bn14(x44)
        x46 = self.relu14(x45)
        x47 = self.conv15(x46)
        x48 = self.bn15(x47)
        x49 = self.relu15(x48)
        x50 = self.conv16(x49)  # 14*14*512

        if self.fuse_type == 'Avg':
            x50 = (x41 + x44 + x47 + x50) / 4
        elif self.fuse_type == 'Add':
            x50 = x41 + x44 + x47 + x50
        elif self.fuse_type == 'Max':
            x50 = torch.max(torch.max(x41, x44), torch.max(x47, x50))
        elif self.fuse_type == 'Min':
            x50 = torch.min(torch.max(x41, x44), torch.min(x47, x50))
        elif self.fuse_type == 'None':
            x50 = x50

        x51 = self.bn16(x50)
        x52 = self.relu16(x51)
        x53 = self.mpool16(x52)

        x54 = self.bn17(self.conv17(x53))
        x55 = self.gap(self.relu17(x54))

        x55 = x55.view(-1, self.num_flat_features(x55))
        e_pred = self.emotion_branch(x55)
        a_pred = self.age_branch(x55)
        r_pred = self.race_branch(x55)
        g_pred = self.gender_branch(x55)
        # l_pred = self.ldmk_branch(x55)

        return e_pred, a_pred, r_pred, g_pred

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features

    def _init_imagenet_weights_on_features(self):
        vgg19 = models.vgg19_bn(pretrained=True)
        print('[INFO] Start Initializing CFPNet with vgg19_BN Pretrained Weights...')

        for name, module in self.named_modules():
            if name == 'conv1':
                module.weight.data.copy_(vgg19.features[0].weight.data)
                # module.bias.data.copy_(vgg19.features[0].bias.data)
            # elif name == 'bn1':
            #     module.weight.data.copy_(vgg19.features[1].weight.data)
                # module.bias.data.copy_(vgg19.features[1].bias.data)
            elif name == 'conv2':
                module.weight.data.copy_(vgg19.features[3].weight.data)
                # module.bias.data.copy_(vgg19.features[3].bias.data)
            # elif name == 'bn2':
            #     module.weight.data.copy_(vgg19.features[4].weight.data)
                # module.bias.data.copy_(vgg19.features[4].bias.data)
            elif name == 'conv3':
                module.weight.data.copy_(vgg19.features[7].weight.data)
                # module.bias.data.copy_(vgg19.features[7].bias.data)
            # elif name == 'bn3':
            #     module.weight.data.copy_(vgg19.features[8].weight.data)
                # module.bias.data.copy_(vgg19.features[8].bias.data)
            elif name == 'conv4':
                module.weight.data.copy_(vgg19.features[10].weight.data)
                # module.bias.data.copy_(vgg19.features[10].bias.data)
            # elif name == 'bn4':
            #     module.weight.data.copy_(vgg19.features[11].weight.data)
                # module.bias.data.copy_(vgg19.features[11].bias.data)
            elif name == 'conv5':
                module.weight.data.copy_(vgg19.features[14].weight.data)
                # module.bias.data.copy_(vgg19.features[14].bias.data)
            # elif name == 'bn5':
            #     module.weight.data.copy_(vgg19.features[15].weight.data)
                # module.bias.data.copy_(vgg19.features[15].bias.data)
            elif name == 'conv6':
                module.weight.data.copy_(vgg19.features[17].weight.data)
                # module.bias.data.copy_(vgg19.features[17].bias.data)
            # elif name == 'bn6':
            #     module.weight.data.copy_(vgg19.features[18].weight.data)
                # module.bias.data.copy_(vgg19.features[18].bias.data)
            elif name == 'conv7':
                module.weight.data.copy_(vgg19.features[20].weight.data)
                # module.bias.data.copy_(vgg19.features[20].bias.data)
            # elif name == 'bn7':
            #     module.weight.data.copy_(vgg19.features[21].weight.data)
                # module.bias.data.copy_(vgg19.features[21].bias.data)
            elif name == 'conv8':
                module.weight.data.copy_(vgg19.features[23].weight.data)
                # module.bias.data.copy_(vgg19.features[24].bias.data)
            # elif name == 'bn8':
            #     module.weight.data.copy_(vgg19.features[25].weight.data)
                # module.bias.data.copy_(vgg19.features[25].bias.data)
            elif name == 'conv9':
                module.weight.data.copy_(vgg19.features[27].weight.data)
                # module.bias.data.copy_(vgg19.features[27].bias.data)
            # elif name == 'bn9':
            #     module.weight.data.copy_(vgg19.features[28].weight.data)
                # module.bias.data.copy_(vgg19.features[28].bias.data)
            elif name == 'conv10':
                module.weight.data.copy_(vgg19.features[30].weight.data)
                # module.bias.data.copy_(vgg19.features[30].bias.data)
            # elif name == 'bn10':
            #     module.weight.data.copy_(vgg19.features[31].weight.data)
                # module.bias.data.copy_(vgg19.features[31].bias.data)
            elif name == 'conv11':
                module.weight.data.copy_(vgg19.features[33].weight.data)
                # module.bias.data.copy_(vgg19.features[34].bias.data)
            # elif name == 'bn11':
            #     module.weight.data.copy_(vgg19.features[35].weight.data)
                # module.bias.data.copy_(vgg19.features[35].bias.data)
            elif name == 'conv12':
                module.weight.data.copy_(vgg19.features[36].weight.data)
                # module.bias.data.copy_(vgg19.features[37].bias.data)
            # elif name == 'bn12':
            #     module.weight.data.copy_(vgg19.features[38].weight.data)
                # module.bias.data.copy_(vgg19.features[38].bias.data)
            elif name == 'conv13':
                module.weight.data.copy_(vgg19.features[40].weight.data)
                # module.bias.data.copy_(vgg19.features[40].bias.data)
            # elif name == 'bn13':
            #     module.weight.data.copy_(vgg19.features[41].weight.data)
                # module.bias.data.copy_(vgg19.features[41].bias.data)

        print('[INFO] Finish Initializing CFPNet with vgg19_BN ImageNet Pretrained Weights...')


class SIMNet(nn.Module):
    """
    NN definition of SIMNet
    """

    def __init__(self, fuse_type=FUSE_TYPE):
        super(SIMNet, self).__init__()
        assert fuse_type in ['None', 'Avg', 'Add', 'Max', 'Min']
        self.fuse_type = fuse_type

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)  # 224*224
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)  # 224*224
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU()
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112*112

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)  # 112*112
        self.bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)  # 112*112
        self.bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU()
        self.mpool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56*56

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)  # 56*56
        self.bn5 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)  # 56*56
        self.bn6 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)  # 56*56
        self.bn7 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)  # 56*56
        self.bn8 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu8 = nn.ReLU()
        self.mpool8 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28*28

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)  # 28*28
        self.bn9 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)  # 28*28
        self.bn10 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu10 = nn.ReLU()
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)  # 28*28
        self.bn11 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)  # 28*28
        self.bn12 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu12 = nn.ReLU()
        self.mpool12 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14*14

        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)  # 14*14
        self.bn13 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu13 = nn.ReLU()
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)  # 14*14
        self.bn14 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu14 = nn.ReLU()
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)  # 14*14
        self.bn15 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu15 = nn.ReLU()
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)  # 14*14
        self.bn16 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu16 = nn.ReLU()
        self.mpool16 = nn.MaxPool2d(kernel_size=2, stride=2)  # 7*7

        self.conv17 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)  # 7*7
        self.bn17 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu17 = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.emotion_branch = EmotionBranch()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu1(x2)
        x4 = self.conv2(x3)

        if self.fuse_type == 'Avg':
            x4 = (x1 + x4) / 2
        elif self.fuse_type == 'Add':
            x4 = x1 + x4
        elif self.fuse_type == 'Max':
            x4 = torch.max(x1, x4)
        elif self.fuse_type == 'Min':
            x4 = torch.min(x1, x4)
        elif self.fuse_type == 'None':
            x4 = x4

        x5 = self.bn2(x4)
        x6 = self.relu2(x5)
        x7 = self.mpool2(x6)

        x8 = self.conv3(x7)
        x9 = self.bn3(x8)
        x10 = self.relu3(x9)
        x11 = self.conv4(x10)

        if self.fuse_type == 'Avg':
            x11 = (x8 + x11) / 2
        elif self.fuse_type == 'Add':
            x11 = x8 + x11
        elif self.fuse_type == 'Max':
            x11 = torch.max(x8, x11)
        elif self.fuse_type == 'Min':
            x11 = torch.min(x8, x11)
        elif self.fuse_type == 'None':
            x11 = x11

        x12 = self.bn4(x11)
        x13 = self.relu4(x12)
        x14 = self.mpool4(x13)

        x15 = self.conv5(x14)
        x16 = self.bn5(x15)
        x17 = self.relu5(x16)
        x18 = self.conv6(x17)
        x19 = self.bn6(x18)
        x20 = self.relu6(x19)
        x21 = self.conv7(x20)
        x22 = self.bn7(x21)
        x23 = self.relu7(x22)
        x24 = self.conv8(x23)  # 56*56*256

        if self.fuse_type == 'Avg':
            x24 = (x15 + x18 + x21 + x24) / 4
        elif self.fuse_type == 'Add':
            x24 = x15 + x18 + x21 + x24
        elif self.fuse_type == 'Max':
            x24 = torch.max(torch.max(x15, x18), torch.max(x21, x24))
        elif self.fuse_type == 'Min':
            x24 = torch.max(torch.min(x15, x18), torch.min(x21, x24))
        elif self.fuse_type == 'None':
            x24 = x24

        x25 = self.bn8(x24)
        x26 = self.relu8(x25)
        x27 = self.mpool8(x26)

        x28 = self.conv9(x27)
        x29 = self.bn9(x28)
        x30 = self.relu9(x29)
        x31 = self.conv10(x30)
        x32 = self.bn10(x31)
        x33 = self.relu10(x32)
        x34 = self.conv11(x33)
        x35 = self.bn11(x34)
        x36 = self.relu11(x35)
        x37 = self.conv12(x36)  # # 28*28*512

        if self.fuse_type == 'Avg':
            x37 = (x28 + x31 + x34 + x37) / 4
        elif self.fuse_type == 'Add':
            x37 = x28 + x31 + x34 + x37
        elif self.fuse_type == 'Max':
            x37 = torch.max(torch.max(x28, x31), torch.max(x34, x37))
        elif self.fuse_type == 'Min':
            x37 = torch.max(torch.min(x28, x31), torch.min(x34, x37))
        elif self.fuse_type == 'None':
            x37 = x37

        x38 = self.bn12(x37)
        x39 = self.relu12(x38)
        x40 = self.mpool12(x39)

        x41 = self.conv13(x40)
        x42 = self.bn13(x41)
        x43 = self.relu13(x42)
        x44 = self.conv14(x43)
        x45 = self.bn14(x44)
        x46 = self.relu14(x45)
        x47 = self.conv15(x46)
        x48 = self.bn15(x47)
        x49 = self.relu15(x48)
        x50 = self.conv16(x49)  # 14*14*512

        if self.fuse_type == 'Avg':
            x50 = (x41 + x44 + x47 + x50) / 4
        elif self.fuse_type == 'Add':
            x50 = x41 + x44 + x47 + x50
        elif self.fuse_type == 'Max':
            x50 = torch.max(torch.max(x41, x44), torch.max(x47, x50))
        elif self.fuse_type == 'Min':
            x50 = torch.min(torch.max(x41, x44), torch.min(x47, x50))
        elif self.fuse_type == 'None':
            x50 = x50

        x51 = self.bn16(x50)
        x52 = self.relu16(x51)
        x53 = self.mpool16(x52)

        x54 = self.bn17(self.conv17(x53))
        x55 = self.gap(self.relu17(x54))

        x55 = x55.view(-1, self.num_flat_features(x55))
        e_pred = self.emotion_branch(x55)

        return e_pred

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class FaceBlock(nn.Module):
    """
    Face Block definition
    from Conv5 to MaxPool16
    """

    def __init__(self, fuse_type=FUSE_TYPE):
        super(FaceBlock, self).__init__()
        assert fuse_type in ['None', 'Avg', 'Add', 'Max', 'Min']

        self.emotion_branch = EmotionBranch(7)
        self.age_branch = AgeBranch(5)
        self.race_branch = RaceBranch(3)
        self.gender_branch = GenderBranch(3)
        self.fuse_type = fuse_type
        # self.ldmk_branch = LdmkBranch(10)

    def forward(self, x):
        x1 = self.fbconv1(x)
        x2 = self.fbbn1(x1)
        x3 = self.fbrelu1(x2)
        x4 = self.fbconv2(x3)
        x5 = self.fbbn2(x4)
        x6 = self.fbrelu2(x5)
        x7 = self.fbconv3(x6)
        x8 = self.fbbn3(x7)
        x9 = self.fbrelu3(x8)
        x10 = self.fbconv4(x9)  # 56*56*256

        if self.fuse_type == 'Avg':
            x10 = (x1 + x4 + x7 + x10) / 4
        elif self.fuse_type == 'Add':
            x10 = x1 + x4 + x7 + x10
        elif self.fuse_type == 'Max':
            x10 = torch.max(torch.max(x1, x4), torch.max(x7, x10))
        elif self.fuse_type == 'Min':
            x10 = torch.min(torch.min(x1, x4), torch.min(x7, x10))
        elif self.fuse_type == 'None':
            x10 = x10

        x11 = self.fbbn4(x10)
        x12 = self.fbrelu4(x11)
        x13 = self.fbmpool4(x12)

        x14 = self.fbconv5(x13)
        x15 = self.fbbn5(x14)
        x16 = self.fbrelu5(x15)
        x17 = self.fbconv6(x16)
        x18 = self.fbbn6(x17)
        x19 = self.fbrelu6(x18)
        x20 = self.fbconv7(x19)
        x21 = self.fbbn7(x20)
        x22 = self.fbrelu7(x21)
        x23 = self.fbconv8(x22)  # # 28*28*512

        if self.fuse_type == 'Avg':
            x23 = (x14 + x17 + x20 + x23) / 4
        elif self.fuse_type == 'Add':
            x23 = x14 + x17 + x20 + x23
        elif self.fuse_type == 'Max':
            x23 = torch.max(torch.max(x14, x17), torch.max(x20, x23))
        elif self.fuse_type == 'Min':
            x23 = torch.min(torch.min(x14, x17), torch.min(x20, x23))
        elif self.fuse_type == 'None':
            x23 = x23

        x24 = self.fbbn8(x23)
        x25 = self.fbrelu8(x24)
        x26 = self.fbmpool8(x25)

        x27 = self.fbconv9(x26)
        x28 = self.fbbn9(x27)
        x29 = self.fbrelu9(x28)
        x30 = self.fbconv10(x29)
        x31 = self.fbbn10(x30)
        x32 = self.fbrelu10(x31)
        x33 = self.fbconv11(x32)
        x34 = self.fbbn11(x33)
        x35 = self.fbrelu11(x34)
        x36 = self.fbconv12(x35)  # 14*14*512

        if self.fuse_type == 'Avg':
            x36 = (x27 + x30 + x33 + x36) / 4
        elif self.fuse_type == 'Add':
            x36 = x27 + x30 + x33 + x36
        elif self.fuse_type == 'Max':
            x36 = torch.max(torch.max(x27, x30), torch.max(x33, x36))
        elif self.fuse_type == 'Min':
            x36 = torch.min(torch.min(x27, x30), torch.min(x33, x36))
        elif self.fuse_type == 'None':
            x36 = x36

        x37 = self.fbbn12(x36)
        x38 = self.fbrelu12(x37)
        x39 = self.fbmpool12(x38)

        x40 = self.fbbn13(self.fbconv13(x39))
        x41 = self.gap(self.fbrelu13(x40))

        x41 = x41.view(-1, self.num_flat_features(x41))
        e_pred = self.emotion_branch(x41)
        a_pred = self.age_branch(x41)
        r_pred = self.race_branch(x41)
        g_pred = self.gender_branch(x41)
        # l_pred = self.ldmk_branch(x41)

        return e_pred, a_pred, r_pred, g_pred

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class EmotionBranch(nn.Module):
    """
    Emotion Branch for Facial Expression Classification
    FC layers
    """

    def __init__(self, num_classes=7):
        super(EmotionBranch, self).__init__()
        self.emlinear1 = nn.Linear(512, 512, bias=False)
        self.emrelu1 = nn.ReLU()
        self.emlinear2 = nn.Linear(512, 256, bias=False)
        self.emrelu2 = nn.ReLU()
        self.emlinear3 = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        x1 = self.emlinear1(x)
        x1 = F.dropout(x1, training=self.training)
        x2 = self.emrelu1(x1)
        x3 = self.emlinear2(x2)
        x3 = F.dropout(x3, training=self.training)
        x4 = self.emrelu2(x3)
        x5 = self.emlinear3(x4)

        return x5

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class AgeBranch(nn.Module):
    """
    Age Branch for Age Classification
    FC layers
    """

    def __init__(self, num_classes=5):
        super(AgeBranch, self).__init__()
        self.alinear1 = nn.Linear(512, 512, bias=False)
        self.arelu1 = nn.ReLU()
        self.alinear2 = nn.Linear(512, 256, bias=False)
        self.arelu2 = nn.ReLU()
        self.alinear3 = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        x1 = self.alinear1(x)
        x1 = F.dropout(x1, training=self.training)
        x2 = self.arelu1(x1)
        x3 = self.alinear2(x2)
        x3 = F.dropout(x3, training=self.training)
        x4 = self.arelu2(x3)
        x5 = self.alinear3(x4)

        return x5

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class GenderBranch(nn.Module):
    """
    Gender Branch for Gender Classification
    FC layers
    """

    def __init__(self, num_classes=3):
        super(GenderBranch, self).__init__()
        self.glinear1 = nn.Linear(512, 512, bias=False)
        self.grelu1 = nn.ReLU()
        self.glinear2 = nn.Linear(512, 256, bias=False)
        self.grelu2 = nn.ReLU()
        self.glinear3 = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        x1 = self.glinear1(x)
        x1 = F.dropout(x1, training=self.training)
        x2 = self.grelu1(x1)
        x3 = self.glinear2(x2)
        x3 = F.dropout(x3, training=self.training)
        x4 = self.grelu2(x3)
        x5 = self.glinear3(x4)

        return x5

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class RaceBranch(nn.Module):
    """
    Race Branch for Race Classification
    FC layers
    """

    def __init__(self, num_classes=3):
        super(RaceBranch, self).__init__()
        self.rlinear1 = nn.Linear(512, 512, bias=False)
        self.rrelu1 = nn.ReLU()
        self.rlinear2 = nn.Linear(512, 256, bias=False)
        self.rrelu2 = nn.ReLU()
        self.rlinear3 = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        x1 = self.rlinear1(x)
        x1 = F.dropout(x1, training=self.training)
        x2 = self.rrelu1(x1)
        x3 = self.rlinear2(x2)
        x3 = F.dropout(x3, training=self.training)
        x4 = self.rrelu2(x3)
        x5 = self.rlinear3(x4)

        return x5

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class LdmkBranch(nn.Module):
    """
    Landmark Branch for Landmark localization
    FC layers
    """

    def __init__(self, num_classes=10):
        super(LdmkBranch, self).__init__()
        self.llinear1 = nn.Linear(512, 256, bias=False)
        self.lrelu1 = nn.ReLU()
        self.llinear2 = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        x1 = self.llinear1(x)
        x2 = self.lrelu1(x1)
        x3 = self.llinear2(x2)

        return x3

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


if __name__ == '__main__':
    vgg19 = models.vgg19_bn(pretrained=False)
    print(vgg19)