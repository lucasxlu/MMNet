import sys
import os
import math

import cv2
import numpy as np
import pandas as pd
from skimage import io
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.color import gray2rgb
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

sys.path.append('../')
from config.cfg import cfg


class RafFaceDataset(Dataset):
    """
    RAF-Face dataset for Face Expression Recognition
    """

    def __init__(self, train=True, type='basic', transform=None):
        manual_annotation_dir = os.path.join(cfg['root'], 'RAF-Face', '%s/Annotation/manual' % type)
        emotion_label_txt_path = os.path.join(cfg['root'], 'RAF-Face', "%s/EmoLabel/list_patition_label.txt" % type)

        emotion_dict = dict(np.loadtxt(emotion_label_txt_path, dtype=np.str))

        if train:
            face_files = []
            genders = []
            races = []
            ages = []
            emotions = []
            ldmks = []
            for _ in os.listdir(manual_annotation_dir):
                if _.startswith('train_'):
                    face_fname = _.replace('_manu_attri', '_aligned').replace('.txt', '.jpg')
                    face_files.append(os.path.join(cfg['root'], 'RAF-Face', '%s/Image/aligned' % type, face_fname))
                    with open(os.path.join(manual_annotation_dir, _), mode='rt') as f:
                        manu_info_list = f.readlines()
                    genders.append(int(manu_info_list[5]))
                    races.append(int(manu_info_list[6]))
                    ages.append(int(manu_info_list[7]))
                    emotions.append(int(emotion_dict[face_fname.replace('_aligned', '')].strip()) - 1)
                    ldmks.append(np.array([[[float(_.replace('\n', ''))] for _ in line.split('\t')] for line in
                                           manu_info_list[0:5]]).flatten().tolist())
        else:
            face_files = []
            genders = []
            races = []
            ages = []
            emotions = []
            ldmks = []
            for _ in os.listdir(manual_annotation_dir):
                if _.startswith('test_'):
                    face_fname = _.replace('_manu_attri', '_aligned').replace('.txt', '.jpg')
                    face_files.append(os.path.join(cfg['root'], 'RAF-Face', '%s/Image/aligned' % type, face_fname))
                    with open(os.path.join(manual_annotation_dir, _), mode='rt') as f:
                        manu_info_list = f.readlines()
                    genders.append(int(manu_info_list[5]))
                    races.append(int(manu_info_list[6]))
                    ages.append(int(manu_info_list[7]))
                    emotions.append(int(emotion_dict[face_fname.replace('_aligned', '')].strip()) - 1)
                    ldmks.append(np.array([[[float(_.replace('\n', ''))] for _ in line.split('\t')] for line in
                                           manu_info_list[0:5]]).flatten().tolist())

        self.face_files = face_files
        self.genders = genders
        self.races = races
        self.ages = ages
        self.emotions = emotions
        self.ldmks = ldmks

        self.transform = transform

    def __len__(self):
        return len(self.face_files)

    def __getitem__(self, idx):
        image = io.imread(self.face_files[idx])
        gender = self.genders[idx]
        race = self.races[idx]
        age = self.ages[idx]
        emotion = self.emotions[idx]
        ldmk = self.ldmks[idx]

        sample = {'image': image, 'gender': gender, 'race': race, 'age': age, 'emotion': emotion,
                  'landmark': np.array(ldmk), 'filename': self.face_files[idx]}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class RafPartDataset(Dataset):
    """
    RAF-Face dataset for Local Part
    """

    def __init__(self, train=True, type='basic', part_name="Mouth", transform=None):
        """

        :param train:
        :param type:
        :param part_name: Mouth, LeftEye, RightEye, Nose
        :param transform:
        """
        # manual_annotation_dir = os.path.join(cfg['root'], 'RAF-Face', '%s/Annotation/manual' % type)
        emotion_label_txt_path = os.path.join(cfg['root'], 'RAF-Face', "%s/EmoLabel/list_patition_label.txt" % type)
        local_part_img_dir = os.path.join(cfg['root'], 'RAF-Face', '{0}/LocalParts/{1}'.format(type, part_name))

        emotion_dict = dict(np.loadtxt(emotion_label_txt_path, dtype=np.str))

        if train:
            local_part_imgs = []
            emotions = []
            for _ in os.listdir(local_part_img_dir):
                if _.startswith('train_'):
                    local_part_imgs.append(os.path.join(local_part_img_dir, _))
                    emotions.append(int(emotion_dict[_.replace('_aligned', '')].strip()) - 1)

        else:
            local_part_imgs = []
            emotions = []
            for _ in os.listdir(local_part_img_dir):
                if _.startswith('test_'):
                    local_part_imgs.append(os.path.join(local_part_img_dir, _))
                    emotions.append(int(emotion_dict[_.replace('_aligned', '')].strip()) - 1)

        self.local_part_imgs = local_part_imgs
        self.emotions = emotions

        self.transform = transform

    def __len__(self):
        return len(self.local_part_imgs)

    def __getitem__(self, idx):
        image = io.imread(self.local_part_imgs[idx])
        emotion = self.emotions[idx]

        sample = {'image': image, 'emotion': emotion, 'filename': self.local_part_imgs[idx]}

        if self.transform:
            trans_image = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))
            sample['image'] = trans_image

        return sample


class CelebADataset(Dataset):
    """
    CelebA dataset
    """

    def __init__(self, transform=None):
        list_attr_celeba_txt = os.path.join(cfg['root'], 'CelebA', 'Anno', 'list_attr_celeba.txt')
        df = pd.read_csv(list_attr_celeba_txt, delim_whitespace=True, header=None)
        df.columns = ["File", "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs",
                      "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
                      "Chubby",
                      "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male",
                      "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin",
                      "Pointy_Nose",
                      "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
                      "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
                      "Young"]

        self.file_list = df['File']
        self.o_clock_shadow_list = df['5_o_Clock_Shadow']
        self.arched_eyebrows_list = df['Arched_Eyebrows']
        self.attractive_list = df['Attractive']
        self.bags_under_eyes_list = df['Bags_Under_Eyes']
        self.bald_list = df['Bald']
        self.bangs_list = df['Bangs']
        self.big_lips_list = df['Big_Lips']
        self.big_nose_list = df['Big_Nose']
        self.black_hair_list = df['Black_Hair']
        self.blond_hair_list = df['Blond_Hair']
        self.blurry_list = df['Blurry']
        self.brown_hair_list = df['Brown_Hair']
        self.bushy_eyebrows_list = df['Bushy_Eyebrows']
        self.chubby_list = ['Chubby']
        self.double_chin_list = ['Double_Chin']
        self.eyeglasses_list = df['Eyeglasses']
        self.goatee_list = df['Goatee']
        self.gray_hair_list = df['Gray_Hair']
        self.heavy_makeup_list = df['Heavy_Makeup']
        self.high_cheekbones_list = df['High_Cheekbones']
        self.male_list = df['Male']
        self.mouth_slightly_open_list = df['Mouth_Slightly_Open']
        self.mustache_list = df['Mustache']
        self.narrow_eyes_list = df['Narrow_Eyes']
        self.no_beard_list = df['No_Beard']
        self.oval_face_list = df['Oval_Face']
        self.pale_skin_list = df['Pale_Skin']
        self.pointy_nose_list = df['Pointy_Nose']
        self.receding_hairline_list = df['Receding_Hairline']
        self.rosy_cheeks_list = df['Rosy_Cheeks']
        self.sideburns_list = df['Sideburns']
        self.smiling_list = df['Smiling']
        self.straight_hair_list = df['Straight_Hair']
        self.wavy_hair_list = df['Wavy_Hair']
        self.wearing_earrings_list = df['Wearing_Earrings']
        self.wearing_hat_list = df['Wearing_Hat']
        self.wearing_lipstick_list = df['Wearing_Lipstick']
        self.wearing_necklace_list = df['Wearing_Necklace']
        self.wearing_necktie_list = df['Wearing_Necktie']
        self.young_list = df['Young']

        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(cfg['coot', 'CelebA', 'Img', 'img_align_celeba', self.file_list[idx]]))
        sample = {'image': image, '5_o_Clock_Shadow': max(self.o_clock_shadow_list[idx], 0),
                  'Arched_Eyebrows': max(self.arched_eyebrows_list[idx], 0),
                  'Attractive': max(self.attractive_list[idx], 0),
                  'Bags_Under_Eyes': max(self.bags_under_eyes_list[idx], 0),
                  'Bald': max(self.bald_list[idx], 0),
                  'Bangs': max(self.bangs_list[idx], 0), 'Big_Lips': max(self.big_lips_list[idx], 0),
                  'Big_Nose': max(self.big_nose_list[idx], 0), 'Black_Hair': max(self.black_hair_list[idx], 0),
                  'Blond_Hair': max(self.blond_hair_list[idx], 0), 'Blurry': max(self.blurry_list[idx], 0),
                  'Brown_Hair': max(self.brown_hair_list[idx], 0),
                  'Bushy_Eyebrows': max(self.bushy_eyebrows_list[idx], 0),
                  'Chubby': max(self.chubby_list[idx], 0), 'Double_Chin': max(self.double_chin_list[idx], 0),
                  'Eyeglasses': max(self.eyeglasses_list[idx], 0), 'Goatee': max(self.goatee_list[idx], 0),
                  'Gray_Hair': max(self.gray_hair_list[idx], 0), 'Heavy_Makeup': max(self.heavy_makeup_list[idx], 0),
                  'High_Cheekbones': max(self.high_cheekbones_list[idx], 0),
                  'Male': max(self.male_list[idx], 0),
                  'Mouth_Slightly_Open': max(self.mouth_slightly_open_list[idx], 0),
                  'Mustache': max(self.mustache_list[idx], 0),
                  'Narrow_Eyes': max(self.narrow_eyes_list[idx], 0), 'No_Beard': max(self.no_beard_list[idx], 0),
                  'Oval_Face': max(self.oval_face_list[idx], 0),
                  'Pale_Skin': max(self.pale_skin_list[idx], 0), 'Pointy_Nose': max(self.pointy_nose_list[idx], 0),
                  'Receding_Hairline': max(self.receding_hairline_list[idx], 0),
                  'Rosy_Cheeks': max(self.rosy_cheeks_list[idx], 0), 'Sideburns': max(self.sideburns_list[idx], 0),
                  'Smiling': max(self.smiling_list[idx], 0), 'Straight_Hair': max(self.straight_hair_list[idx], 0),
                  'Wavy_Hair': max(self.wavy_hair_list[idx], 0),
                  'Wearing_Earrings': max(self.wearing_earrings_list[idx], 0),
                  'Wearing_Hat': max(self.wearing_hat_list[idx], 0),
                  'Wearing_Lipstick': max(self.wearing_lipstick_list[idx], 0),
                  'Wearing_Necklace': max(self.wearing_necklace_list[idx], 0),
                  'Wearing_Necktie': max(self.wearing_necktie_list[idx], 0), 'Young': max(self.young_list[idx], 0)}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class UTKFaceDataset(Dataset):
    """
    UTKFace dataset
    """

    def __init__(self, train=True, transform=None):
        files = os.listdir(os.path.join(cfg['root'], 'UTKFace'))
        ages = [int(fname.split("_")[0]) for fname in files]

        train_files, test_files, train_ages, test_ages = train_test_split(files, ages, test_size=0.2, random_state=42)

        if train:
            self.filelist = train_files
            self.agelist = train_ages
            self.genderlist = [int(fname.split("_")[1]) for fname in train_files]
            self.racelist = [int(fname.split("_")[2]) if len(fname.split("_")[2]) == 1 else 4 for fname in train_files]
        else:
            self.filelist = test_files
            self.agelist = test_ages
            self.genderlist = [int(fname.split("_")[1]) for fname in test_files]
            self.racelist = [int(fname.split("_")[2]) if len(fname.split("_")[2]) == 1 else 4 for fname in test_files]

        self.transform = transform

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        img_name = os.path.join(cfg['root'], 'UTKFace', self.filelist[idx])

        image = io.imread(img_name)
        sample = {'image': image, 'age': self.agelist[idx], "gender": self.genderlist[idx],
                  "race": self.racelist[idx]}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class FER2013Dataset(Dataset):
    """
    FER2013 dataset
    """

    def __init__(self, train=True, transform=None):
        imgs = []
        labels = []

        type_ = 'train' if train else 'test'

        for cat in os.listdir(os.path.join(cfg['root'], 'FER2013', type_)):
            for img_f in os.listdir(os.path.join(cfg['root'], 'FER2013', type_, cat)):
                imgs.append(os.path.join(cfg['root'], 'FER2013', type_, cat, img_f))
                labels.append(int(cat))

        self.imagefiles = imgs
        self.labels = labels

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'filename': self.imagefiles[idx], 'image': io.imread(self.imagefiles[idx]),
                  'emotion': self.labels[idx], "gender": 0, "race": 0, "age": 0}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(gray2rgb(sample['image']).astype(np.uint8)))

        return sample

# class FER2013Dataset(Dataset):
#     """
#     FER2013 dataset
#     """
#
#     def __init__(self, train=True, fer2013_csv=os.path.join(cfg['root'], 'FER2013', 'fer2013.csv'), transform=None):
#         df = pd.read_csv(fer2013_csv)
#         train_imgs = []
#         test_imgs = []
#         train_labels = []
#         test_labels = []
#
#         for i in range(len(df['Usage'])):
#             if df['Usage'][i] == 'Training':
#                 img_array = np.zeros((48, 48, 3))
#                 img_array[:, :, 0] = np.array(df['pixels'][i].split(" ")).reshape(48, 48).astype(np.float)
#                 img_array[:, :, 1] = np.array(df['pixels'][i].split(" ")).reshape(48, 48).astype(np.float)
#                 img_array[:, :, 2] = np.array(df['pixels'][i].split(" ")).reshape(48, 48).astype(np.float)
#                 test_imgs.append(img_array)
#                 train_imgs.append(img_array)
#                 train_labels.append(df['emotion'][i])
#             elif df['Usage'][i] == 'PrivateTest':
#                 img_array = np.zeros((48, 48, 3))
#                 img_array[:, :, 0] = np.array(df['pixels'][i].split(" ")).reshape(48, 48).astype(np.float)
#                 img_array[:, :, 1] = np.array(df['pixels'][i].split(" ")).reshape(48, 48).astype(np.float)
#                 img_array[:, :, 2] = np.array(df['pixels'][i].split(" ")).reshape(48, 48).astype(np.float)
#                 test_imgs.append(img_array)
#                 test_labels.append(df['emotion'][i])
#
#         if train:
#             self.images = train_imgs
#             self.labels = train_labels
#         else:
#             self.images = test_imgs
#             self.labels = test_labels
#
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         sample = {'image': self.images[idx], 'emotion': self.labels[idx], "gender": 0, "race": 0, "age": 0}
#
#         if self.transform:
#             sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))
#
#         return sample
