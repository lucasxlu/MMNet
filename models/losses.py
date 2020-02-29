from __future__ import print_function

import torch.nn as nn


class MTLoss(nn.Module):
    """
    Multi-task Loss definition
    """

    def __init__(self, emotion_branch_w=3, age_branch_w=1, race_branch_w=1, gender_branch_w=1):
        super(MTLoss, self).__init__()

        self.emotion_branch_w = emotion_branch_w
        self.age_branch_w = age_branch_w
        self.race_branch_w = race_branch_w
        self.gender_branch_w = gender_branch_w
        # self.ldmk_branch_w = ldmk_branch_w

        self.emotion_criterion = nn.CrossEntropyLoss()
        self.age_criterion = nn.CrossEntropyLoss()
        self.race_criterion = nn.CrossEntropyLoss()
        self.gender_criterion = nn.CrossEntropyLoss()
        # self.ldmk_criterion = nn.MSELoss()

    def forward(self, gender_pred, gender_gt, race_pred, race_gt, age_pred, age_gt, emotion_pred, emotion_gt):
        gender_loss = self.gender_criterion(gender_pred, gender_gt)
        race_loss = self.race_criterion(race_pred, race_gt)
        emotion_loss = self.emotion_criterion(emotion_pred, emotion_gt)
        age_loss = self.age_criterion(age_pred, age_gt)
        # ldmk_loss = self.ldmk_criterion(ldmk_pred, ldmk_gt)

        mt_loss = self.emotion_branch_w * emotion_loss + self.age_branch_w * age_loss + self.race_branch_w * race_loss + self.gender_branch_w * gender_loss

        return mt_loss


class RegionLoss(nn.Module):
    """
    RegionLoss definition
    """

    def __init__(self, region_block_w=1, mouth_branch_w=0.2, left_eye_branch_w=0.1, right_eye_branch_w=0.1,
                 nose_branch_w=0.1):
        super(RegionLoss, self).__init__()

        self.region_block_w = region_block_w

        self.left_eye_branch_w = left_eye_branch_w
        self.right_eye_branch_w = right_eye_branch_w
        self.mouth_branch_w = mouth_branch_w
        self.nose_branch_w = nose_branch_w

        self.region_criterion = nn.CrossEntropyLoss()

    def forward(self, emt_left_eye, emt_right_eye, emt_nose, emt_mouth, emt_gt):
        left_eye_loss = self.region_criterion(emt_left_eye, emt_gt)
        right_eye_loss = self.region_criterion(emt_right_eye, emt_gt)
        nose_loss = self.region_criterion(emt_nose, emt_gt)
        mouth_loss = self.region_criterion(emt_mouth, emt_gt)

        tree_loss = self.region_block_w * (
                self.left_eye_branch_w * left_eye_loss + self.right_eye_branch_w * right_eye_loss +
                self.nose_branch_w * nose_loss + self.mouth_branch_w * mouth_loss)

        return tree_loss


class MTLossUTKFace(MTLoss):
    """
    Multi-task Loss on UTKFace
    """

    def __init__(self, age_branch_w=0.2, race_branch_w=0.1, gender_branch_w=0.1):
        super(MTLossUTKFace, self).__init__()

        self.age_branch_w = age_branch_w
        self.race_branch_w = race_branch_w
        self.gender_branch_w = gender_branch_w

        self.age_criterion = nn.CrossEntropyLoss()
        self.race_criterion = nn.CrossEntropyLoss()
        self.gender_criterion = nn.CrossEntropyLoss()

    def forward(self, gender_pred, gender_gt, race_pred, race_gt, age_pred, age_gt):
        gender_loss = self.gender_criterion(gender_pred, gender_gt)
        race_loss = self.race_criterion(race_pred, race_gt)
        age_loss = self.age_criterion(age_pred, age_gt)

        mt_loss = self.age_branch_w * age_loss + self.race_branch_w * race_loss + self.gender_branch_w * gender_loss

        return mt_loss
