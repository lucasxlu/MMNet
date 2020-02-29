"""
Performance Comparision with Commercial APIs like Face++, Google, MS and Amazon
"""
import sys
import os
import requests

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

sys.path.append('../')
from config.cfg import cfg


def prepare_test_imgs(type='basic'):
    face_with_emt = {}

    manual_annotation_dir = 'E:\DataSet\CV\TreeCNN\RAF-Face/basic\Annotation\manual'
    emotion_label_txt_path = os.path.join(cfg['root'], 'RAF-Face', "%s/EmoLabel/list_patition_label.txt" % type)
    emotion_dict = dict(np.loadtxt(emotion_label_txt_path, dtype=np.str))

    for _ in os.listdir(manual_annotation_dir):
        if _.startswith('test_'):
            face_fname = _.replace('_manu_attri', '_aligned').replace('.txt', '.jpg')
            face_with_emt[os.path.join(cfg['root'], 'RAF-Face', '%s/Image/aligned' % type, face_fname)] = int(
                emotion_dict[face_fname.replace('_aligned', '')].strip()) - 1

    return face_with_emt


def facepp(img_path):
    """
    Recognition with Face++ Emotion Recognition API
    :param img_path:
    :return:
    """
    apikey = ''
    apisecret = ''
    url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'

    files = {'image_file': open(img_path, 'rb')}

    payload = {'api_key': apikey, 'api_secret': apisecret,
               # 'return_landmark': 0, 'face_tokens': 'none',
               'return_attributes': 'emotion'}

    response = requests.post(url, data=payload, files=files)
    if response.status_code == 200:
        res_json = response.json()

        max_k = ''
        max_v = 0
        for k, v in res_json['faces'][0]['attributes']['emotion'].items():
            if v > max_v:
                max_v = v
                max_k = k

        return max_k
    else:
        print(response)

        return None


if __name__ == '__main__':
    img_files = prepare_test_imgs()
    print(img_files)

    basic_emt_map = {
        'surprise': 0,
        'fear': 1,
        'disgust': 2,
        'happiness': 3,
        'sadness': 4,
        'anger': 5,
        'neutral': 6
    }

    gt = []
    pred = []

    for imgf, e in img_files.items():
        try:
            emt = facepp(imgf)
            print(emt)
            gt.append(e)
            pred.append(basic_emt_map[emt])
        except:
            pass

    print('Accuracy of Emotion Recognition: %s' % str(accuracy_score(gt, pred)))

    print('Confusion Matrix on FER: ')
    print(confusion_matrix(np.array(gt).ravel().tolist(), np.array(pred).ravel().tolist()))
