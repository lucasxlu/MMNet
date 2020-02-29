import os
import sys

from mtcnn.mtcnn import MTCNN
import numpy as np
import cv2
import pandas as pd

sys.path.append('../')
from util.file_utils import mkdirs_if_not_exist


def det_landmarks(image_path, dlib_model="E:/ModelZoo/shape_predictor_68_face_landmarks.dat"):
    """
    detect faces in one image, return face bbox and landmarks
    :param image_path:
    :return:
    """
    import dlib
    predictor = dlib.shape_predictor(dlib_model)
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(image_path)
    faces = detector(img, 1)

    result = {}
    if len(faces) > 0:
        for k, d in enumerate(faces):
            shape = predictor(img, d)
            result[k] = {"bbox": [d.left(), d.top(), d.right(), d.bottom()],
                         "landmarks": [[shape.part(i).x, shape.part(i).y] for i in range(68)]}

    return result


def crop_local_part(face_path):
    """
    crop local part from a given image, using dlib(MTCNN) face detector
    :param face_path:
    :param part: "LeftEye", "RightEye", "Nose", "Mouth"
    :return:
    """

    img = cv2.imread(face_path)
    # detector = MTCNN()
    # result = detector.detect_faces(img)
    # print(result)

    result = det_landmarks(face_path)
    print(result)

    for index, face in result.items():
        # draw facial landmarks
        # for i, ldmk in enumerate(face['landmarks']):
        #     cv2.circle(img, (ldmk[0], ldmk[1]), 2, (255, 245, 0), -1)
        #     cv2.putText(img, str(i), (ldmk[0], ldmk[1] - 2),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (106, 106, 255), 0, cv2.LINE_AA)

        left_eye = (min([face['landmarks'][i][0] for i in range(17, 22)]), face['landmarks'][19][1],
                    face['landmarks'][39][0], face['landmarks'][28][1])

        right_eye = (min([face['landmarks'][i][0] for i in range(22, 27)]), face['landmarks'][23][1],
                     face['landmarks'][16][0], face['landmarks'][28][1],)

        nose = (face['landmarks'][39][0], face['landmarks'][28][1], face['landmarks'][42][0], face['landmarks'][33][1])

        mouth = (face['landmarks'][48][0], face['landmarks'][33][1],
                 face['landmarks'][54][0], int((face['landmarks'][58][1] + face['landmarks'][8][1]) / 2))

        # show bbox with dlib
        # cv2.rectangle(img, (face['bbox'][0], face['bbox'][1]), (face['bbox'][2], face['bbox'][3]), (0, 255, 225), 2)
        # cv2.rectangle(img, (left_eye[0], left_eye[1]), (left_eye[2], left_eye[3]), (255, 0, 225), 1)
        # cv2.rectangle(img, (right_eye[0], right_eye[1]), (right_eye[2], right_eye[3]), (155, 155, 225), 1)
        # cv2.rectangle(img, (nose[0], nose[1]), (nose[2], nose[3]), (225, 155, 125), 1)
        # cv2.rectangle(img, (mouth[0], mouth[1]), (mouth[2], mouth[3]), (115, 155, 125), 1)

    # show bbox with MTCNN
    # cv2.rectangle(img, (result[0]['box'][0], result[0]['box'][1]),
    #               (result[0]['box'][0] + result[0]['box'][2], result[0]['box'][1] + result[0]['box'][3]),
    #               (255, 0, 0), 2)
    #
    # for _ in result:
    #     cv2.rectangle(img, (_['box'][0], _['box'][1]), (_['box'][0] + _['box'][2], _['box'][1] + _['box'][3]),
    #                   (0, 0, 255), 2)
    #     cv2.circle(img, (_['keypoints']['left_eye'][0], _['keypoints']['left_eye'][1]), 2, (255, 245, 0), -1)
    #     cv2.circle(img, (_['keypoints']['right_eye'][0], _['keypoints']['right_eye'][1]), 2, (255, 245, 0), -1)
    #     cv2.circle(img, (_['keypoints']['nose'][0], _['keypoints']['nose'][1]), 2, (255, 245, 0), -1)
    #     cv2.circle(img, (_['keypoints']['mouth_left'][0], _['keypoints']['mouth_left'][1]), 2, (255, 245, 0), -1)
    #     cv2.circle(img, (_['keypoints']['mouth_right'][0], _['keypoints']['mouth_right'][1]), 2, (255, 245, 0), -1)

    # cv2.imshow('img', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    local_parts = {}

    left_eye_crop_region = img[left_eye[1]: left_eye[3], left_eye[0]: left_eye[2]]
    left_eye_file_dir = 'E:/DataSet/CV/TreeCNN/RAF-Face/basic/LocalParts/{0}/'.format("LeftEye")

    right_eye_crop_region = img[right_eye[1]: right_eye[3], right_eye[0]: right_eye[2]]
    right_eye_file_dir = 'E:/DataSet/CV/TreeCNN/RAF-Face/basic/LocalParts/{0}/'.format("RightEye")

    nose_crop_region = img[nose[1]: nose[3], nose[0]: nose[2]]
    nose_file_dir = 'E:/DataSet/CV/TreeCNN/RAF-Face/basic/LocalParts/{0}/'.format("Nose")

    mouth_crop_region = img[mouth[1]: mouth[3], mouth[0]: mouth[2]]
    mouth_file_dir = 'E:/DataSet/CV/TreeCNN/RAF-Face/basic/LocalParts/{0}/'.format("Mouth")

    local_parts[left_eye_file_dir] = left_eye_crop_region
    local_parts[right_eye_file_dir] = right_eye_crop_region
    local_parts[nose_file_dir] = nose_crop_region
    local_parts[mouth_file_dir] = mouth_crop_region

    for k, v in local_parts.items():
        mkdirs_if_not_exist(k)
        cv2.imwrite(k + '{0}'.format(face_path.split(os.path.sep)[-1]), v)


if __name__ == '__main__':
    # crop_local_part('./ly.jpg')

    rt = 'E:/DataSet/CV/TreeCNN/RAF-Face/basic/Image/aligned'
    fail_list = []
    for imgf in os.listdir(rt):
        try:
            crop_local_part(os.path.join(rt, imgf))
        except:
            fail_list.append(imgf)

        print('Processing image %s successfully...' % imgf)

    df = pd.DataFrame(fail_list)
    df.to_csv('./fail.csv', index=False, index_label=False, header=False)
