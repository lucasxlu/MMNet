import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def viz_confusion_matrix_from_gts_and_preds(gts, preds):
    """
    show Confusion Matrix with matplotlib
    """
    # lab = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
    lab = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]
    groundtruth = lab
    predicted = lab

    cm = np.array(confusion_matrix(gts, preds))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    ax.imshow(cm)

    ax.set_xticks(np.arange(len(predicted)))
    ax.set_yticks(np.arange(len(groundtruth)))

    ax.set_xticklabels(predicted)
    ax.set_yticklabels(groundtruth)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(groundtruth)):
        for j in range(len(predicted)):
            text = ax.text(j, i, round(cm[i, j], 2),
                           ha="center", va="center", color="w")

    # ax.set_title("Confusion Matrix of MMNet on RAF-DB with basic expressions")
    fig.tight_layout()
    plt.show()


def viz_confusion_matrix_from_cm_matrix(cm):
    """
    show Confusion Matrix with matplotlib
    """
    # lab = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
    lab = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]
    groundtruth = lab
    predicted = lab

    cm = np.array(cm)
    cm = cm.astype('float') * 100 / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    ax.imshow(cm)

    ax.set_xticks(np.arange(len(predicted)))
    ax.set_yticks(np.arange(len(groundtruth)))

    ax.set_xticklabels(predicted)
    ax.set_yticklabels(groundtruth)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(groundtruth)):
        for j in range(len(predicted)):
            text = ax.text(j, i, round(cm[i, j], 2),
                           ha="center", va="center", color="w")

    # ax.set_title("Confusion Matrix of MMNet on RAF-DB with basic expressions")
    fig.tight_layout()
    plt.show()


def viz_feature_map():
    """
    show deep feature visualization
    :return:
    """
    data = np.random.rand(224, 224)

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='hot', interpolation='nearest')

    # We want to show all ticks...
    # ax.set_xticks(np.arange(len(predicted)))
    # ax.set_yticks(np.arange(len(groundtruth)))
    # ... and label them with the respective list entries
    # ax.set_xticklabels(predicted)
    # ax.set_yticklabels(groundtruth)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # for i in range(len(groundtruth)):
    #     for j in range(len(predicted)):
    #         text = ax.text(j, i, data[i, j],
    #                        ha="center", va="center", color="w")

    # ax.set_title("Heatmap of Deep Feature Activiation")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('RAF_DB_MMNet.csv')
    cm = confusion_matrix(df['gt'], df['pred'])
    viz_confusion_matrix_from_cm_matrix(cm)
