from sklearn.metrics import confusion_matrix
import itertools
import matplotlib
import numpy as np
import tfplot
import re
from textwrap import wrap

def Compare2(pre, true): #bad name
	'''
	:param pre: single sparse predictions
	:param true: single sparse labels
	:return:
	'''
	tp = 0
	tn = 0
	fp = 0
	fn = 0

	if true == 0:
		if pre == true:
			tn = tn + 1
		else:
			fp = fp + 1
	else:
		if pre == true:
			tp = tp + 1
		else:
			fn = fn + 1

	return tp, fp, tn, fn

def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix',
                          tensor_name='MyFigure/image', normalize=False):
    '''
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    if normalize:
        cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predictions ', fontsize=25)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=20, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('Truths', fontsize=25)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=20, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=25,
                verticalalignment='center', color="black")
    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)  # Convert a matplotlib figure fig into a TensorFlow Summary object that can be directly fed into Summary.FileWriter
    return summary
