import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.metrics import accuracy_score

def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label, linewidth=2)

def plot_subfigure(X, Y, subplot, transform):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    classif = OneVsRestClassifier(LogisticRegression())
    classif.fit(X, Y)
    y_pred = classif.predict(X)
    
    print('{} + OneVsRestClassifier + LogisticRegression accuracy_score {}'.format(transform, accuracy_score(Y, y_pred)))

    plt.subplot(1, 2, subplot)    
    plt.scatter(X[:, 0], X[:, 1], s=15, c='gray', edgecolors=(0, 0, 0))
    
    for i in np.unique(Y.argmax(axis=1)):
        class_ = np.where(Y[:, i])
        plt.scatter(X[class_, 0], X[class_, 1], s=25,  linewidths=2, label='Class {}'.format(str(i)))
    
    for i in range(len(classif.estimators_)):
        plot_hyperplane(classif.estimators_[i], min_x, max_x, 'k--', 'Boundary\nfor class {}'.format(str(i)))

    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .1 * max_x, max_x + .1 * max_x)
    plt.ylim(min_y - .1 * max_y, max_y + .1 * max_y)
    
def plot_transform_hyperplanes(X, y, save=None):

    plt.figure(figsize=(27,9), dpi=600)

    plot_subfigure(X, y, 1, 'cca')
    plot_subfigure(X, y, 2, 'pca')

    
    plt.subplots_adjust(.04, .02, .97, .94, .09, .2)

    if save is not None:
        plt.savefig(save)    
    
    plt.show()

        

from matplotlib import offsetbox
#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_dataset(X, y, images=None, labels=None, gray=False, save=None, y_original=None):
    uni_y = len(np.unique(y))
    
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    
    plt.figure(figsize=(27,18), dpi=600)
    ax = plt.subplot(111)
    
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / uni_y),
                 fontdict={'weight': 'bold', 'size': 9})
            

    if images is not None:
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    # don't show points that are too close
                    continue
                
                if labels is not None:
                    if y_original is not None:
                        plt.text(X[i, 0]-0.01, X[i, 1]-0.033, labels[y_original[i]], fontdict={'weight': 'bold', 'size': 15})
                    else:
                        plt.text(X[i, 0]-0.01, X[i, 1]-0.033, labels[y[i]], fontdict={'weight': 'bold', 'size': 15})
                
                shown_images = np.r_[shown_images, [X[i]]]
                if gray:
                    image_ =  offsetbox.OffsetImage(images[i], cmap=plt.cm.gray)
                else:
                    image_ =  offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r)
                
                imagebox = offsetbox.AnnotationBbox(image_ , X[i])
                
                ax.add_artist(imagebox)
            
    plt.xticks([]), plt.yticks([])
    
    if save is not None:
        plt.savefig(save)
        
    plt.show()