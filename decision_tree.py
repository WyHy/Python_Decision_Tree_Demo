# coding: utf-8

from sklearn import tree
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import os
import random

model_name = './data/dts.model'


def train(X, Y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    # 保存模型
    joblib.dump(clf, model_name)

    return clf


if __name__ == '__main__':
    with open('./data/X.txt') as f:
        lines = f.readlines()
        items = [line.replace('\n', '').split('\t') for line in lines]
        X = [[int(ele) for ele in item] for item in items]

    with open('./data/Y.txt') as f:
        lines = f.readlines()
        Y = [line.replace('\n', '').split('\t') for line in lines]

    lst = list(zip(Y, X))
    random.shuffle(lst)

    Y, X = [], []
    for item in lst:
        Y.append(item[0][0])
        X.append(item[1][0])

    X_train = X[:-100]
    Y_train = Y[:-100]
    X_test = X[-100:]
    Y_test = Y[-100:]

    if os.path.exists(model_name):
        # 加载模型
        clf = joblib.load(model_name)
    else:
        clf = train(X, Y)

    Y_predict = clf.predict(X_test)

    total = len(Y_test)
    count = 0
    for i in range(total):
        print(Y_predict[i], Y_test[i])
        if Y_predict[i] == Y_test[i]:
            count += 1

    print("%s / %s" % (count, total))
    print(accuracy_score(Y_test, Y_predict))



