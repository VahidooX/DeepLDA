from sklearn import svm
from sklearn.metrics import accuracy_score


def svm_classify(x_train, y_train, x_test, y_test, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(x_train, y_train.ravel())

    p = clf.predict(x_train)
    train_acc = accuracy_score(y_train, p)

    p = clf.predict(x_test)
    test_acc = accuracy_score(y_test, p)

    return [train_acc, test_acc]


