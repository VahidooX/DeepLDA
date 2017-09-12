import pickle
import gzip
import numpy as np
from keras.datasets import mnist
from svm_classification import svm_classify
from models import create_model
from keras.optimizers import Adam
from objectives import lda_loss


if __name__ == '__main__':
    ############
    # Parameters Section

    # the path to save the final learned features
    save_to = './new_features.gz'

    # the size of the new space learned by the model (number of the new features)
    outdim_size = 10

    # the parameters for training the network
    epoch_num = 100
    batch_size = 800

    # the regularization parameter of the network
    # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
    reg_par = 1e-5

    # The margin and n_components (number of components) parameter used in the loss function
    # n_components should be at most class_size-1
    margin = 1.0
    n_components = 9

    # Parameter C of SVM
    C = 1e-1
    # end of parameters section
    ############

    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.reshape(x_train, (len(x_train), -1))
    x_test = np.reshape(x_test, (len(x_test), -1))

    # Building, training, and producing the new features by Deep LDA
    model = create_model(x_train.shape[-1], reg_par, outdim_size)

    model_optimizer = Adam()
    model.compile(loss=lda_loss(n_components, margin), optimizer=model_optimizer)

    model.summary()

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_num, shuffle=True, validation_data=(x_test, y_test), verbose=2)

    x_train_new = model.predict(x_train)
    x_test_new = model.predict(x_test)

    # Training and testing of SVM with linear kernel on the new features
    [train_acc, test_acc] = svm_classify(x_train_new, y_train, x_test_new, y_test, C=C)
    print("Accuracy on train data is:", train_acc * 100.0)
    print("Accuracy on test data is:", test_acc*100.0)

    # Saving new features in a gzip pickled file specified by save_to
    print('Saving new features ...')
    f = gzip.open(save_to, 'wb')
    pickle.dump([(x_train_new, y_train), (x_test_new, y_test)], f)
    f.close()
