from keras.layers import Dense, Merge
from keras.models import Sequential
from keras.regularizers import l2


def create_model(input_dim, reg_par, outdim_size):
    """
    Builds the model
    The structure of the model can get easily substituted with a more efficient and powerful network like CNN
    """
    model = Sequential()

    model.add(Dense(1024, input_shape=(input_dim,), activation='sigmoid', kernel_regularizer=l2(reg_par)))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=l2(reg_par)))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=l2(reg_par)))
    model.add(Dense(outdim_size, activation='linear', kernel_regularizer=l2(reg_par)))

    return model