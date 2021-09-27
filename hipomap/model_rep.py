from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Input, Flatten, AveragePooling2D
from tensorflow.keras.models import Model


def model_rep(top, acti_size):
    """
    Build a model for classifying the generated hipomap into Cancer/Normal.

    Parameters
    ----------
    top: int
        The number of top patches which wanted to deal with K.
    acti_size: int
        The width of hipomap ndarray.
        The activation map size of the last conv layer from model which generated hipomap (activation map's width * height).
    Returns
    -------

    """
    input_ = Input(shape=(top, acti_size, 1))
    out = Conv2D(128, kernel_size=(2, 2), activation='relu')(input_)
    out = AveragePooling2D(pool_size=(2, 2))(out)
    out = Dropout(0.3)(out)
    out = Conv2D(64, kernel_size=(2, 2), activation='relu')(out)
    out = AveragePooling2D(pool_size=(2, 2))(out)
    out = Dropout(0.3)(out)
    out = Conv2D(32, kernel_size=(2, 2), activation='relu')(out)
    out = AveragePooling2D(pool_size=(2, 2))(out)
    out = Dropout(0.3)(out)
    out = Conv2D(16, kernel_size=(2, 2), activation='relu')(out)
    out = AveragePooling2D(pool_size=(2, 2))(out)
    out = Dropout(0.3)(out)
    out = Flatten()(out)
    out = Dense(units=1024, activation='relu')(out)
    out = Dense(units=1, activation='sigmoid')(out)

    return Model(input_, out)
