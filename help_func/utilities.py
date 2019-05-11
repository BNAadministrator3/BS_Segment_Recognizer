from keras import backend as K
from keras.layers import *
from keras.activations import relu

def focal_loss(gamma=2, alpha=0.25):

    def focal_loss_fixed(y_true, y_pred):#with tensorflow
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = 10.0 * K.sum(loss, axis=1)
        return loss

    return focal_loss_fixed

def ReguBlock(n_output):
    # n_output: number of feature maps in the block
    # keras functional api: return the function of type
    # Tensor -> Tensor
    def f(x):

        # H_l(x):
        # first pre-activation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # first convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=regularizers.l2(0.0005))(h)

        # second pre-activation
        h = BatchNormalization()(h)
        h = Activation(relu)(h)
        # second convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=regularizers.l2(0.0005))(h)
        return h

    return f

#The codes reference to the https://www.kaggle.com/meownoid/tiny-resnet-with-keras-99-314
def ResiBlock(n_output, upscale=True):
    # n_output: number of feature maps in the block
    # upscale: should we use the 1x1 conv2d mapping for shortcut or not

    # keras functional api: return the function of type
    # Tensor -> Tensor
    def f(x):

        # H_l(x):
        # first pre-activation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # h = LeakyReLU(0.1)(h)
        # first convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=regularizers.l2(0.0005))(h)

        # second pre-activation
        h = BatchNormalization()(h)
        h = Activation(relu)(h)
        # h = LeakyReLU(0.1)(h)
        # second convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=regularizers.l2(0.0005))(h)

        # f(x):
        if upscale:
            # 1x1 conv2d
            f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)
        else:
            # identity
            f = x

        # F_l(x) = f(x) + H_l(x):
        return add([f, h])


    return f

def XcepBlock(n_output, upscale=True):
    # n_output: number of feature maps in the block
    # upscale: should we use the 1x1 conv2d mapping for shortcut or not

    # keras functional api: return the function of type
    # Tensor -> Tensor
    def f(x):

        # H_l(x):
        # first pre-activation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # h = LeakyReLU(0.1)(h)
        # first convolution
        h = SeparableConv2D(kernel_size=3, filters=n_output, strides=1, kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=regularizers.l2(0.0005))(h)

        # second pre-activation
        h = BatchNormalization()(h)
        h = Activation(relu)(h)
        # h = LeakyReLU(0.1)(h)
        # second convolution
        h = SeparableConv2D(kernel_size=3, filters=n_output, strides=1, kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=regularizers.l2(0.0005))(h)

        # f(x):
        if upscale:
            # 1x1 conv2d
            f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)
        else:
            # identity
            f = x

        # F_l(x) = f(x) + H_l(x):
        return add([f, h])

    return f

def focal_loss(gamma=2, alpha=0.25):

    def focal_loss_fixed(y_true, y_pred):#with tensorflow
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = 10.0 * K.sum(loss, axis=1)
        return loss

    return focal_loss_fixed