from data_generator import gen_val_data
from keras.layers.convolutional import Conv2D
from keras.layers import Input, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Concatenate, Activation
from keras.optimizers import Adam
from keras import initializers
from keras.models import Model
from keras import backend as K
from dataGenerator import gen_train_data, test_data, gen_val_data

# the pre-trained model on other datasets, if you want fine-tune
path_weight = './model_weight.h5'

def iu_acc(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[1, 2, 3], keepdims=False)
    sum_ = K.sum(y_true + y_pred_pos, axis=[1, 2, 3], keepdims=False)
    jac = (intersection) / (sum_ - intersection + smooth)
    return K.mean(jac)

def dice_acc(y_true, y_pred):
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[1, 2, 3], keepdims=False)
    sum_ = K.sum(y_true + y_pred_pos, axis=[1, 2, 3], keepdims=False)
    jac = (2*intersection + smooth) / (sum_ + smooth)
    return K.mean(jac)

# loss function
def my_loss(y_true,y_pred):
    bc = K.binary_crossentropy(y_pred, y_true)
    false_bc = (1.0 - y_true) * bc
    true_bc = y_true * bc
    mloss = false_bc + 10.0 * true_bc
    return mloss

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
# dice_coef may have better performance
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def ConvActive(np_ker, size, inputs, 
                        initial = initializers.glorot_normal(seed=None)):
    
    return Conv2D(np_ker, size, activation='relu',padding = 'same', kernel_initializer = initial)(inputs)

def ConvBatchnormActive(np_ker, size, inputs, 
                        initial = initializers.glorot_normal(seed=None)):
    
    return Activation('relu')(BatchNormalization()(Conv2D(np_ker, size,padding = 'same', kernel_initializer = initial)(inputs)))

def UpConvActiveContact(np_ker, size, inputs, contact,
                  initial = initializers.glorot_normal(seed=None)):
    up = UpSampling2D(size = (2,2))(inputs)
    upconv = Conv2D(np_ker, size, activation = 'relu', padding = 'same', kernel_initializer = initial)(up) 
    return Concatenate(axis=3)([upconv, contact])

def get_unet(pretrained_weights = None):
    
    inputs = Input((512, 512, 3))

    conv1 = ConvBatchnormActive(16, 3, inputs)
    conv1 = ConvBatchnormActive(16, 3, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = ConvBatchnormActive(32, 3, pool1)
    conv2 = ConvBatchnormActive(32, 3, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = ConvBatchnormActive(64, 3, pool2)
    conv3 = ConvBatchnormActive(64, 3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = ConvBatchnormActive(128, 3, pool3)
    conv4 = ConvBatchnormActive(128, 3, conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = ConvBatchnormActive(256, 3, pool4)
    conv5 = ConvBatchnormActive(256, 3, conv5)
    
    up6 = UpConvActiveContact(128, 2, conv5, conv4)
    conv6 = ConvBatchnormActive(128, 3, up6)
    conv6 = ConvBatchnormActive(128, 3, conv6)
    
    up7 = UpConvActiveContact(64, 2, conv6, conv3)
    conv7 = ConvBatchnormActive(64, 3, up7)
    conv7 = ConvBatchnormActive(64, 3, conv7)
    
    up8 = UpConvActiveContact(32, 2, conv7, conv2)
    conv8 = ConvBatchnormActive(32, 3, up8)
    conv8 = ConvBatchnormActive(32, 3, conv8)
    
    up9 = UpConvActiveContact(16, 2, conv8, conv1)
    conv9 = ConvBatchnormActive(16, 3, up9)
    conv9 = ConvBatchnormActive(16, 3, conv9)
    
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    
    model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, metrics = [iu_acc, dice_acc, 'accuracy'])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#%% training 
model = get_unet()
train_data = gen_train_data()
val_data = gen_val_data()
model.fit_generator(train_data, 
                    steps_per_epoch=10000, 
                    epochs=16, 
                    verbose=1,
                    validation_data = val_data, 
                    shuffle=True)

#%% fine tuning with pre-trained weight (optional)
# model = get_unet(path_weight)
# train_data = gen_train_data()
# val_data = gen_val_data()
# model.fit_generator(train_data, steps_per_epoch=10000, epochs=4, verbose=1, validation_data=val_data,
#                     shuffle=True)

#%% predict your model
# model = get_unet(path_weight)
# test_data = test_data()
# y_pred = model.predict(test_data, verbose=1)
