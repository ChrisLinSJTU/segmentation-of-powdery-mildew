from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import backend as K
import os

target_size = (512, 512)
batch_size = 2

path_train = 'data/train'
path_val = 'data/val'
path_test = 'data/test'

# generate train data
def gen_train_data():
    data_gen_args = dict(
                         rotation_range=180.0,   
                         width_shift_range=0.1,  
                         height_shift_range=0.1, 
                         zoom_range=[0.6, 1.4], 
                         fill_mode='constant',
                         cval=0.,
                         horizontal_flip=True,
                         vertical_flip=True,
                         data_format=K.image_data_format())
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    image_generator = image_datagen.flow_from_directory(
                        directory = path_train,
                        classes = ['img'],
                        class_mode = None,
                        color_mode = 'rgb',
                        target_size = target_size,
                        batch_size = batch_size,
                        save_prefix  = '2',
                        seed = 1)
    
    mask_generator = mask_datagen.flow_from_directory(
                        directory = path_train,
                        classes = ['imgAno'],
                        class_mode = None,
                        color_mode = 'grayscale',
                        target_size = target_size,
                        batch_size = batch_size,
                        save_prefix  = '2',
                        seed = 1)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img = img.astype(np.float32)/255
        mask = mask.astype(np.float32)/255
        mask[mask>0] = 1 
        yield (img,mask)

# generate val data, used to find best hyperparameters
def gen_val_data():
    data_gen_args = dict(
                         rotation_range=180.0,   
                         width_shift_range=0.1,  
                         height_shift_range=0.1, 
                         zoom_range=[0.6, 1.4], 
                         fill_mode='constant',
                         cval=0.,
                         horizontal_flip=True,
                         vertical_flip=True,
                         data_format=K.image_data_format())
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    image_generator = image_datagen.flow_from_directory(
                        directory = path_val,
                        classes = ['img'],
                        class_mode = None,
                        color_mode = 'rgb',
                        target_size = target_size,
                        batch_size = batch_size,
                        save_prefix  = '2',
                        seed = 1)
    
    mask_generator = mask_datagen.flow_from_directory(
                        directory = path_val,
                        classes = ['imgAno'],
                        class_mode = None,
                        color_mode = 'grayscale',
                        target_size = target_size,
                        batch_size = batch_size,
                        save_prefix  = '2',
                        seed = 1)
    generator = zip(image_generator, mask_generator)
    for (img,mask) in generator:
        img = img.astype(np.float32)/255
        mask = mask.astype(np.float32)/255
        mask[mask>0] = 1 
        yield (img,mask)

# test data
def test_data():
    import numpy as np
    import cv2
    testimg = np.zeros([20,512,512,3], dtype='uint8')
    testimgAno = np.zeros([20,512,512,1], dtype='uint8')
    for i in range(20):
        img = cv2.imread(os.path.join(path_test, 'img', np.str(i) + '.png'))
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        imgAno = cv2.imread(os.path.join(path_test, 'imgAno', np.str(i) + '.png'), 0)
        imgAno[imgAno>0] = 255
        testimg[i,:,:,:] = img
        testimgAno[i,:,:,0] = imgAno
    testimg = testimg.astype(np.float32)/255.0
    testimgAno = testimgAno.astype(np.float32)/255.0
    return testimg,testimgAno
