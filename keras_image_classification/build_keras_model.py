

###################################################################################
#
#   Keras Image Classification
#
###################################################################################


'''
requirements.txt

tensorflow==1.12.0
keras==2.2.4
numpy==1.15.4
pillow==5.3.0

'''


'''
Example Directory Structure for Images:

./data
    /training/
        /dog/
            image01.jpg
            image02.jpg
            image03.jpg
        /cat/
            image04.jpg
            image05.jpg
            image06.jpg
        
    /testing/
        /dog/
            image07.jpg
            image08.jpg
            image09.jpg
        /cat/
            image10.jpg
            image11.jpg
            image12.jpg
    
    /validation/
        /dog/
            image13.jpg
            image14.jpg
            image15.jpg
        /cat/
            image16.jpg
            image17.jpg
            image18.jpg
    
'''


import os,sys
import math
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
from keras.models import load_model



def calculate_batch_size(total_files, min_files):
    batch_size = min_files if min_files <= 10 else 10
    while (total_files % batch_size != 0):
        batch_size -= 1
    return batch_size



def save_bottleneck_features(args, batch_size_training, batch_size_validation):
    
    print('[ INFO ] Loading Imagenet Model..')
    model = applications.VGG16(include_top=False, weights='imagenet')
    
    datagen = ImageDataGenerator(rescale=1. / 255)
    
    generator = datagen.flow_from_directory(
        args['train_data_dir'],
        target_size=(args['img_width'], args['img_height']),
        batch_size=batch_size_training,
        class_mode=None,
        shuffle=False)
    
    #print(len(generator.filenames))
    #print(generator.class_indices)
    #print(len(generator.class_indices))
    
    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)
    
    predict_size_train = int(math.ceil(nb_train_samples / batch_size_training))
    print('[ INFO ] Predicted Train Size: ' + str(predict_size_train))
    
    print('[ INFO ] Generate training bottleneck features')
    bottleneck_features_train = model.predict_generator(generator, predict_size_train)
    
    print('[ INFO ] Saving Features for Training ( {} )'.format( args['bottleneck_training'] ))
    np.save(args['bottleneck_training'], bottleneck_features_train)
    
    generator = datagen.flow_from_directory(
        args['validation_data_dir'],
        target_size=(args['img_width'], args['img_height']),
        batch_size=batch_size_validation,
        class_mode=None,
        shuffle=False)
    
    nb_validation_samples = len(generator.filenames)
    
    predict_size_validation = int(math.ceil(nb_validation_samples / batch_size_validation))
    
    print('[ INFO ] Generate validation bottleneck features')
    bottleneck_features_validation = model.predict_generator(generator, predict_size_validation)
    
    print('[ INFO ] Saving Features for Validation ( {} )'.format( args['bottleneck_validation'] ))
    np.save(args['bottleneck_validation'], bottleneck_features_validation)



def train_top_model(args, tensorboard):
    
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    
    generator_top = datagen_top.flow_from_directory(
        args['train_data_dir'],
        target_size=(args['img_width'], args['img_height']),
        batch_size=batch_size_training,
        class_mode='categorical',
        shuffle=False)
    
    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)
    
    print('[ INFO ] Saving class_indices ( {} )'.format( args['class_indices_path'] ))
    np.save(args['class_indices_path'], generator_top.class_indices)
    
    print('[ INFO ] Load training bottleneck features ( {} )'.format( args['bottleneck_training'] ))
    train_data = np.load(args['bottleneck_training'])
    
    train_labels = generator_top.classes
    # Convert the training labels to categorical vectors
    # https://github.com/fchollet/keras/issues/3467
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    
    generator_top = datagen_top.flow_from_directory(
        args['validation_data_dir'],
        target_size=(args['img_width'], args['img_height']),
        batch_size=batch_size_validation,
        class_mode=None,
        shuffle=False)
    
    nb_validation_samples = len(generator_top.filenames)
    
    print('[ INFO ] Load validation bottleneck features ( {} )'.format( args['bottleneck_validation'] ))
    validation_data = np.load(args['bottleneck_validation'])
    
    validation_labels = generator_top.classes
    validation_labels = to_categorical(validation_labels, num_classes=num_classes)
    
    print('[ INFO ] Building Neural Network Model (hidden layers)')
    model=Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(num_classes, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print('[ INFO ] Fitting CNN...')
    history = model.fit(train_data, train_labels,
                        epochs=args['epochs'],
                        batch_size=batch_size_training,
                        validation_data=(validation_data, validation_labels),
                        callbacks=[tensorboard])                                    # This line is used for Tensorboard. Remove this for faster runtimes.
    
    print('[ INFO ] Saving CNN model to {}'.format( args['top_model_path'] ))
    model.save( args['top_model_path'] )
    print('[ INFO ] Saving CNN model weights to {}'.format( args['top_model_weights_path'] ))
    model.save_weights( args['top_model_weights_path'] )
    
    (eval_loss, eval_accuracy) = model.evaluate(validation_data, validation_labels, batch_size=batch_size_validation, verbose=1)
    
    print('[INFO] Accuracy: ' + str(eval_accuracy * 100) + '%')
    print('[INFO] Loss: ' + str(eval_loss))



if __name__ == "__main__":
    
    # Args
    args = {
        'img_width':              224,
        'img_height':             224,
        'top_model_path':         '{}/model_output/model.h5'.format( sys.path[0] ),
        'top_model_weights_path': '{}/model_output/weights.h5'.format( sys.path[0] ),
        'class_indices_path':     '{}/model_output/class_indices.npy'.format( sys.path[0] ),
        'bottleneck_training':    '{}/model_output/bottleneck_features_training.npy'.format( sys.path[0] ),
        'bottleneck_validation':  '{}/model_output/bottleneck_features_validation.npy'.format( sys.path[0] ),
        'train_data_dir':         '{}/data/training'.format( sys.path[0] ),
        'validation_data_dir':    '{}/data/validation'.format( sys.path[0] ),
        'epochs':                 50,
        'use_tensorboard':        True
    }
    
    # Setup callback for Tensorboard
    # This isn't required and slows down the training a bit, so comment out for faster speed
    if args['use_tensorboard']:
        from keras import callbacks
        tensorboard = callbacks.TensorBoard(log_dir='/tmp/keras_tensorboard', histogram_freq=0, write_graph=True, write_images=False)
    else:
        tensorboard = None
    
    total_files_training    = sum([len(files) for r, d, files in os.walk( args['train_data_dir'] )])
    min_files_training      = min([len(files) for r, d, files in os.walk( args['train_data_dir'] )][1:])
    batch_size_training     = calculate_batch_size(total_files_training, min_files_training)
    
    total_files_validation  = sum([len(files) for r, d, files in os.walk( args['validation_data_dir'] )])
    min_files_validation    = min([len(files) for r, d, files in os.walk( args['validation_data_dir'] )][1:])
    batch_size_validation   = calculate_batch_size(total_files_validation, min_files_validation)
    
    # Save Bottleneck Features to local (tmp) directory
    save_bottleneck_features(args, batch_size_training, batch_size_validation)
    
    # Train model and save output to local (tmp) directory
    train_top_model(args, tensorboard)



#ZEND
