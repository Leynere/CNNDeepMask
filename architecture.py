from keras.layers import Dense,Input,Convolution2D,Flatten,Reshape, Dropout, MaxPooling2D
from keras.applications import vgg16
from keras.models import Model, model_from_json
from keras.layers.core import Lambda, Merge
from keras import backend as K
import numpy as np
import os.path

def crosschannelnormalization(alpha = 1e-4, k=2, beta=0.75, n=5,**kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """
    def f(X):
        b, ch, r, c = X.shape
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0,2,3,1))
                                              , (0,half))
        extra_channels = K.permute_dimensions(extra_channels, (0,3,1,2))
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:,i:i+ch,:,:]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape:input_shape,**kwargs)


def splittensor(axis=1, ratio_split=1, id_split=0,**kwargs):
    def f(X):
        div = X.shape[axis] // ratio_split

        if axis == 0:
            output =  X[id_split*div:(id_split+1)*div,:,:,:]
        elif axis == 1:
            output =  X[:, id_split*div:(id_split+1)*div, :, :]
        elif axis == 2:
            output = X[:,:,id_split*div:(id_split+1)*div,:]
        elif axis == 3:
            output = X[:,:,:,id_split*div:(id_split+1)*div]
        else:
            raise ValueError("This axis is not possible")

        return output

    def g(input_shape):
        output_shape=list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f,output_shape=lambda input_shape:g(input_shape),**kwargs)


def AlexNet(weights_path=None): 
    inputs = Input(shape=(3,227,227))

    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = merge([
        Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000,name='dense_3')(dense_3)
    prediction = Activation("softmax",name="softmax")(dense_3)


    model = Model(input=inputs, output=prediction)

    model.load_weights(weights_path)

    return model

def initAlexNet():
    anet = AlexNet("alexnet_weights")
    return anet

def initVgg16():
    vgg = vgg16.VGG16(weights="imagenet")
    inp = Input(shape=(224,224,3), name='in')
    shared_layers = vgg.layers[1](inp)
    for i in range(len(vgg.layers)):
        if(i>1 and i < len(vgg.layers)-5):
            shared_layers = vgg.layers[i](shared_layers)
    return (inp,shared_layers)

def getModel(filename):
    if(os.path.isfile(filename + '.json')):
        return loadModel(filename)
    
    inp, shared_layers = initVgg16();
    score_predictions = MaxPooling2D(pool_size=(2,2),strides=(2,2))(shared_layers)
    score_predictions = Flatten()(score_predictions)
    score_predictions = Dense(512,activation='relu')(score_predictions)
    score_predictions = Dropout(0.5)(score_predictions)
    score_predictions = Dense(10,activation='relu')(score_predictions)#to change in order to the number of classes
    score_predictions = Dropout(0.5)(score_predictions)
    score_predictions = Dense(1,name='score_out')(score_predictions)

    seg_predictions = Convolution2D(512,1,1,activation='relu')(shared_layers)
    seg_predictions = Flatten()(seg_predictions)
    seg_predictions = Dense(512)(seg_predictions)
    seg_predictions = Dense(56*56)(seg_predictions)
    seg_predictions = Reshape(target_shape=(56,56),name='seg_out')(seg_predictions)

    model = Model(input=inp,output=[seg_predictions,score_predictions])
    return model

def loadModel(filename):
    json_file = open(filename + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(filename + '.h5')
    return loaded_model

def saveModel(model, filename):
    model_json = model.to_json()
    with open(filename + '.json','w') as json_file:
        json_file.write(model_json)
        
    model.save_weights(filename + '.h5')