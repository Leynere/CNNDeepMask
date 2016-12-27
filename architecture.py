from keras.layers import Dense,Input,Convolution2D,Flatten,Reshape, Dropout, MaxPooling2D
from keras.applications import vgg16
from keras.models import Model


def getModel():
    vgg = vgg16.VGG16(weights="imagenet")
    inp = Input(shape=(225,224,3), name='in')
    shared_layers = vgg.layers[1](inp)
    for i in range(len(vgg.layers)):
        if(i>1 and i < len(vgg.layers)-5):
            shared_layers = vgg.layers[i](shared_layers)

    seg_predictions = Convolution2D(512,1,1,activation='relu')(shared_layers)
    seg_predictions = Flatten()(seg_predictions)
    seg_predictions = Dense(512)(seg_predictions)
    seg_predictions = Dense(56*56)(seg_predictions)
    seg_predictions = Reshape(target_shape=(56,56),name='seg_out')(seg_predictions)

    score_predictions = MaxPooling2D(pool_size=(2,2),strides=(2,2))(shared_layers)
    score_predictions = Flatten()(score_predictions)
    score_predictions = Dense(512,activation='relu')(score_predictions)
    score_predictions = Dropout(0.5)(score_predictions)
    score_predictions = Dense(1024,activation='relu')(score_predictions)#to change in order to the number of classes
    score_predictions = Dropout(0.5)(score_predictions)
    score_predictions = Dense(1,name='score_out')(score_predictions)

    model = Model(input=inp,output=[seg_predictions,score_predictions])
    return model