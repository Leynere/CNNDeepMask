from keras.layers import Dense,Input,Convolution2D,Flatten,Reshape, Dropout, MaxPooling2D
from keras.applications import vgg16
from keras.models import Model, model_from_json
import os.path

def getModel():
    if(os.path.isfile('model.json')):
        return loadModel('model')
    vgg = vgg16.VGG16(weights="imagenet")
    inp = Input(shape=(224L,224L,3L), name='in')
    shared_layers = vgg.layers[1](inp)
    for i in range(len(vgg.layers)):
        if(i>1 and i < len(vgg.layers)-5):
            shared_layers = vgg.layers[i](shared_layers)
            #print vgg.layers[i]

    seg_predictions = Convolution2D(512,1,1,activation='relu')(shared_layers)
    seg_predictions = Flatten()(seg_predictions)
    seg_predictions = Dense(512)(seg_predictions)
    seg_predictions = Dense(56*56)(seg_predictions)
    seg_predictions = Reshape(target_shape=(56,56),name='seg_out')(seg_predictions)

    score_predictions = MaxPooling2D(pool_size=(2,2),strides=(2,2))(shared_layers)
    score_predictions = Flatten()(score_predictions)
    score_predictions = Dense(512,activation='relu')(score_predictions)
    score_predictions = Dropout(0.5)(score_predictions)
    score_predictions = Dense(10,activation='relu')(score_predictions)#to change in order to the number of classes
    score_predictions = Dropout(0.5)(score_predictions)
    score_predictions = Dense(1,name='score_out')(score_predictions)

    model = Model(input=inp,output=[seg_predictions,score_predictions])
    return model

def loadModel(fileName):
    json_file = open(filename + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(filename + '.h5')
    return loaded_model

def saveModel(model, fileName):
    model_json = model.to_json()
    with open(filename + '.json') as json_file:
        json_file.write(model_json)
        
    model.save_weights(filename + '.h5')