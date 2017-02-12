from architecture import getModel,saveModel
from objective import segObjectiveFunction,scoreObjectiveFunction
from dataProcessing import getScore, getDatas, prepareAllData
from keras.optimizers import SGD

model = getModel('none')
sgd = SGD(lr=0.001, decay=0.00005,momentum=0.9, nesterov=True,clipvalue=500)
model.compile(optimizer=sgd,loss={'score_out': scoreObjectiveFunction, 'seg_out': segObjectiveFunction})

inputs, masks, scores = prepareAllData(1000,['outdoor', 'food', 'indoor', 'appliance', 'sports', 'person', 'animal', 'vehicle', 'furniture', 'accessory'],offset)

model.fit({'in' : inputs}, { 'score_out': scores, 'seg_out': masks}, nb_epoch=1, batch_size=32, verbose=2, shuffle=True)

saveModel(model,"deepmask10000")