from architecture import getModel,saveModelvModel
from objective import segObjectiveFunction,scoreObjectiveFunction
from dataProcessing import getScore, getDatas, prepareAllData

inputs, masks, scores = prepareAllData()

model = getModel()
model.compile(optimizer='sgd',loss={'seg_out': segObjectiveFunction, 'score_out': scoreObjectiveFunction})
model.fit(inputs, {'seg_out': masks, 'score_out': scores}, nb_epoch=10, batch_size=32, verbose=2, shuffle=True)