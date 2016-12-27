from architecture import getModel
from objective import segObjectiveFunction,scoreObjectiveFunction

model = getModel()
model.compile(optimizer='sgd',loss={'seg_out': segObjectiveFunction, 'score_out': scoreObjectiveFunction})