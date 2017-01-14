from keras import backend

def segObjectiveFunction(y_true,y_pred):
    '''factor = (1 + y_true[1][0][0])/2*56*56
    return factor * backend.sum(backend.log(1 + backend.exp(-y_true*y_pred)))'''
    return 0.5 * (1 - y_true[1][0][0]) * backend.mean(backend.log(1 + backend.exp(-y_true*y_pred)))

def scoreObjectiveFunction(y_true,y_pred):
    lambd = 1/32
    return lambd * backend.log(1 + backend.exp(-y_true*y_pred))