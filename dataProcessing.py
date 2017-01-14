import time
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import cv2


dataDir='../../coco-master'
dataType='train2014'
annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)


def getScore(mask):
    isCentered = -1
    centerFrame = 16
    offset = (224/2)-centerFrame
    for x in range(centerFrame*2):
        for y in range(centerFrame*2):
            if(mask[offset+x][offset+y] == 1):
                isCentered = 1
            if(isCentered == 1):
                break
        if(isCentered == 1):
            break 
            
    isNotTooLarge = 1
    if(isCentered == -1):
        return -1
    
    offset = (224-128)/2
    for x in range(128):
        if(mask[offset][offset+x] == 1):
            isNotTooLarge = -1
        if(mask[offset+x][offset] == 1):
            isNotTooLarge = -1
        if(mask[224 - offset][offset+x] == 1):
            isNotTooLarge = -1
        if(mask[offset+x][224 - offset] == 1):
            isNotTooLarge = -1
        if(isNotTooLarge == -1):
            break
    return isNotTooLarge

def setupMask(mask,length):
    for x in range(length):
        for y in range(length):
            if(mask[x][y] != 1):
                mask[x][y] = -1
    
    return mask

def getDatas(coco, cat, nbMax):
    
    tic = time.time()

    global dataDir
    global dataType
    global annFile
    
    catIds = coco.getCatIds(catNms=[cat]);
    imgIds = coco.getImgIds(catIds=catIds );
    nbPos = nbMax/2;
    nbNeg = nbMax/2;
    
    retIn = []
    retMask = []
    retScore = []
    
    #retIn = np.empty([224,224,3])
    #retMask = np.empty([56,56])
    #retScore = np.empty([1])
    
    for i in range(len(imgIds)):
        img = coco.loadImgs(imgIds[i])[0]
        I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
        I = cv2.resize(I,(224,224))

        '''I[:,:,0] -= 103.939
        I[:,:,1] -= 116.779
        I[:,:,2] -= 123.68'''
        
        if(I.shape == (224,224,3)):
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=0)
            anns = coco.loadAnns(annIds)
            for ann in anns:
                for seg in ann['segmentation']:
                    nI = Image.new('L', (img['width'], img['height']))
                    ImageDraw.Draw(nI).polygon(seg, outline=1, fill=1)
                    nI = np.asarray(nI)
                    nI = cv2.resize(nI, (224, 224))

                    sI = getScore(nI)
                    #nI = setupMask(nI,224)
                    nI = cv2.resize(nI,(56,56))
                    if((sI == -1 and nbNeg > 0) or (sI == 1 and nbPos > 0)):
                        retIn.append(I)
                        retMask.append(nI)
                        retScore.append(sI)
                        nbMax = nbMax - 1
                        if(nbMax <= 0):
                            print ('Done (t={:0.2f}s)'.format(time.time()- tic))
                            return (retIn, retMask, retScore)
                        if (sI == 1):
                            nbPos = nbPos -1
                        if (sI == -1):
                            nbNeg = nbNeg -1
                        
                        
def prepareAllData(nbElem, cats):
    global dataDir
    global dataType
    global annFile
    
    coco=COCO(annFile)
    allInputs = []
    allMasks = []
    allScores = []
    for catStr in cats:
        inputs, masks, scores = getDatas(coco, catStr, nbElem)
        allInputs.extend(inputs)
        allMasks.extend(masks)
        allScores.extend(scores)
    return (np.asarray(allInputs),np.asarray(allMasks), np.asarray(allScores))
    #return (allInputs,allMasks,allScores)
    
    