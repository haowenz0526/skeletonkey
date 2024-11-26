import torch
import numpy
import ShuffleDefense
from ModelPlus import ModelPlus
import DataManagerPytorch as DMP
import AttackWrappersRayS
import AttackWrappersAdaptiveBlackBox
import AttackWrappersSAGA
from TransformerModels import VisionTransformer, CONFIGS
import BigTransferModels
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

#Load the shuffle defense containing ViT-L-16 and BiT-M-R101x3
#For all attacks except SAGA, vis should be false (makes the Vision tranformer return the attention weights if true)
def LoadShuffleDefenseAndCIFAR10(vis=False):
    modelPlusList = []
    #Basic variable and data setup
    device = torch.device("cuda")
    numClasses = 10
    imgSize = 224
    batchSize = 8
    #Load the CIFAR-10 data
    valLoader = DMP.GetCIFAR10Validation(32, batchSize)
    #Load ViT-L-16
    config = CONFIGS["ViT-L_16"]
    model = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses, vis = vis)
    dir = "/data2/shh20007/haowenz/VisionTransformersRobustness/data/ViT-L_16,cifar10,run0_15K_checkpoint.bin"
    dict = torch.load(dir)
    model.load_state_dict(dict)
    model.eval()
    #Wrap the model in the ModelPlus class
    modelPlusV = ModelPlus("ViT-L_16", model, device, imgSizeH=imgSize, imgSizeW=imgSize, batchSize=batchSize)
    modelPlusList.append(modelPlusV)
    #Load the BiT-M-R101x3
    dirB = "/data2/shh20007/haowenz/VisionTransformersRobustness/data/BiT-M-R101x3-Run0.tar"
    modelB = BigTransferModels.KNOWN_MODELS["BiT-M-R101x3"](head_size=numClasses, zero_head=False)
    #Get the checkpoint 
    checkpoint = torch.load(dirB, map_location="cpu")
    #Remove module so that it will load properly
    new_state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #Load the dictionary
    modelB.load_state_dict(new_state_dict)
    modelB.eval()
    #Wrap the model in the ModelPlus class
    #Here we hard code the Big Transfer Model Plus class input size to 160x128 (what it was trained on)
    modelBig101Plus = ModelPlus("BiT-M-R101x3", modelB, device, imgSizeH=160, imgSizeW=128, batchSize=batchSize)
    modelPlusList.append(modelBig101Plus)
    #Now time to build the defense 
    defense = ShuffleDefense.ShuffleDefense(modelPlusList, numClasses)
    return valLoader, defense




def GradientAttack(device, cleanLoader, stepSizeforAttack, steps, modelListPlus, weightF, clipMin, clipMax):
    x, y = DMP.DataLoaderToTensor(cleanLoader)
    xadv = x
    for i in range(steps):
        G = torch.zeros(x.shape) #
        for j in range(len(modelListPlus)):
            currModel = modelListPlus[j].model 
            currModel.eval()
            currModel.to(device)
            dataLoader = currentModel.formatDataLoader(cleanLoader)
            x1, y1 = DMP.DataLoaderToTensor(dataLoader)
            xG = torch.zeros(x1.shape)

            for x, y in dataLoader:
                x.requires_grad = True
                loss = torch.nn.CrossEntropyLoss()
                y_ = model(x)
                cost = loss(y_, y).to(device)
                cost.backward()
                gradient = x.grad.data
                for index in range(x.shape[0]):
                    xG[index] = gradient[index]
            
            xG = f.interpolate(xG, size = (x.shape[2], x.shape[3]))
            xG += weightF[j] * xG
        
        xadv = xadv + stepSizeforAttack * xG.sign()
        xadv = torch.clamp(xadv, clipMin, clipMax)

    return DMP.TensorToDataLoader(xadv, y, batchSize = x.shape[1])

#Native (no attack library) implementation of the MIM attack in Pytorch 
#This is only for the L-infinty norm and cross entropy loss function 
def MIMNativePytorch(device, dataLoader, model, decayFactor, epsilonMax, epsilonStep, numSteps, clipMin, clipMax, targeted):
    model.eval() #Change model to evaluation mode for the attack 
    #Generate variables for storing the adversarial examples 
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    batchSize = 0 #just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    #Go through each sample 
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device 
        xAdvCurrent = xData.to(device)
        yCurrent = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine) 
        #Initalize memory for the gradient momentum
        gMomentum = torch.zeros(batchSize, xShape[0], xShape[1], xShape[2])
        #Do the attack for a number of steps
        for attackStep in range(0, numSteps):   
            xAdvCurrent.requires_grad = True
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()
            #Update momentum term 
            gMomentum = decayFactor*gMomentum + GradientNormalizedByL1(xAdvCurrent.grad)
            #Update the adversarial sample 
            if targeted == True:
                advTemp = xAdvCurrent - (epsilonStep*torch.sign(gMomentum)).to(device)
            else:
                advTemp = xAdvCurrent + (epsilonStep*torch.sign(gMomentum)).to(device)
            #Adding clipping to maintain the range
            xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()
        #Save the adversarial images from the batch 
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index 
    #All samples processed, now time to save in a dataloader and return 
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader

def GradientNormalizedByL1(gradient):
    #Do some basic error checking first
    if gradient.shape[1] != 3:
        raise ValueError("Shape of gradient is not consistent with an RGB image.")
    #basic variable setup
    batchSize = gradient.shape[0]
    colorChannelNum = gradient.shape[1]
    imgRows = gradient.shape[2]
    imgCols = gradient.shape[3]
    gradientNormalized = torch.zeros(batchSize, colorChannelNum, imgRows, imgCols)
    #Compute the L1 gradient for each color channel and normalize by the gradient
    #Go through each color channel and compute the L1 norm
    for i in range(0, batchSize):
        for c in range(0, colorChannelNum):
           norm = torch.linalg.norm(gradient[i,c], ord=1)
           gradientNormalized[i,c] = gradient[i,c]/norm #divide the color channel by the norm
    return gradientNormalized

#This operation can all be done in one line but for readability later
#the projection operation is done in multiple steps for l-inf norm
def ProjectionOperation(xAdv, xClean, epsilonMax):
    #First make sure that xAdv does not exceed the acceptable range in the positive direction
    xAdv = torch.min(xAdv, xClean + epsilonMax)
    #Second make sure that xAdv does not exceed the acceptable range in the negative direction
    xAdv = torch.max(xAdv, xClean - epsilonMax)
    return xAdv


if __name__ == '__main__':
    #parameter setting

    numAttackSamples = 1000
    epsForAttacks = 0.031  # Epsilon for attack
    targeted = False         # Untargeted attack
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numClasses = 10
    imgSize = 224
    batchSize = 8
    decayFactor = 1.0
    numSteps = 10 
    epsStep = epsForAttacks/numSteps
    clipMin, clipMax = 0.0, 1.0


    #Load dataset and model
    valLoader, defense = LoadShuffleDefenseAndCIFAR10(vis=False)
    modelPlusList = defense.modelPlusList

    modelPlusList[0].batchSize = 8 #ViT
    modelPlusList[1].batchSize = 2 #BiT

    #Get the cleanloader for both models
    # cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(modelPlusList[1], numAttackSamples, valLoader, numClasses)
    cleanLoader = AttackWrappersSAGA.GetFirstCorrectlyOverlappingSamplesBalanced(device, numAttackSamples, numClasses, valLoader, modelPlusList)
    advLoader = MIMNativePytorch(device, cleanLoader, modelPlusList[0].model, decayFactor, epsForAttacks, epsStep, numSteps, clipMin, clipMax, targeted = False)
    advLoader_transfer = modelPlusList[1].formatDataLoader(advLoader)

    acc = modelPlusList[1].validateD(advLoader_transfer)
    print(modelPlusList[1].modelName+" Transfer Acc:", acc)





