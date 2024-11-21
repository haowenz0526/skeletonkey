import torch 
import DataManagerPytorch as DMP
import torchvision
import torch.nn.functional as F
import numpy as np
import cv2
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP



#This operation can all be done in one line but for readability later
#the projection operation is done in multiple steps for l-inf norm
def ProjectionOperation(xAdv, xClean, epsilonMax):
    #First make sure that xAdv does not exceed the acceptable range in the positive direction
    xAdv = torch.min(xAdv, xClean + epsilonMax)
    #Second make sure that xAdv does not exceed the acceptable range in the negative direction
    xAdv = torch.max(xAdv, xClean - epsilonMax)
    return xAdv

def resize_if_needed(x, target_size=(224, 224)):
    """
    Resize the input tensor if its spatial dimensions do not match the target size.
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        target_size (tuple): Target spatial dimensions (height, width)
    Returns:
        torch.Tensor: Resized tensor if resizing was needed; otherwise, original tensor
    """
    _, _, h, w = x.shape
    if (h, w) != target_size:
        print(f"Resizing input from ({h}, {w}) to {target_size}")
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
    return x

import torch.nn.functional as F

def SelfAttentionGradientAttack(device, epsMax, numSteps, modelListPlus, coefficientArray, dataLoader, clipMin, clipMax):
    # Basic gradient variable setup
    xClean, yClean = DMP.DataLoaderToTensor(dataLoader)
    xAdv = xClean  # Set the initial adversarial samples
    numSamples = len(dataLoader.dataset)
    xShape = DMP.GetOutputShape(dataLoader)
    epsStep = epsMax / numSteps
    dataLoaderCurrent = dataLoader
    
    for i in range(numSteps):
        print("Running Step=", i)
        xGradientCumulative = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        
        for m, currentModelPlus in enumerate(modelListPlus):
            dataLoaderCurrent = DMP.TensorToDataLoader(
                resize_if_needed(xAdv, target_size=(224, 224)),  # Resize to 224x224 for models requiring this input
                yClean, 
                transforms=None, 
                batchSize=dataLoader.batch_size, 
                randomizer=None
            )
            
            xGradientCurrent = FGSMNativeGradient(device, dataLoaderCurrent, currentModelPlus)
            xGradientCurrent = F.interpolate(xGradientCurrent, size=(xShape[1], xShape[2]))  # Resize gradient to original size

            if "ViT" in currentModelPlus.modelName:
                attmap = GetAttention(dataLoaderCurrent, currentModelPlus)
                
                # Resize `attmap` back to match `xGradientCurrent` (32x32 in this case)
                attmap = F.interpolate(attmap, size=(xShape[1], xShape[2]), mode='bilinear', align_corners=False)
                
                xGradientCumulative += coefficientArray[m] * xGradientCurrent * attmap
            else:
                xGradientCumulative += coefficientArray[m] * xGradientCurrent
        
        xAdv = xAdv + epsStep * xGradientCumulative.sign()
        #projectionOperation
        xAdv = ProjectionOperation(xAdv, xClean, epsMax)
        xAdv = torch.clamp(xAdv, clipMin, clipMax)
        torch.cuda.empty_cache()

    return DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)

# #Main attack method, takes in a list of models and a clean data loader
# #Returns a dataloader with the adverarial samples and corresponding clean labels
# def SelfAttentionGradientAttack(device, epsMax, numSteps, modelListPlus, coefficientArray, dataLoader, clipMin, clipMax):
#     #Basic graident variable setup 
#     xClean, yClean = DMP.DataLoaderToTensor(dataLoader)
#     xAdv = xClean #Set the initial adversarial samples 
#     numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
#     xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this) 
#     #Compute eps step
#     epsStep = epsMax/numSteps
#     dataLoaderCurrent = dataLoader
#     for i in range(0, numSteps):
#         print("Running Step=", i)
#         xGradientCumulative = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
#         for m in range(0, len(modelListPlus)):
#             currentModelPlus = modelListPlus[m]
#             #First get the gradient from the model 
#             xGradientCurrent = FGSMNativeGradient(device, dataLoaderCurrent, currentModelPlus)
#             #Resize the graident to be the correct size 
#             xGradientCurrent = torch.nn.functional.interpolate(xGradientCurrent, size=(xShape[1], xShape[2]))
#             #Add the current computed gradient to the result 
#             if currentModelPlus.modelName.find("ViT")>=0: 
#                 attmap = GetAttention(dataLoaderCurrent, currentModelPlus)
#                 xGradientCumulative = xGradientCumulative + coefficientArray[m]*xGradientCurrent *attmap
#             else:
#                 xGradientCumulative = xGradientCumulative + coefficientArray[m]*xGradientCurrent
            
#         #Compute the sign of the graident and create the adversarial example 
#         xAdv = xAdv + epsStep*xGradientCumulative.sign()
#         #Do the clipping 
#         xAdv = torch.clamp(xAdv, clipMin, clipMax)
#         #Convert the result to dataloader
#         dataLoaderCurrent = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)
#     return dataLoaderCurrent

def GetAttention(dLoader, modelPlus):
    numSamples = len(dLoader.dataset)
    attentionMaps = torch.zeros(numSamples, modelPlus.imgSizeH, modelPlus.imgSizeW,3)
    currentIndexer = 0
    model = modelPlus.model.to(modelPlus.device)
    for ii, (x, y) in enumerate(dLoader):
        x = x.to(modelPlus.device)
        y = y.to(modelPlus.device)
        bsize = x.size()[0]
        attentionMapBatch = get_attention_map(model, x, bsize)
        # for i in range(0, dLoader.batch_size):
        for i in range(0, bsize):
            attentionMaps[currentIndexer] = attentionMapBatch[i]
            currentIndexer = currentIndexer + 1 
    del model
    torch.cuda.empty_cache() 
    print("attention maps generated")
    # change order
    attentionMaps = attentionMaps.permute(0,3,1,2)
    return attentionMaps


# #Native (no attack library) implementation of the FGSM attack in Pytorch 
def FGSMNativeGradient(device, dataLoader, modelPlus):
    #Basic variable setup
    model = modelPlus.model
    model.eval() #Change model to evaluation mode for the attack 
    model.to(device)
    sizeCorrectedLoader = modelPlus.formatDataLoader(dataLoader)
    #Generate variables for storing the adversarial examples 
    numSamples = len(sizeCorrectedLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(sizeCorrectedLoader) #Get the shape of the input (there may be easier way to do this)
    xGradient = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    batchSize = 0 #just do dummy initalization, will be filled in later
    #Go through each sample 
    tracker = 0
    for xData, yData in sizeCorrectedLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        #print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device 
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        yData = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine) 
        xDataTemp.requires_grad = True
        # Forward pass the data through the model
        output = model(xDataTemp)
        # Calculate the loss
        loss = torch.nn.CrossEntropyLoss()
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = loss(output, yData).to(device)
        cost.backward()
        xDataTempGrad = xDataTemp.grad.data
        #Save the adversarial images from the batch 
        for j in range(0, batchSize):
            xGradient[advSampleIndex] = xDataTempGrad[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
        #Not sure if we need this but do some memory clean up 
        del xData, yData, output, xDataTempGrad
        torch.cuda.empty_cache()
    #Memory management 
    del model
    torch.cuda.empty_cache() 
    return xGradient

def GetFirstOverlappingSamplesBalanced(device, sampleNum, numClasses, dataLoader, modelPlusList):
    numModels = len(modelPlusList)
    totalSampleNum = len(dataLoader.dataset)
    xTestOrig, yTestOrig = DMP.DataLoaderToTensor(dataLoader)

    # Resize if necessary
    if modelPlusList[0].imgSizeH != xTestOrig.shape[2] or modelPlusList[0].imgSizeW != xTestOrig.shape[3]:
        xTestOrigResize = torch.zeros(xTestOrig.shape[0], 3, modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW)
        rs = torchvision.transforms.Resize((modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW))
        for i in range(xTestOrig.shape[0]):
            xTestOrigResize[i] = rs(xTestOrig[i])
        dataLoader = DMP.TensorToDataLoader(xTestOrigResize, yTestOrig, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)

    # Ensure sampleNum is divisible by numClasses
    if sampleNum % numClasses != 0:
        raise ValueError("Number of samples not divisible by the number of classes")

    # Basic setup for balanced sampling
    samplePerClassCount = torch.zeros(numClasses)
    maxRequireSamplesPerClass = sampleNum // numClasses
    xTest, yTest = DMP.DataLoaderToTensor(dataLoader)
    xClean = torch.zeros(sampleNum, 3, modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW)
    yClean = torch.zeros(sampleNum)
    sampleIndexer = 0

    # Iterate over the dataset
    for i in range(totalSampleNum):
        currentClass = int(yTest[i])
        
        # Add sample if it contributes to class balance and limit per class not yet reached
        if samplePerClassCount[currentClass] < maxRequireSamplesPerClass:
            xClean[sampleIndexer] = xTest[i]
            yClean[sampleIndexer] = yTest[i]
            sampleIndexer += 1
            samplePerClassCount[currentClass] += 1

            # Stop if we reach the required number of samples
            if sampleIndexer == sampleNum:
                break

    # Final check for class distribution
    if sampleIndexer != sampleNum:
        print("Not enough samples found to meet class balance.")

    # Convert to DataLoader
    cleanDataLoader = DMP.TensorToDataLoader(xClean, yClean, transforms=None, batchSize=modelPlusList[0].batchSize, randomizer=None)

    return cleanDataLoader
    
#Do the computation from scratch to get the correctly identified overlapping examples  
#Note these samples will be the same size as the input size required by the 0th model 
def GetFirstCorrectlyOverlappingSamplesBalanced(device, sampleNum, numClasses, dataLoader, modelPlusList):
    numModels = len(modelPlusList)
    totalSampleNum = len(dataLoader.dataset)
    #First check if modelA needs resize
    xTestOrig, yTestOrig = DMP.DataLoaderToTensor(dataLoader)
    #We need to resize first 
    if modelPlusList[0].imgSizeH != xTestOrig.shape[2] or modelPlusList[0].imgSizeW != xTestOrig.shape[3]:
        xTestOrigResize = torch.zeros(xTestOrig.shape[0], 3, modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW)
        rs = torchvision.transforms.Resize((modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW)) #resize the samples for model A
        #Go through every sample 
        for i in range(0, xTestOrig.shape[0]):
            xTestOrigResize[i] = rs(xTestOrig[i]) #resize to match dimensions required by modelA
        #Make a new dataloader
        dataLoader = DMP.TensorToDataLoader(xTestOrigResize, yTestOrig, transforms = None, batchSize=dataLoader.batch_size, randomizer=None)
    #Get accuracy array for each model 
    accArrayCumulative = torch.zeros(totalSampleNum) #Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(dataLoader)
        accArrayCumulative = accArrayCumulative + accArray
    #Do some basic error checking
    if sampleNum % numClasses != 0:
        raise ValueError("Number of samples not divisable by the number of classes")
    #Basic variable setup 
    samplePerClassCount = torch.zeros(numClasses) #keep track of samples per class
    maxRequireSamplesPerClass = int(sampleNum / numClasses) #Max number of samples we need per class
    xTest, yTest = DMP.DataLoaderToTensor(dataLoader) #Get all the data as tensors 
    #Memory for the solution 
    xClean = torch.zeros(sampleNum, 3, modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW)
    yClean = torch.zeros(sampleNum)
    sampleIndexer = 0
    #Go through all the samples
    for i in range(0, totalSampleNum):
        currentClass = int(yTest[i])
        #Check to make sure all classifiers identify the sample correctly AND we don't have enough of this class yet
        if accArrayCumulative[i] == numModels and samplePerClassCount[currentClass]<maxRequireSamplesPerClass:
            #xClean[sampleIndexer] = rs(xTest[i]) #resize to match dimensions required by modelA
            xClean[sampleIndexer] = xTest[i]
            yClean[sampleIndexer] = yTest[i]
            sampleIndexer = sampleIndexer +1 #update the indexer 
            samplePerClassCount[currentClass] = samplePerClassCount[currentClass] + 1 #Update the number of samples for this class
    #Check the over all number of samples as well
    if sampleIndexer != sampleNum:
        print("Not enough clean samples found.")
    #Do some error checking on the classes
    for i in range(0, numClasses):
        if samplePerClassCount[i] != maxRequireSamplesPerClass:
            print(samplePerClassCount[i])
            raise ValueError("We didn't find enough of class: "+str(i))
    #Conver the solution into a dataloader
    cleanDataLoader = DMP.TensorToDataLoader(xClean, yClean, transforms = None, batchSize = modelPlusList[0].batchSize, randomizer = None)
    #Do one last check to make sure all samples identify the clean loader correctly 
    for i in range(0, numModels):
        cleanAcc = modelPlusList[i].validateD(cleanDataLoader)
        if cleanAcc != 1.0:
            print("Clean Acc "+ modelPlusList[i].modelName+":", cleanAcc)
            raise ValueError("The clean accuracy is not 1.0")
    #All error checking done, return the clean balanced loader 
    return cleanDataLoader


def get_attention_map(model, xbatch, batch_size, img_size=224):
    _, _, img_size, _ = xbatch.shape  # Assuming xbatch has shape [batch_size, channels, height, width]
    attentionMaps = torch.zeros(batch_size,img_size, img_size,3)
    index = 0
    for i in range(0,batch_size):
        ximg = xbatch[i].cpu().numpy().reshape(1,3,img_size,img_size)
        ximg = torch.tensor(ximg).cuda()
        model.eval()
        res, att_mat = model.forward2(ximg)  
        # model should return attention_list for all attention_heads
        # each element in the list contains attention for each layer
       
        att_mat = torch.stack(att_mat).squeeze(1)
        # Average the attention weights across all heads.
        att_mat = torch.mean(att_mat, dim=1)

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat.cpu().detach() + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
    
        # Attention from the output token to the input space.
        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), (img_size, img_size))[..., np.newaxis]
        mask = np.concatenate((mask,)*3, axis=-1)
        #print(mask.shape)
        attentionMaps[index] = torch.from_numpy(mask)
        index = index + 1
    return attentionMaps