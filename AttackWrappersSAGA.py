import torch 
import DataManagerPytorch as DMP
import torchvision
import torch.nn.functional as F
import numpy as np
import cv2
import torch.nn as nn

# Returns a dataloader with the adverarial samples and corresponding clean labels
def SkeletonKeyAuto(device, epsMax, numSteps, modelListPlus, dataLoader, clipMin, clipMax,
                                         alphaLearningRate, fittingFactor):
    print("Using hard coded experimental function not advisable.")
    # Basic graident variable setup
    xClean, yClean = DMP.DataLoaderToTensor(dataLoader)
    xAdv = xClean # Set the initial adversarial samples
    xOridata = xClean
    xOriMax = xOridata + epsMax
    xOriMin = xOridata - epsMax
    numSamples = len(dataLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  # Get the shape of the input (there may be easier way to do this)
    # Compute eps step
    epsStep = epsMax / numSteps
    dataLoaderCurrent = dataLoader
    # Hardcoded for alpha right now, put in the method later
    confidence = 0
    nClasses = 10
    alpha = torch.ones(len(modelListPlus), numSamples, xShape[0], xShape[1],
                       xShape[2])  # alpha for every model and every sample
    # End alpha setup
    numSteps = 10
    for i in range(0, numSteps):
        print("Running step", i)
        # Keep track of dC/dX for each model where C is the cross entropy function
        dCdX = torch.zeros(len(modelListPlus), numSamples, xShape[0], xShape[1], xShape[2])
        # Keep track of dF/dX for each model where F, is the Carlini-Wagner loss function (for updating alpha)
        dFdX = torch.zeros(numSamples, xShape[0], xShape[1],
                           xShape[2])  # Change to the math here to take in account all objecitve functions
        # Go through each model and compute dC/dX
        for m in range(0, len(modelListPlus)):
            dataLoaderCurrent = modelListPlus[m].formatDataLoader(dataLoaderCurrent)
            dCdXTemp = FGSMNativeGradient(device, dataLoaderCurrent, modelListPlus[m])
            # Resize the graident to be the correct size and save it
            dCdX[m] = torch.nn.functional.interpolate(dCdXTemp, size=(xShape[1], xShape[2]))
            # Now compute the inital adversarial example with the base alpha
        xGradientCumulative = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])

        #change the sign of the targeted model and remain the sign for the non-targeted model

        #**changed

        xGradientCumulative = xGradientCumulative - alpha[m] * dCdX[m]
        for m in range(1, len(modelListPlus)):
            xGradientCumulative = xGradientCumulative + alpha[m] * dCdX[m]
        # Change the sign of the gradient
        #changed
        xAdvStepOne = xAdv + epsStep * xGradientCumulative.sign()
        # Convert the current xAdv to dataloader
        dataLoaderStepOne = DMP.TensorToDataLoader(xAdvStepOne, yClean, transforms=None,
                                                   batchSize=dataLoader.batch_size, randomizer=None)
        print("===Pre-Alpha Optimization===")
        costMultiplier = torch.zeros(len(modelListPlus), numSamples)
        for m in range(0, len(modelListPlus)):
            cost = CheckCarliniLoss(device, dataLoaderStepOne, modelListPlus[m], confidence, nClasses)
            costMultiplier[m] = CarliniSingleSampleLoss(device, dataLoaderStepOne, modelListPlus[m], confidence,
                                                        nClasses)
            print("For model", m, "the Carlini value is", cost)
        # Compute dF/dX (cumulative)
        for m in range(0, len(modelListPlus)):
            dFdX = dFdX + torch.nn.functional.interpolate(
                dFdXCompute(device, dataLoaderStepOne, modelListPlus[m], confidence, nClasses),
                size=(xShape[1], xShape[2]))
        # Compute dX/dAlpha
        dXdAlpha = dXdAlphaCompute(fittingFactor, epsStep, alpha, dCdX, len(modelListPlus), numSamples, xShape)
        # Compute dF/dAlpha = dF/dx * dX/dAlpha
        dFdAlpha = torch.zeros(len(modelListPlus), numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            # dFdAlpha = dFdX * dXdAlpha[m]
            dFdAlpha[m] = dFdX * dXdAlpha[m]
        # Now time to update alpha(changed)
        #changed
        alpha = alpha - dFdAlpha * alphaLearningRate
        # Compute final adversarial example using best alpha
        xGradientCumulativeB = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            xGradientCumulativeB = xGradientCumulativeB + alpha[m] * dCdX[m]
        xAdv = xAdv + epsStep * xGradientCumulativeB.sign()
        # clipMin = 0.0
        # clipMax = 1.0
        xAdv = torch.min(xOridata + epsMax, xAdv)
        xAdv = torch.max(xOridata - epsMax, xAdv)
        xAdv = torch.clamp(xAdv, clipMin, clipMax)
        xAdv = ProjectionOperation(xAdv, xClean, epsMax)
        
        # Convert the current xAdv to dataloader
        dataLoaderCurrent = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size,
                                                   randomizer=None)
        # Debug HERE
        print("===Post-Alpha Optimization===")
        for m in range(0, len(modelListPlus)):
            cost = CheckCarliniLoss(device, dataLoaderCurrent, modelListPlus[m], confidence, nClasses)
            print("For model", m, "the Carlini value is", cost)
    return dataLoaderCurrent


# Returns a dataloader with the adverarial samples and corresponding clean labels
def SelfAttentionGradientAttackProtoAuto(device, epsMax, numSteps, modelListPlus, dataLoader, clipMin, clipMax,
                                         alphaLearningRate, fittingFactor):
    print("Using hard coded experimental function not advisable.")
    # Basic graident variable setup
    xClean, yClean = DMP.DataLoaderToTensor(dataLoader)
    xAdv = xClean # Set the initial adversarial samples
    xOridata = xClean
    xOriMax = xOridata + epsMax
    xOriMin = xOridata - epsMax
    numSamples = len(dataLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  # Get the shape of the input (there may be easier way to do this)
    # Compute eps step
    epsStep = epsMax / numSteps
    dataLoaderCurrent = dataLoader
    # Hardcoded for alpha right now, put in the method later
    confidence = 0
    nClasses = 10
    alpha = torch.ones(len(modelListPlus), numSamples, xShape[0], xShape[1],
                       xShape[2])  # alpha for every model and every sample
    # End alpha setup
    numSteps = 10
    for i in range(0, numSteps):
        print("Running step", i)
        # Keep track of dC/dX for each model where C is the cross entropy function
        dCdX = torch.zeros(len(modelListPlus), numSamples, xShape[0], xShape[1], xShape[2])
        # Keep track of dF/dX for each model where F, is the Carlini-Wagner loss function (for updating alpha)
        dFdX = torch.zeros(numSamples, xShape[0], xShape[1],
                           xShape[2])  # Change to the math here to take in account all objecitve functions
        # Go through each model and compute dC/dX
        for m in range(0, len(modelListPlus)):
            dataLoaderCurrent = modelListPlus[m].formatDataLoader(dataLoaderCurrent)
            dCdXTemp = FGSMNativeGradient(device, dataLoaderCurrent, modelListPlus[m])
            # Resize the graident to be the correct size and save it
            dCdX[m] = torch.nn.functional.interpolate(dCdXTemp, size=(xShape[1], xShape[2]))
            # Now compute the inital adversarial example with the base alpha
        xGradientCumulative = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            xGradientCumulative = xGradientCumulative + alpha[m] * dCdX[m]
        xAdvStepOne = xAdv + epsStep * xGradientCumulative.sign()
        # Convert the current xAdv to dataloader
        dataLoaderStepOne = DMP.TensorToDataLoader(xAdvStepOne, yClean, transforms=None,
                                                   batchSize=dataLoader.batch_size, randomizer=None)
        print("===Pre-Alpha Optimization===")
        costMultiplier = torch.zeros(len(modelListPlus), numSamples)
        for m in range(0, len(modelListPlus)):
            cost = CheckCarliniLoss(device, dataLoaderStepOne, modelListPlus[m], confidence, nClasses)
            costMultiplier[m] = CarliniSingleSampleLoss(device, dataLoaderStepOne, modelListPlus[m], confidence,
                                                        nClasses)
            print("For model", m, "the Carlini value is", cost)
        # Compute dF/dX (cumulative)
        for m in range(0, len(modelListPlus)):
            dFdX = dFdX + torch.nn.functional.interpolate(
                dFdXCompute(device, dataLoaderStepOne, modelListPlus[m], confidence, nClasses),
                size=(xShape[1], xShape[2]))
        # Compute dX/dAlpha
        dXdAlpha = dXdAlphaCompute(fittingFactor, epsStep, alpha, dCdX, len(modelListPlus), numSamples, xShape)
        # Compute dF/dAlpha = dF/dx * dX/dAlpha
        dFdAlpha = torch.zeros(len(modelListPlus), numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            # dFdAlpha = dFdX * dXdAlpha[m]
            dFdAlpha[m] = dFdX * dXdAlpha[m]
        # Now time to update alpha
        alpha = alpha - dFdAlpha * alphaLearningRate
        # Compute final adversarial example using best alpha
        xGradientCumulativeB = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            xGradientCumulativeB = xGradientCumulativeB + alpha[m] * dCdX[m]
        xAdv = xAdv + epsStep * xGradientCumulativeB.sign()
        # clipMin = 0.0
        # clipMax = 1.0
        xAdv = torch.min(xOridata + epsMax, xAdv)
        xAdv = torch.max(xOridata - epsMax, xAdv)
        xAdv = torch.clamp(xAdv, clipMin, clipMax)
        # Convert the current xAdv to dataloader
        dataLoaderCurrent = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size,
                                                   randomizer=None)
        # Debug HERE
        print("===Post-Alpha Optimization===")
        for m in range(0, len(modelListPlus)):
            cost = CheckCarliniLoss(device, dataLoaderCurrent, modelListPlus[m], confidence, nClasses)
            print("For model", m, "the Carlini value is", cost)
    return dataLoaderCurrent


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


def SelfAttentionGradientAttackMDSE(
    device, epsMax, numSteps, modelListPlus, coefficientArray, dataLoader, 
    clipMin, clipMax, sigma, learningRate, k=0
):
    # Helper functions
    def sech(x):
        return 1 / torch.cosh(x)

    def F_loss(x_adv, y_clean, model, k=0):
        
        with torch.no_grad():
            logits = model(x_adv)  # Forward pass
            softmax_outputs = F.softmax(logits, dim=1)
            correct_class_scores = softmax_outputs[range(len(y_clean)), y_clean]
            max_other_class_scores = torch.max(
                softmax_outputs.masked_fill(F.one_hot(y_clean, num_classes=softmax_outputs.shape[1]).bool(), -1e10), dim=1
            )[0]
            return torch.clamp(correct_class_scores - max_other_class_scores + kappa, min=0)

    # Initialize variables
    xClean, yClean = DMP.DataLoaderToTensor(dataLoader)
    xAdv = xClean  # Initial adversarial examples
    numSamples = len(dataLoader.dataset)
    xShape = DMP.GetOutputShape(dataLoader)
    epsStep = epsMax / numSteps

    # Iterate through the attack steps
    for i in range(numSteps):
        print(f"Running Step {i + 1}")
        xGradientCumulative = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])

        for m, currentModelPlus in enumerate(modelListPlus):
            dataLoaderCurrent = DMP.TensorToDataLoader(
                resize_if_needed(xAdv, target_size=(224, 224)),
                yClean, 
                transforms=None, 
                batchSize=dataLoader.batch_size, 
                randomizer=None
            )
            
            # Compute gradients for current model
            xGradientCurrent = FGSMNativeGradient(device, dataLoaderCurrent, currentModelPlus)
            xGradientCurrent = F.interpolate(xGradientCurrent, size=(xShape[1], xShape[2]))

            # Attention rollout for ViT models
            if "ViT" in currentModelPlus.modelName:
                attmap = GetAttention(dataLoaderCurrent, currentModelPlus)
                attmap = F.interpolate(attmap, size=(xShape[1], xShape[2]), mode='bilinear', align_corners=False)
                xGradientCumulative += coefficientArray[m] * xGradientCurrent * attmap
            else:
                xGradientCumulative += coefficientArray[m] * xGradientCurrent

        # Generate adversarial examples
        xAdv = xAdv + epsStep * xGradientCumulative.sign()
        xAdv = ProjectionOperation(xAdv, xClean, epsMax)
        xAdv = torch.clamp(xAdv, clipMin, clipMax)

        # Update coefficients
        for m, currentModelPlus in enumerate(modelListPlus):
            dataLoaderCurrent = DMP.TensorToDataLoader(
                resize_if_needed(xAdv, target_size=(224, 224)),
                yClean, 
                transforms=None, 
                batchSize=dataLoader.batch_size, 
                randomizer=None
            )
            # Compute F_loss for current model
            F_current = F_loss(xAdv, yClean, currentModelPlus.model, kappa)

            # Compute gradient of x_adv with respect to α_m (Equation 26)
            xGradientCurrent = FGSMNativeGradient(device, dataLoaderCurrent, currentModelPlus)
            xGradientCurrent = F.interpolate(xGradientCurrent, size=(xShape[1], xShape[2]))
            sech_term = sech(sigma * torch.sum(xGradientCurrent, dim=(1, 2, 3)))**2
            grad_alpha = epsStep * sech_term * (xGradientCurrent * coefficientArray[m]).sum(dim=(1, 2, 3))

            # Update α_m using gradient descent (Equation 24)
            coefficientArray[m] = coefficientArray[m] - learningRate * grad_alpha

    return DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None), coefficientArray



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

# Custom loss function for updating alpha
# def UntargetedCarliniLoss(logits, targets, confidence, nClasses, device):
#     # This converts the normal target labels to one hot vectors e.g. y=1 will become [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
#     yOnehot = torch.nn.functional.one_hot(targets, nClasses).to(torch.float)
#     zC = torch.max(yOnehot * logits,
#                    1).values  # Need to use .values to get the Tensor because PyTorch max function doesn't want to give us a tensor
#     zOther = torch.max((1 - yOnehot) * logits, 1).values
#     loss = torch.max(zC - zOther + confidence, torch.tensor(0.0).to(device))
#     return loss

def dFdXCompute(device, dataLoader, modelPlus, confidence, nClasses):
    # Basic variable setup
    model = modelPlus.model
    model.eval()  # Change model to evaluation mode for the attack
    model.to(device)
    sizeCorrectedLoader = modelPlus.formatDataLoader(dataLoader)
    # Generate variables for storing the adversarial examples
    numSamples = len(sizeCorrectedLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(sizeCorrectedLoader)  # Get the shape of the input (there may be easier way to do this)
    xGradient = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  # just do dummy initalization, will be filled in later
    # Go through each sample
    tracker = 0
    for xData, yData in sizeCorrectedLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        # print("Processing up to sample=", tracker)
        # Put the data from the batch onto the device
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        yData = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
        xDataTemp.requires_grad = True
        if modelPlus.modelName == 'SNN ResNet Backprop':
            functional.reset_net(model)  # Line to reset model memory to accodomate Spiking Jelly (new attack iteration)
            # Forward pass the data through the model
            outputLogits = model(xDataTemp).mean(0)
        else:
            # Forward pass the data through the model
            outputLogits = model(xDataTemp)
        # Calculate the loss with respect to the Carlini Wagner loss function
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = UntargetedCarliniLoss(outputLogits, yData, confidence, nClasses, device).sum().to(
            device)  # Not sure about the sum
        cost.backward()
        if modelPlus.modelName == 'SNN VGG-16 Backprop':
            xDataTempGrad = xDataTemp.grad.data.sum(-1)
        else:
            xDataTempGrad = xDataTemp.grad.data
        # Save the adversarial images from the batch
        for j in range(0, batchSize):
            xGradient[advSampleIndex] = xDataTempGrad[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex + 1  # increment the sample index
        # Not sure if we need this but do some memory clean up
        del xDataTemp
        torch.cuda.empty_cache()
    # Memory management
    del model
    torch.cuda.empty_cache()
    return xGradient

def dXdAlphaCompute(fittingFactor, epsStep, alpha, dCdX, numModels, numSamples, xShape):
    # Allocate memory for the solution
    dXdAlpha = torch.zeros(numModels, numSamples, xShape[0], xShape[1], xShape[2])
    innerSum = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    # First compute the inner summation sum m=1,...M: a_{m}*dC/dX_{m}
    for m in range(0, numModels):
        innerSum = innerSum + alpha[m] * dCdX[m]
    # Multiply inner sum by the fitting factor to approximate the sign(.) function
    innerSum = innerSum * fittingFactor
    # Now compute the sech^2 of the inner sum
    innerSumSecSquare = SechSquared(innerSum)
    # Now do the final computation to get dX/dAlpha (may not actually need for loop)
    for m in range(0, numModels):
        dXdAlpha[m] = fittingFactor * epsStep * dCdX[m] * innerSumSecSquare
    # All done so return
    return dXdAlpha

# Compute sech^2(x) using torch functions
def SechSquared(x):
    y = 4 * torch.exp(2 * x) / ((torch.exp(2 * x) + 1) * (torch.exp(2 * x) + 1))
    return y
def UntargetedCarliniLoss(logits, targets, confidence, nClasses, device):
    """
    Custom loss function for updating alpha using the Carlini & Wagner untargeted loss.
    
    Args:
        logits (torch.Tensor): Logits from the model, shape (batch_size, nClasses).
        targets (torch.Tensor): Ground truth labels, shape (batch_size).
        confidence (float): Confidence margin for the attack.
        nClasses (int): Number of classes.
        device (torch.device): Device to run the computation on.
        
    Returns:
        torch.Tensor: Loss tensor, shape (batch_size).
    """
    # Check and reshape logits if necessary
    if logits.ndim != 2 or logits.size(1) != nClasses:
        raise ValueError(f"Expected logits to have shape (batch_size, nClasses), but got {logits.shape}")

    # Check and convert targets to one-hot encoding
    if targets.ndim != 1:
        targets = targets.view(-1)  # Ensure targets is a 1D tensor
    
    yOnehot = torch.nn.functional.one_hot(targets, nClasses).to(torch.float).to(device)

    # Ensure one-hot encoding matches the shape of logits
    if yOnehot.size(0) != logits.size(0):
        raise ValueError(f"Batch size mismatch: logits {logits.size(0)} vs targets {yOnehot.size(0)}")

    # Compute zC and zOther
    zC = torch.max(yOnehot * logits, dim=1).values  # Max over each row (batch)
    zOther = torch.max((1 - yOnehot) * logits, dim=1).values

    # Compute loss
    loss = torch.max(zC - zOther + confidence, torch.tensor(0.0).to(device))
    return loss

# Get the loss associated with single samples
def CarliniSingleSampleLoss(device, dataLoader, modelPlus, confidence, nClasses):
    # Basic variable setup
    model = modelPlus.model
    model.eval()  # Change model to evaluation mode for the attack
    model.to(device)
    sizeCorrectedLoader = modelPlus.formatDataLoader(dataLoader)
    # Generate variables for storing the adversarial examples
    numSamples = len(sizeCorrectedLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(sizeCorrectedLoader)  # Get the shape of the input (there may be easier way to do this)
    xGradient = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    # Variables to store the associated costs values
    costValues = torch.zeros(numSamples)
    batchSize = 0  # just do dummy initalization, will be filled in later
    # Go through each sample
    tracker = 0
    for xData, yData in sizeCorrectedLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        # Put the data from the batch onto the device
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        yData = yData.type(torch.LongTensor).to(device)
        if modelPlus.modelName == 'SNN ResNet Backprop':
            functional.reset_net(model)  # Line to reset model memory to accodomate Spiking Jelly (new attack iteration)
            # Forward pass the data through the model
            outputLogits = model(xDataTemp).mean(0)
        else:
            # Forward pass the data through the model
            outputLogits = model(xDataTemp)
        # Calculate the loss with respect to the Carlini Wagner loss function
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = UntargetedCarliniLoss(outputLogits, yData, confidence, nClasses, device)
        cost.sum().backward()
        # Store the current cost values
        costValues[tracker:tracker + batchSize] = cost.to("cpu")
        tracker = tracker + batchSize
        # Not sure if we need this but do some memory clean up
        del xDataTemp
        torch.cuda.empty_cache()
    # Memory management
    del model
    torch.cuda.empty_cache()
    return costValues
def CheckCarliniLoss(device, dataLoader, modelPlus, confidence, nClasses):
    # Basic variable setup
    model = modelPlus.model
    model.eval()  # Change model to evaluation mode for the attack
    model.to(device)
    sizeCorrectedLoader = modelPlus.formatDataLoader(dataLoader)
    # Generate variables for storing the adversarial examples
    numSamples = len(sizeCorrectedLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(sizeCorrectedLoader)  # Get the shape of the input (there may be easier way to do this)
    xGradient = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  # just do dummy initalization, will be filled in later
    # Go through each sample
    tracker = 0
    cumulativeCost = 0
    for xData, yData in sizeCorrectedLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        # print("Processing up to sample=", tracker)
        # Put the data from the batch onto the device
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        yData = yData.type(torch.LongTensor).to(device)
        # Forward pass the data through the model
        if modelPlus.modelName == 'SNN ResNet Backprop':
            functional.reset_net(model)  # Line to reset model memory to accodomate Spiking Jelly (new attack iteration)
            # Forward pass the data through the model
            outputLogits = model(xDataTemp).mean(0)
        else:
            # Forward pass the data through the model
            outputLogits = model(xDataTemp)
        # Calculate the loss with respect to the Carlini Wagner loss function
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = UntargetedCarliniLoss(outputLogits, yData, confidence, nClasses, device).sum()  # Not sure about the sum
        cumulativeCost = cumulativeCost + cost.to("cpu")
        cost.backward()
        # Not sure if we need this but do some memory clean up
        del xDataTemp
        torch.cuda.empty_cache()
    # Memory management
    del model
    torch.cuda.empty_cache()
    return cumulativeCost
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

        # Align output and target batch sizes
        if output.size(0) != yData.size(0):
            min_batch_size = min(output.size(0), yData.size(0))
            output = output[:min_batch_size]
            yData = yData[:min_batch_size]
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
        if cleanAcc < 0.996666666666667:
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
