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
from ResNetPytorch import resnet56, resnet164
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torch.nn.functional as f
from wideresnet import wideresnet
from wideresnetwithswish import wideresnetwithswish
import os
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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
    # valLoader = DMP.GetCIFAR10Training(32, batchSize)


    # Load ResNet 164
    modelR = resnet56(32, numClasses)
    # dirR = "/data2/shh20007/haowenz/VisionTransformersRobustness/data/ModelResNet164-Run0.th"
    dirR = "/data2/shh20007/haowenz/VisionTransformersRobustness/data/ModelResNet56-Run0.th"
    checkpointR = torch.load(dirR)
    state_dict = checkpointR['state_dict']
    modelR.load_state_dict(state_dict)
   
    modelR.eval()
    #Wrap the ResNet model in the ModelPlus class
    modelPlusR = ModelPlus("ResNet-56", modelR, device, imgSizeH=32, imgSizeW=32, batchSize=batchSize)
    modelPlusList.append(modelPlusR)
    

    # # Load ViT-L-16
    # config = CONFIGS["ViT-L_16"]
    # model = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses, vis = vis)
    # dir = "/data2/shh20007/haowenz/VisionTransformersRobustness/data/ViT-L_16,cifar10,run0_15K_checkpoint.bin"
    # dict = torch.load(dir)
    # model.load_state_dict(dict)
    # model.eval()
    # #Wrap the model in the ModelPlus class
    # modelPlusV = ModelPlus("ViT-L_16", model, device, imgSizeH=224, imgSizeW=224, batchSize=batchSize)
    # modelPlusList.append(modelPlusV)

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
    

    # Instantiate the model
    model_name = "WRN-28-10-swish" 
    backbone = wideresnetwithswish(model_name)
    modelD = torch.nn.Sequential(backbone)
    # Load the pre-trained weights
    # checkpoint_path = '/data2/shh20007/haowenz/DM-Improves-AT/trained_models/mymodel/weights-best.pt'
    checkpoint_path = '/data2/shh20007/haowenz/DM-Improves-AT/trained_models/mymodel/cifar10_l2_wrn28-10.pt'
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Remove `module.` prefix
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    modelD.load_state_dict(new_state_dict, strict=False)
    # Set model to evaluation mode
    modelD.eval()  
    modelDPlus = ModelPlus("Difussion", modelD, device, imgSizeH=224, imgSizeW=224, batchSize=512)
    modelPlusList.append(modelDPlus)
    

    #Now time to build the defense 
    defense = ShuffleDefense.ShuffleDefense(modelPlusList, numClasses)
    return valLoader, defense


#Run the Self-Attention Gradient Attack (SAGA) on ViT-L and BiT-M-R101x3
def Skeleton_CIFAR10():
    #Set up the parameters for the attack 
    attackSampleNum = 1000
    numClasses = 10
    coefficientArray = torch.zeros(3)
    secondcoeff = 2.0000e-04
    coefficientArray[2] = 1 - secondcoeff
    coefficientArray[0] = - (secondcoeff / 2)
    coefficientArray[1] = - (secondcoeff / 2)
    
    print("Coeff Array:")
    print(coefficientArray)
    device = torch.device("cuda")
    epsMax = 0.031
    clipMin = 0.0
    clipMax = 1.0
    numSteps = 10
    #Load the models and the dataset
    #Note it is important to set vis to true so the transformer's model output returns the attention weights 
    valLoader, defense = LoadShuffleDefenseAndCIFAR10(vis=True)
    modelPlusList = defense.modelPlusList
    
    #Note that the batch size will effect how the gradient is computed in PyTorch
    #Here we use batch size 8 for ViT-L and batch size 2 for BiT-M. Other batch sizes are possible but they will not generate the same result
    # modelPlusList[0].batchSize = 2
    # modelPlusList[1].batchSize = 8

    
    #Get the clean examples 
    # cleanLoader = AttackWrappersSAGA.GetFirstCorrectlyOverlappingSamplesBalanced(device, attackSampleNum, numClasses, valLoader, modelPlusList)
    balancedLoader = AttackWrappersSAGA.GetFirstOverlappingSamplesBalanced(device, attackSampleNum, numClasses, valLoader, modelPlusList)
    #Do the attack
    advLoader = AttackWrappersSAGA.SelfAttentionGradientAttack(device, epsMax, numSteps, modelPlusList, coefficientArray, balancedLoader, clipMin, clipMax)


    # for i in range(0, len(modelPlusList)):
    #     acc = modelPlusList[i].validateD(advLoader)
    #     print(modelPlusList[i].modelName+" Robust Acc:", acc)
    acc = modelPlusList[2].validateD(advLoader)
    print(acc)


    # batchSize = 8
    # model = resnet56(32, numClasses)
    # dir1 = "/data2/shh20007/haowenz/VisionTransformersRobustness/data/ModelResNet56-Run0.th"
    # model = resnet164(32, numClasses)
    # dir1 = "/data2/shh20007/haowenz/VisionTransformersRobustness/data/ModelResNet164-Run1.th"

    # config = CONFIGS["ViT-L_16"]
    # numClasses = 10
    # imgSize = 224
    # model = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses, vis = False)
    # dir1 = "/data2/shh20007/haowenz/VisionTransformersRobustness/data/ViT-L_16,cifar10,run1_15K_checkpoint.bin"
    # checkpoint = torch.load(dir1)
    # # state_dict = checkpoint['state_dict']
    # model.load_state_dict(checkpoint)
    # model.eval()
    # #Wrap the ResNet model in the ModelPlus class
    # modelPlus = ModelPlus("vit2", model, device, imgSizeH=224, imgSizeW=224, batchSize=batchSize)
    # numClasses = 10
    # dir1 = "/data2/shh20007/haowenz/VisionTransformersRobustness/data/BiT-M-R101x3-Run1.tar"
    # model1 = BigTransferModels.KNOWN_MODELS["BiT-M-R101x3"](head_size=numClasses, zero_head=False)
    # #Get the checkpoint 
    # checkpoint = torch.load(dir1, map_location="cpu")
    # #Remove module so that it will load properly
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint["model"].items():
    #     name = k[7:] # remove `module.`
    #     new_state_dict[name] = v
    # #Load the dictionary
    # model1.load_state_dict(new_state_dict)
    # model1.eval()
    # modelPlus = ModelPlus("BiT-M-R101x3", model1, device, imgSizeH=160, imgSizeW=128, batchSize=batchSize)
     # Instantiate the model
    model_name = "WRN-70-16-swish" 
    backbone = wideresnetwithswish(model_name)
    modelD = torch.nn.Sequential(backbone)

    # Load the pre-trained weights
    # checkpoint_path = '/data2/shh20007/haowenz/DM-Improves-AT/trained_models/mymodel/weights-best.pt'
    checkpoint_path = '/data2/shh20007/haowenz/DM-Improves-AT/trained_models/mymodel/weights-best.pt'
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Remove `module.` prefix
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    modelD.load_state_dict(new_state_dict, strict=False)
    # Set model to evaluation mode
    modelD.eval()  
    device = torch.device("cuda")
    modelDPlus = ModelPlus("Difussion", modelD, device, imgSizeH=224, imgSizeW=224, batchSize=512)

    def print_memory_usage(model_name):
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MiB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # Convert to MiB
        print(f"{model_name}: Allocated: {allocated:.2f} MiB, Reserved: {reserved:.2f} MiB")

    Skeleton_Keys = 0
    fake = 0
    total_attacks = 0
    numClasses = 10
    # Get predictions for the entire dataset using predictD
    print("Memory usage before predictions:")
    print_memory_usage("Initial")

    def stagger_predictions(model, advLoader, numClasses, device):
        torch.cuda.empty_cache()
        preds = model.predictD(advLoader, numClasses).argmax(dim=1)
        torch.cuda.empty_cache()
        return preds

    preds_model_1 = stagger_predictions(modelPlusList[0], advLoader, numClasses, device)
    preds_model_2 = stagger_predictions(modelPlusList[1], advLoader, numClasses, device)
    preds_model_3 = stagger_predictions(modelPlusList[2], advLoader, numClasses, device)
    preds_model_4 = stagger_predictions(modelDPlus, advLoader, numClasses, device)


    # Iterate through the dataset and labels
    for idx, (inputs, labels) in enumerate(advLoader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Access predictions batch-wise
        batch_start = idx * advLoader.batch_size
        batch_end = batch_start + len(labels)
        preds_1 = preds_model_1[batch_start:batch_end]
        preds_2 = preds_model_2[batch_start:batch_end]
        preds_3 = preds_model_3[batch_start:batch_end]
        preds_4 = preds_model_4[batch_start:batch_end]

        # Analyze predictions
        for i in range(len(labels)):
            if preds_3[i] != labels[i] and preds_1[i] == labels[i] and preds_2[i] == labels[i]:
                Skeleton_Keys += 1
            if preds_4[i] != labels[i] and preds_1[i] == labels[i] and preds_2[i] == labels[i]:
                fake += 1
            total_attacks += 1

    # Calculate success rates
    attack_success_rate = (Skeleton_Keys / total_attacks) * 100
    yield_success_rate = (fake / total_attacks) * 100
    print(f"Attack Success Rate: {attack_success_rate:.2f}%")
    print(f"Yield_success_rate: {yield_success_rate:.2f}%")
    # for inputs, labels in advLoader:
    #     inputs, labels = inputs.to(device), labels.to(device)
        
    #     # Get predictions and find the class with the highest probability
    #     preds_model_1 = modelPlusList[0].predictT(inputs).argmax(dim=1)
    #     preds_model_2 = modelPlusList[1].predictT(inputs).argmax(dim=1)
    #     preds_model_3 = modelPlusList[2].predictT(inputs).argmax(dim=1)
        
    #     preds_model_4 = modelDPlus.predictT(inputs).argmax(dim=1)


    #     for i in range(len(labels)):
    #         if preds_model_3[i] != labels[i] and preds_model_1[i] == labels[i] and preds_model_2[i] == labels[i]:
    #             Skeleton_Keys += 1
    #         if preds_model_4[i] != labels[i] and preds_model_1[i] == labels[i] and preds_model_2[i] == labels[i]:
    #             fake += 1
    #         total_attacks += 1

    # attack_success_rate = (Skeleton_Keys / total_attacks) * 100
    # yield_success_rate = (fake / total_attacks) * 100
    # print(f"Attack Success Rate: {attack_success_rate:.2f}%")
    # print(f"Yield_success_rate: {yield_success_rate:.2f}%")
    

Skeleton_CIFAR10()




