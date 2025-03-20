import torch
import numpy
import ShuffleDefense
from AttackWrappersSAGA import SelfAttentionGradientAttackProtoAuto
from ModelPlus import ModelPlus
import DataManagerPytorch as DMP
# import AttackWrappersRayS
# import AttackWrappersAdaptiveBlackBox
import AttackWrappersSAGA
from TransformerModels import VisionTransformer, CONFIGS
# import spikingjelly_surrogate_extension as sse


import BigTransferModels
from ResNetPytorch import resnet56, resnet164
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torch.nn.functional as f
# from wideresnet import wideresnet
from wideresnetwithswish import wideresnetwithswish
import os
import torch.nn as nn
# from AttackWrappersWhiteBoxJelly import PGDNativePytorch as PGDJelly
from AttackWrappersProtoSAGA import SelfAttentionGradientAttack_EOT, MIM_EOT_Wrapper, AutoAttackNativePytorch, MIMNativePytorch_cnn, SelfAttentionGradientAttackProto_Old
from spikingjelly.clock_driven import neuron, surrogate


import sys
sys.path.append('/data2/shh20007/haowenz/DM-Improves-AT')
sys.path.append('/data2/shh20007/haowenz/VMamba/classification/models')
sys.path.append('/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/Defenses/SNN')
sys.path.append('/data2/shh20007/haowenz/EfficientVMamba/classification')
sys.path.append('/data2/shh20007/haowenz/EfficientVMamba/classification/models')
sys.path.append('/data2/shh20007/haowenz/MambaVision/mambavision/models')
from mamba_vision import mamba_vision_T, _load_checkpoint
import SpikingJelly_sew_resnet as sew_resnet
from SpikingJelly_sew_resnet import sew_resnet18
from vmamba_efficient import efficient_vmamba_tiny 
from core.models import create_model




# Import the create_model function
from vmamba import vmamba_tiny_s1l8


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

#Load the CIFAR-10 Spiking Jelly model 
def LoadCIFAR10SNNResNetBackProp(modelDir, sg = 'ATan'):
    #Model parameters
    sg = sg
    argsNeuron = 'MultiStepParametricLIFNode'
    arch = 'sew_resnet18'
    num_classes = 10
    timeStep = 4
    surrogate_gradient = {
        'ATan': surrogate.ATan(),
        'Sigmoid': surrogate.Sigmoid(),
        'PiecewiseLeakyReLU': surrogate.PiecewiseLeakyReLU(),
        'S2NN': surrogate.S2NN(),
        'QPseudoSpike': surrogate.QPseudoSpike(),
        'PiecewiseQuadratic': surrogate.PiecewiseQuadratic(),
        'PiecewiseExp': surrogate.PiecewiseExp(),
        # 'Erfc': sse.Erfc(),
        # 'FastSigmoid': sse.FastSigmoid(),
        # 'STBPActFun': sse.STBPActFun()
                    }
    sg_type = surrogate_gradient[sg]
    neuron_dict = {
        'MultiStepIFNode'               : neuron.MultiStepIFNode,
        'MultiStepParametricLIFNode'    : neuron.MultiStepParametricLIFNode,
        'MultiStepEIFNode'              : neuron.MultiStepEIFNode,
        'MultiStepLIFNode'              : neuron.MultiStepLIFNode,
    }
    neuron_type = neuron_dict[argsNeuron]
    model_arch_dict = {
                    'sew_resnet18'       : sew_resnet.multi_step_sew_resnet18, 
                    'sew_resnet34'       : sew_resnet.multi_step_sew_resnet34, 
                    'sew_resnet50'       : sew_resnet.multi_step_sew_resnet50,
                    # 'spiking_resnet18'   : spiking_resnet.multi_step_spiking_resnet18, 
                    # 'spiking_resnet34'   : spiking_resnet.multi_step_spiking_resnet34, 
                    # 'spiking_resnet50'   : spiking_resnet.multi_step_spiking_resnet50,
    }
    model_type = model_arch_dict[arch]
    model = model_type(T=timeStep, num_classes=num_classes, cnf='ADD', multi_step_neuron=neuron_type, surrogate_function=sg_type)
    #Load the model from the baseDir
    checkpoint = torch.load(modelDir)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model

def map_checkpoint_keys(checkpoint_state_dict):
    """Map checkpoint keys to match the model's expected keys."""
    key_mapping = {
        # Example mappings (adjust based on your specific mismatch)
        "patch_embed.conv_down.weight": "patch_embed.conv_down.0.weight",
        "patch_embed.conv_down.bias": "patch_embed.conv_down.0.bias",
        "levels.blocks.0.convweight": "levels.0.blocks.0.conv1.weight",
        "levels.blocks.0.convbias": "levels.0.blocks.0.conv1.bias",
        # Add more mappings as needed based on the error message
    }
    
    new_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        # Apply the mapping if the key exists in the mapping
        new_key = key_mapping.get(key, key)
        new_state_dict[new_key] = value
    return new_state_dict

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


    # vmamba_tiny
    # model = vmamba_tiny_s1l8()  # Initialize model with correct number of classes
    # model.num_classes = numClasses
    # model.classifier.head = nn.Linear(768, numClasses)
    # ckpt = torch.load('/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/SavedModels/Mamba/weights-best.pt', map_location='cpu')
    # # Check the keys in the checkpoint
    # print("Checkpoint keys:", ckpt.keys())  
    # # Load checkpoint using the correct key
    # model.load_state_dict(ckpt['model_state_dict'])
    # model.eval()
    # # #Wrap the ResNet model in the ModelPlus class
    # modelPlusM = ModelPlus("Vmamba", model, device, imgSizeH=32, imgSizeW=32, batchSize=64)
    # modelPlusList.append(modelPlusM)

    model = efficient_vmamba_tiny()
    model.num_classes = numClasses
    model.classifier.head = nn.Linear(384, numClasses)
    ckpt = torch.load('/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/vmamba_eff/weights-best (2).pt', map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model = model.to(device)
    modelPlusV = ModelPlus("EfficientVmamba", model, device, imgSizeH=32, imgSizeW=32, batchSize=batchSize)
    modelPlusList.append(modelPlusV)

    # model = mamba_vision_T(pretrained=False, num_classes=numClasses)
    # # model = torch.nn.Linear(model.head.in_features, 10)
    # model = torch.nn.Sequential(
    #     torch.nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
    #     model
    # )
    # ckpt = torch.load('/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/mamba_vision/weights-best (1).pt', map_location='cpu')
    # # ckpt = torch.load('/data2/shh20007/haowenz/VisionTransformersRobustness/data/best_checkpoint.pth', map_location='cpu')
    # # Apply key mapping to the checkpoint state dict
    # state_dict = map_checkpoint_keys(ckpt['model_state_dict'])
    # # state_dict = ckpt['model']
    # # Load the state dict partially (ignore missing keys)
    # model.load_state_dict(state_dict, strict=False)
    # model.eval()
    # model = model.to(device)
    # modelPlusV = ModelPlus("MambaVision", model, device, imgSizeH=32, imgSizeW=32, batchSize=16)
    # modelPlusList.append(modelPlusV)
     
    # Resnet
    # modelR = resnet164(32, numClasses)
    # # dirR = "/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/Resnet/checkpoint_R164_cifar10_k10tau0.pth.tar"
    # # dirR = "/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/Resnet/checkpoint_R164_cifar10_k10tau1.pth.tar"
    # # dirR = "/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/Resnet/checkpoint_R164_cifar10_k10tau2.pth.tar"
    # dirR = "/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/Resnet/checkpoint_R164_cifar10_k10tau10_mom.pth.tar"
    # # dirR = "/data2/shh20007/haowenz/VisionTransformersRobustness/data/ModelResNet56-Run0.th"
    # checkpointR = torch.load(dirR)
    # new_state_dict = OrderedDict()
    # state_dict = checkpointR['state_dict']
    # for k, v in state_dict.items():
    #     name = k.replace('module.', '')  # Remove 'module.' prefix
    #     new_state_dict[name] = v
    # modelR.load_state_dict(new_state_dict)
    # modelR.eval()
    # #Wrap the ResNet model in the ModelPlus class
    # modelPlusR = ModelPlus("ResNet-164", modelR, device, imgSizeH=32, imgSizeW=32, batchSize=64)
    # modelPlusList.append(modelPlusR)
    

    # Load ViT-L-16
    # config = CONFIGS["ViT-L_16"]
    # model = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses, vis = vis)
    # # dir = "/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/SavedModels/ViT-L_16_tau2.bin"
    # # dir = '/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/ViT/ViT-L_16,cifar10,run1_15K_k10_tau0.bin'
    # # dir = '/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/ViT/ViT-L_16,cifar10,run1_15K_k10_tau1.bin'
    # # dir = '/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/ViT/ViT-L_16,cifar10,run1_15K_k10_tau2.bin'
    # dir = '/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/ViT/ViT-L_16,cifar10,run1_15K_k10_tau10.bin'
    # dict = torch.load(dir)
    # model.load_state_dict(dict)
    # model.eval()
    # #Wrap the model in the ModelPlus class
    # modelPlusV = ModelPlus("ViT-L_16", model, device, imgSizeH=224, imgSizeW=224, batchSize=batchSize)
    # modelPlusList.append(modelPlusV)

  
    # Diffusion
    # model = create_model("wrn-28-10-swish", False, {'data': 'cifar10', 'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'), 'num_classes': 10, 'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.201]}, "cuda")
    # checkpoint = torch.load('/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/diffusion/cifar10_linf_wrn28-10.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()
    # del checkpoint
    # size = (32,32) #input size
    # bs = 32 #batch size
    # modelDPlus = ModelPlus("Difussion", model, device, imgSizeH=224, imgSizeW=224, batchSize=32)
    # modelPlusList.append(modelDPlus)

    

    #Now time to build the defense 
    defense = ShuffleDefense.ShuffleDefense(modelPlusList, numClasses)
    return valLoader, defense

#Run the Self-Attention Gradient Attack (SAGA) on ViT-L and BiT-M-R101x3
def APGD():
    #Set up the parameters for the attack 
    numClasses = 10
    device = torch.device("cuda")
    #Load the models and the dataset
    #Note it is important to set vis to true so the transformer's model output returns the attention weights 
    valLoader, defense = LoadShuffleDefenseAndCIFAR10(vis=True)
    modelPlusList = defense.modelPlusList
    
    #Get the clean examples 
    cleanLoader = AttackWrappersSAGA.GetFirstCorrectlyOverlappingSamplesBalanced(device, 1000, numClasses, valLoader, modelPlusList)
    # balancedLoader = AttackWrappersSAGA.GetFirstOverlappingSamplesBalanced(device, attackSampleNum, numClasses, valLoader, modelPlusList)


    advloader = AutoAttackNativePytorch("cuda", cleanLoader, modelPlusList[0].model, .031, .005, 40, 0, 1, False)


    for i in range(0, len(modelPlusList)):
        model = modelPlusList[i].model.to(device)  # Ensure model is on correct device
        model.eval()  # Ensure model is in eval mode
        acc = modelPlusList[i].validateD(advloader)
        print(f"{modelPlusList[i].modelName} Robust Acc:", acc)

    
# LoadShuffleDefenseAndCIFAR10()
APGD()




