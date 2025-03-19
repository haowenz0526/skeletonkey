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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

    # modelR = resnet164(32, numClasses)
    # dirR = "/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/Resnet/checkpoint_R164_cifar10_k10tau0.pth.tar"
    # # dirR = "/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/Resnet/checkpoint_R164_cifar10_k10tau1.pth.tar"
    # # dirR = "/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/Resnet/checkpoint_R164_cifar10_k10tau2.pth.tar"
    # # dirR = "/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/Resnet/checkpoint_R164_cifar10_k10tau10.pth.tar"
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
    # modelPlusR = ModelPlus("ResNet-164", modelR, device, imgSizeH=32, imgSizeW=32, batchSize=10)
    # modelPlusList.append(modelPlusR)

    # Load ViT-L-16
    # config = CONFIGS["ViT-L_16"]
    # model = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses, vis = vis)
    # # dir = "/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/SavedModels/ViT-L_16_tau2.bin"
    # dir = '/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/SavedModels/ViT/ViT-L_16_tau2.bin'
    # dict = torch.load(dir)
    # model.load_state_dict(dict)
    # model.eval()
    # #Wrap the model in the ModelPlus class
    # modelPlusV = ModelPlus("ViT-L_16", model, device, imgSizeH=224, imgSizeW=224, batchSize=batchSize)
    # modelPlusList.append(modelPlusV)


    model = efficient_vmamba_tiny()
    model.num_classes = numClasses
    model.classifier.head = nn.Linear(384, numClasses)
    ckpt = torch.load('/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/EfficientVmamba/weights-best (2).pt', map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model = model.to(device)
    modelPlusV = ModelPlus("EfficientVmamba", model, device, imgSizeH=32, imgSizeW=32, batchSize=batchSize)
    modelPlusList.append(modelPlusV)

    
    model = mamba_vision_T(pretrained=False, num_classes=numClasses)
    # model = torch.nn.Linear(model.head.in_features, 10)
    model = torch.nn.Sequential(
        torch.nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
        model
    )
    ckpt = torch.load('/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/mambavision/weights-best (1).pt', map_location='cpu')
    # ckpt = torch.load('/data2/shh20007/haowenz/VisionTransformersRobustness/data/best_checkpoint.pth', map_location='cpu')
    # Apply key mapping to the checkpoint state dict
    state_dict = map_checkpoint_keys(ckpt['model_state_dict'])
    # state_dict = ckpt['model']
    # Load the state dict partially (ignore missing keys)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    modelPlusV = ModelPlus("MambaVision", model, device, imgSizeH=32, imgSizeW=32, batchSize=16)
    modelPlusList.append(modelPlusV)

     # vmamba_tiny
    model = vmamba_tiny_s1l8()  # Initialize model with correct number of classes
    model.num_classes = numClasses
    model.classifier.head = nn.Linear(768, numClasses)
    ckpt = torch.load('/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/model_/VMamba/weights-best.pt', map_location='cpu')
    # Check the keys in the checkpoint
    print("Checkpoint keys:", ckpt.keys())  
    # Load checkpoint using the correct key
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    # #Wrap the ResNet model in the ModelPlus class
    modelPlusM = ModelPlus("Vmamba", model, device, imgSizeH=32, imgSizeW=32, batchSize=64)
    modelPlusList.append(modelPlusM)

    

    


    # Instantiate the Diffusion model
    
    # model = create_model("wrn-28-10-swish", False, {'data': 'cifar10', 'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'), 'num_classes': 10, 'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.201]}, "cuda")
    # checkpoint = torch.load('/data2/shh20007/haowenz/Game-Theoretic-Mixed-Experts/SavedModels/diffusion/cifar10_linf_wrn28-10.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()
    # del checkpoint
    # size = (32,32) #input size
    # bs = 32 #batch size
   
    # modelDPlus = ModelPlus("Difussion", model, device, imgSizeH=32, imgSizeW=32, batchSize=32)
    # modelPlusList.append(modelDPlus)
            
    
    

    

    # # Load the BiT-M-R101x3
    # dirB = "/data2/shh20007/haowenz/VisionTransformersRobustness/data/BiT-M-R101x3-Run1.tar"
    # modelB = BigTransferModels.KNOWN_MODELS["BiT-M-R101x3"](head_size=numClasses, zero_head=False)
    # #Get the checkpoint 
    # checkpoint = torch.load(dirB, map_location="cpu")
    # #Remove module so that it will load properly
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint["model"].items():
    #     name = k[7:] # remove `module.`
    #     new_state_dict[name] = v
    # #Load the dictionary
    # modelB.load_state_dict(new_state_dict)
    
    # modelB.eval()
    # #Wrap the model in the ModelPlus class
    # #Here we hard code the Big Transfer Model Plus class input size to 160x128 (what it was trained on)
    # modelBig101Plus = ModelPlus("BiT-M-R101x3", modelB, device, imgSizeH=160, imgSizeW=128, batchSize=batchSize)
    # modelPlusList.append(modelBig101Plus)
    
    # # Load Vim-t
    # vim_dir = "/data2/shh20007/haowenz/VisionTransformersRobustness/data/vim_t_midclstok_76p1acc.pth"
    # vim_dict = torch.load(vim_dir)
    # model_name = "vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
    # # model_name = "MambaVision-S-1K"
    # # model_vim_t = create_model(model_name, pretrain=True)
    # modelvim = ModelMamba.vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2(pretrained=True)
    # state_dict = OrderedDict()
    # for k, v in checkpoint["model"].items():
    #     name = k[7:] if k.startswith("module.") else k  # 移除 "module."
    #     state_dict[name] = v
    # model_vim_t.load_state_dict(state_dict)
    # model_vim_t.eval()
    # modelPlusVimT = ModelPlus("Vim", model_vim_t, device, imgSizeH=224, imgSizeW=224, batchSize=batchSize)
    # modelPlusList.append(modelPlusVimT)


    
    

    #Now time to build the defense 
    defense = ShuffleDefense.ShuffleDefense(modelPlusList, numClasses)
    return valLoader, defense

#Run the Self-Attention Gradient Attack (SAGA) on ViT-L and BiT-M-R101x3
def Skeleton_CIFAR10():
    #Set up the parameters for the attack 
    attackSampleNum = 500
    numClasses = 10
    coefficientArray = torch.zeros(3)
    secondcoeff = 2.0000e-04
    coefficientArray[1] = - secondcoeff
    coefficientArray[0] = 1 - secondcoeff
    coefficientArray[2] = - secondcoeff
    
    print("Coeff Array:")
    print(coefficientArray)
    device = torch.device("cuda")
    epsMax = 0.031
    clipMin = 0.0
    clipMax = 1.0
    numSteps = 10
    epsStep = 0.005
    #Load the models and the dataset
    #Note it is important to set vis to true so the transformer's model output returns the attention weights 
    valLoader, defense = LoadShuffleDefenseAndCIFAR10(vis=True)
    modelPlusList = defense.modelPlusList
    
    #Note that the batch size will effect how the gradient is computed in PyTorch
    #Here we use batch size 8 for ViT-L and batch size 2 for BiT-M. Other batch sizes are possible but they will not generate the same result
    # modelPlusList[0].batchSize = 2
    # modelPlusList[1].batchSize = 8

    alphaLearningRate = 10000#0#100000
    # alphaLearningRate = 100000#0#100000
    # alphaLearningRate = 10000#0#100000
    fittingFactor = 50.0
    #Get the clean examples 
    cleanLoader = AttackWrappersSAGA.GetFirstCorrectlyOverlappingSamplesBalanced(device, 3000, numClasses, valLoader, modelPlusList)
    # xclean, yclean = DMP.DataLoaderToTensor(cleanLoader)
    
    # balancedLoader = AttackWrappersSAGA.GetFirstOverlappingSamplesBalanced(device, 3000, numClasses, valLoader, modelPlusList)
    # advloader = AutoAttackNativePytorch("cuda", cleanLoader, modelPlusList[0].model, .031, .005, 40, 0, 1, False)
    # torch.cuda.empty_cache()
    # advLoader = AttackWrappersSAGA.SkeletonKeyAuto(device, epsMax, numSteps, modelPlusList, cleanLoader, clipMin, clipMax,
    #                                      alphaLearningRate, fittingFactor)
    advLoader = AttackWrappersSAGA.SelfAttentionGradientAttack(device, epsMax, numSteps, modelPlusList, coefficientArray, cleanLoader, clipMin, clipMax)
    # xadv, yclean = DMP.DataLoaderToTensor(advLoader)
    # torch.save({"x": xadv, "y": yclean}, "Skeleton_key")

    # accArrayProtoSAGA = torch.zeros(attackSampleNum).to(device) # Create an array with one entry for ever sample in the dataset
    #Do the attack
    # advLoader = AttackWrappersSAGA.SelfAttentionGradientAttack(device, epsMax, numSteps, modelPlusList, coefficientArray, balancedLoader, clipMin, clipMax)


    # for i in range(0, len(modelPlusList)):
    #     model = modelPlusList[i].model.to(device)  # Ensure model is on correct device
    #     model.eval()  # Ensure model is in eval mode
    #     acc = modelPlusList[i].validateD(advloader)
    #     print(f"{modelPlusList[i].modelName} Robust Acc:", acc)
    # for i in range(0, len(modelPlusList)):
    #     accArray = modelPlusList[i].validateDA(advLoader)
    #     accArray = accArray.to(device)
    #     accArrayProtoSAGA = accArrayProtoSAGA + accArray
    #     print("ProtoSAGA Acc " + modelPlusList[i].modelName + ":", accArray.sum()/attackSampleNum)
    # MV_ProtoSAGA_acc = (accArrayProtoSAGA>=1).sum() / attackSampleNum
    # print('MV_ProtoSAGA_acc: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    # ALL_MV_ProtoSAGA_acc = (accArrayProtoSAGA==2).sum() / attackSampleNum
    # print('ALL_MV_ProtoSAGA_acc: ', ALL_MV_ProtoSAGA_acc.data.cpu().numpy())
    # MV_ProtoSAGA_acc = (accArrayProtoSAGA==0).sum() / attackSampleNum
    # print('ProtoSAGA attack successful rate: ', MV_ProtoSAGA_acc.data.cpu().numpy())
   


    # path = '/data2/shh20007/haowenz/VisionTransformersRobustness/skeleton/mam-vit-res/vit/Skeleton_key'
    # data = torch.load(path)
    # xadv = data["x"]
    # yclean = data["y"]
    # advLoader = DMP.TensorToDataLoader(xadv, yclean, batchSize=8)
    
    Skeleton_Keys = 0
    fake = 0
    single = 0
    total_attacks = 0
    numClasses = 10
 

    def stagger_predictions(model, advLoader, numClasses, device):
        torch.cuda.empty_cache()
        preds = model.predictD(advLoader, numClasses).argmax(dim=1)
        torch.cuda.empty_cache()
        return preds

    preds_model_1 = stagger_predictions(modelPlusList[0], advLoader, numClasses, device)
    preds_model_2 = stagger_predictions(modelPlusList[1], advLoader, numClasses, device)
    preds_model_3 = stagger_predictions(modelPlusList[2], advLoader, numClasses, device)
    # preds_model_4 = stagger_predictions(modelDPlus, advLoader, numClasses, device)

    for i in range(0, len(modelPlusList)):
        acc = modelPlusList[i].validateD(advLoader)
        print(modelPlusList[i].modelName+" Robust Acc:", acc)


    # Initialize lists to store the inputs and labels
    stored_inputs = []
    stored_labels = []

    # Iterate through the dataset and labels
    for idx, (inputs, labels) in enumerate(advLoader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Access predictions batch-wise
        batch_start = idx * advLoader.batch_size
        batch_end = batch_start + len(labels)
        preds_1 = preds_model_1[batch_start:batch_end]
        preds_2 = preds_model_2[batch_start:batch_end]
        preds_3 = preds_model_3[batch_start:batch_end]
        # preds_4 = preds_model_4[batch_start:batch_end]
        

        # Analyze predictions
        for i in range(len(labels)):
            if preds_1[i] != labels[i] and preds_2[i] == labels[i] and preds_3[i] == labels[i]:
                Skeleton_Keys += 1
                # Store the input and label
                stored_inputs.append(inputs[i].cpu())  # Move to CPU for storage
                stored_labels.append(labels[i].cpu())  # Move to CPU for storage

                # Stop the process if 600 pieces are stored
                if len(stored_inputs) >= 600:
                    break
            if preds_1[i] == labels[i] and preds_2[i] != labels[i] and preds_3[i] != labels[i]:
                fake += 1
            if preds_1[i] == labels[i]:
                single += 1
            total_attacks += 1

        # Break outer loop if 600 pieces are stored
        if len(stored_inputs) >= 600:
            break

    # Convert lists to tensors
    stored_inputs = torch.stack(stored_inputs)  # Shape: [600, C, H, W]
    stored_labels = torch.tensor(stored_labels)  # Shape: [600]

    # Save the stored inputs and labels
    torch.save({"x": stored_inputs, "y": stored_labels}, "Skeleton_key_600_2")

    # End the process
    print("Process ended after storing 600 pieces of data.")
    exit()  # Terminate the script

Skeleton_CIFAR10()




