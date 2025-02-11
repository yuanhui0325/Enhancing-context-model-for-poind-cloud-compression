'''
Author: fuchy@stu.pku.edu.cn
LastEditors: Please set LastEditors
Description: Network parameters and helper functions
FilePath: /compression/networkTool.py
'''
import os,random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
# torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network parameters
bptt = 1024 # Context window length
expName = './Exp/Obj'
DataRoot = './Data/Obj'

checkpointPath = expName+'/checkpoint3'
levelNumK = 4

trainDataRoot = DataRoot+"/train/*.mat" # DON'T FORGET RUN ImageFolder.calcdataLenPerFile() FIRST
expComment = 'OctAttention, trained on MPEG 8i,MVUB 1~10 level. 2021/12. All rights reserved.'
occu1 = [0, 1, 3, 7, 15, 31, 63, 127]
occu2 = [2, 4, 5, 8, 9, 11, 16, 17, 19, 23, 32, 33, 35, 39, 47, 64, 65, 67, 71, 79, 95, 128, 129, 131, 135, 143, 159, 191]
occu3 = [6, 10, 12, 13, 18, 20, 21, 24, 25, 27, 34, 36, 37, 40, 41, 43, 48, 49, 51, 55, 66, 68, 69, 72, 73, 75, 80, 81, 83, 87, 96, 97, 99, 103, 111, 130, 132, 133, 136, 137, 139, 144, 145, 147, 151, 160, 161, 163, 167, 175, 192, 193, 195, 199, 207, 223]
occu4 = [14, 22, 26, 28, 29, 38, 42, 44, 45, 50, 52, 53, 56, 57, 59, 70, 74, 76, 77, 82, 84, 85, 88, 89, 91, 98, 100, 101, 104, 105, 107, 112, 113, 115, 119, 134, 138, 140, 141, 146, 148, 149, 152, 153, 155, 162, 164, 165, 168, 169, 171, 176, 177, 179, 183, 194, 196, 197, 200, 201, 203, 208, 209, 211, 215, 224, 225, 227, 231, 239]
occu5 = [30, 46, 54, 58, 60, 61, 78, 86, 90, 92, 93, 102, 106, 108, 109, 114, 116, 117, 120, 121, 123, 142, 150, 154, 156, 157, 166, 170, 172, 173, 178, 180, 181, 184, 185, 187, 198, 202, 204, 205, 210, 212, 213, 216, 217, 219, 226, 228, 229, 232, 233, 235, 240, 241, 243, 247]
occu6 = [62, 94, 110, 118, 122, 124, 125, 158, 174, 182, 186, 188, 189, 206, 214, 218, 220, 221, 230, 234, 236, 237, 242, 244, 245, 248, 249, 251]
occu7 = [126, 190, 222, 238, 246, 250, 252, 253]
occu8 = [254]
MAX_OCTREE_LEVEL = 12
# Random seed
seed=2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
os.environ["H5PY_DEFAULT_READONLY"] = "1"
# Tool functions
def save(index, saveDict,modelDir='checkpoint',pthType='epoch'):
    if os.path.dirname(modelDir)!='' and not os.path.exists(os.path.dirname(modelDir)):
        os.makedirs(os.path.dirname(modelDir))
    torch.save(saveDict, modelDir+'/encoder_{}_{:08d}.pth'.format(pthType, index))
        
def reload(checkpoint,modelDir='checkpoint',pthType='epoch',print=print,multiGPU=False):
    try:
        if checkpoint is not None:
            saveDict = torch.load(modelDir+'/encoder_{}_{:08d}.pth'.format(pthType, checkpoint),map_location=device)
            pth = modelDir+'/encoder_{}_{:08d}.pth'.format(pthType, checkpoint)
        if checkpoint is None:
            saveDict = torch.load(modelDir,map_location=device)
            pth = modelDir
        saveDict['path'] = pth
        # print('load: ',pth)
        if multiGPU:
            from collections import OrderedDict
            state_dict = OrderedDict()
            new_state_dict = OrderedDict()
            for k, v in saveDict['encoder'].items():
                name = k[7:]  # remove `module.`
                state_dict[name] = v
            saveDict['encoder'] = state_dict
        return saveDict
    except Exception as e:
        print('**warning**',e,' start from initial model')
        # saveDict['path'] = e
    return None

class CPrintl():
    def __init__(self,logName) -> None:
        self.log_file = logName
        if os.path.dirname(logName)!='' and not os.path.exists(os.path.dirname(logName)):
            os.makedirs(os.path.dirname(logName))
    def __call__(self, *args):
        print(*args)
        print(*args, file=open(self.log_file, 'a'))

def model_structure(model,print=print):
    print('-'*120)
    print('|'+' '*30+'weight name'+' '*31+'|' \
            +' '*10+'weight shape'+' '*10+'|' \
            +' '*3+'number'+' '*3+'|')
    print('-'*120)
    num_para = 0
    for _, (key, w_variable) in enumerate(model.named_parameters()):
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para

    
        print('| {:70s} | {:30s} | {:10d} |'.format(key, str(w_variable.shape), each_para))
    print('-'*120)
    print('The total number of parameters: ' + str(num_para))
    print('-'*120)
