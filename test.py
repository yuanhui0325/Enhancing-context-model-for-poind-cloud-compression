'''
Author: fuchy@stu.pku.edu.cn
Description: this file encodes point cloud
FilePath: /compression/encoder.py
All rights reserved.
'''
from numpy import mod
from Preparedata.data import dataPrepare
from encoderTool import main
from networkTool import reload,CPrintl,expName,device
from octAttention import model
import glob,datetime,os
import pt as pointCloud
import torch
import numpy as np
############## warning ###############
## decoder.py relys on this model here
## do not move this lines to somewhere else
model = model.to(device)
saveDic = reload(None,'modelsave/res_loss.pth')
model.load_state_dict(saveDic['encoder'])
bpp = torch.zeros([5])
v = torch.zeros([5])
###########Objct##############
oriDir = './../../../muvb/muvb/phil9/ply/*.ply'
np.set_printoptions(threshold=np.inf)
if __name__=="__main__":
    printl = CPrintl(expName+'/encoderPLY.txt')
    printl('_'*50,'OctAttention V0.4','_'*50)
    printl(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S'))
    printl('load checkpoint', saveDic['path'])
    fileList = sorted(glob.glob(oriDir))
    s = 0    
    for oriFile in fileList:
        s =s+1
        printl(oriFile)
        if (os.path.getsize(oriFile)>300*(1024**2)):#300M
            printl('too large!')
            continue
        ptName = os.path.splitext(os.path.basename(oriFile))[0] 
        for qs in [1]:
            ptNamePrefix = ptName
            matFile,DQpt,refPt = dataPrepare(oriFile,saveMatDir='./Data/testPly',qs=qs,ptNamePrefix='',rotation=True)
            # please set `rotation=True` in the `dataPrepare` function when processing MVUB data
            rate=main(matFile,model,actualcode=True,printl =printl) # actualcode=False: bin file will not be generated
            print('_'*50,'pc_error','_'*50)
            #pointCloud.pcerror(refPt,DQpt,None,'-r 1023',None).wait()
            bpp[qs-1] += rate
        for i in range(0,5):
            v[i] = torch.div(bpp[i], s)
        print(v)
#5.4.0-144-generic
#2.04-1ubuntu26.12

