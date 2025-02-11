'''
Author: fuchy@stu.pku.edu.cn
Description: Prepare data for traning and testing.
             *.mat is generated by dataPrepare from *.ply
             *.mat data structure cell{N*4} :
                N: point clouds number; N=1 in '*.mat'
                    {
                        [TreePoints*K*C] Octree data sequence generated from PQ (Quantized point cloud): N*K*C [n,7,6] array
                                         N[n treepoints]  K[7 ancestors]   C[oct code,level,octant,position(xyz)]
                4:      {Location} Original geometric coordinate P (n*3)
                        {qs,offset,Lmax,name} side information; Quantized point cloud PQ = (P-offset)/qs; The depth of PQ; The name of P (point cloud)
                    }
All rights reserved.
'''
import glob
from Preparedata.data import dataPrepare
from networkTool import CPrintl
def makedFile(dir):
    fileList = sorted(glob.glob(dir))
    return fileList
if __name__=="__main__":

######For MPEG,MVUB######    
    oriDir = './8iVLSF_910bit/*.ply'
    outDir = 'Data/Obj/train1/'
    ptNamePrefix = 'MPEG_' # 'MVUB_'

    printl = CPrintl('Preparedata/makedFileObj.log')
    makeFileList = makedFile(outDir+'*.mat')
    fileList = sorted(glob.glob(oriDir))
    for n,file in enumerate(fileList):
        fileName = file.split('/')[-1][:-4]
        dataName = outDir+ptNamePrefix+fileName+'.mat'
        if dataName in makeFileList:   
            print(dataName,'maked!')
            continue
        dataPrepare(file,saveMatDir=outDir,ptNamePrefix=ptNamePrefix,offset=0,rotation=False)
        # please set `rotation=True` in the `dataPrepare` function when processing MVUB data
        if n%10==0:
            printl(dataName)
