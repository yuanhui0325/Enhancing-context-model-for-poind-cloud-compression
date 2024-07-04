'''
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-20 08:06:11
LastEditTime: 2021-09-20 23:53:24
LastEditors: fcy
Description: the training file
             see networkTool.py to set up the parameters
             will generate training log file loss.log and checkpoint in folder 'expName'
FilePath: /compression/octAttention.py
All rights reserved.
'''
from networkTool import *
import math
import torch
import torch.nn as nn
import os
import datetime

from torch.utils.tensorboard import SummaryWriter
from attentionModel import TransformerLayer,TransformerModule
from collections import Counter


##########################
ntoken2 = 8
ntokens = 255 # the size of vocabulary
ninp = 4*(128+4+6)
ninp2 = 4*(128+4+6)+8  # embedding dimension
inp = 8*(128+4+6)

nhid = 300 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 3 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4 # the number of heads in the multiheadattention models
dropout = 0 # the dropout value
batchSize = 32

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()   
        
    def forward(self, x, y):  
        loss = 0.000000  
        softmax = nn.Softmax(dim=1)
        n = len(x)
        x = softmax(x)
        #print(y)     
        x = -torch.log2(x)
        entropy = torch.mv(x,y)
        loss = sum(entropy)/len(entropy)
        #print(loss)
        
        

        return loss

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.pos_encoder = PositionalEncoding(ninp, dropout)

        encoder_layers = TransformerLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerModule(encoder_layers, nlayers)

        self.encoder = nn.Embedding(ntoken, 128)
        self.encoder1 = nn.Embedding(MAX_OCTREE_LEVEL+1, 6)
        self.encoder2 = nn.Embedding(9, 4)

        self.ninp = ninp
        self.act = nn.ReLU()
        self.decoder0 = nn.Linear(inp, ninp)
        self.decoder1 = nn.Linear(ninp2, ntoken)
        self.decoder2 = nn.Linear(inp, ninp)
        self.decoder3 = nn.Linear(ninp, 8)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data = nn.init.xavier_normal_(self.encoder.weight.data )
        self.decoder0.bias.data.zero_()
        self.decoder0.weight.data= nn.init.xavier_normal_(self.decoder0.weight.data )
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data = nn.init.xavier_normal_(self.decoder1.weight.data )
        self.decoder2.bias.data.zero_()
        self.decoder2.weight.data = nn.init.xavier_normal_(self.decoder2.weight.data )
        self.decoder3.bias.data.zero_()
        self.decoder3.weight.data = nn.init.xavier_normal_(self.decoder3.weight.data )
    def forward(self, src, src_mask, res, dataFeat):
        bpttt = src.shape[0]
        batch = src.shape[1]
        if bpttt!=bptt:
           feature = res[0:1023,:,:].clone().detach().to(device)
        else:
           feature = res.clone().detach().to(device)
        oct = src[:,:,:,0] #oct[bptt,batchsize,FeatDim(levels)] [0~254]
        level = src[:,:,:,1]  # [0~12] 0 for padding data
        octant = src[:,:,:,2] # [0~8] 0 for padding data

        # assert oct.min()>=0 and oct.max()<255
        # assert level.min()>=0 and level.max()<=12
        # assert octant.min()>=0 and octant.max()<=8
        
        level -= torch.clip(level[:,:,-1:] - 10,0,None)# the max level in traning dataset is 10        
        torch.clip_(level,0,MAX_OCTREE_LEVEL) 
        aOct = self.encoder(oct.long()) #a[bptt,batchsize,FeatDim(levels),EmbeddingDim]
        aLevel = self.encoder1(level.long())
        aOctant = self.encoder2(octant.long())

        a = torch.cat((aOct,aLevel,aOctant),3)

        a = a.reshape((bpttt,batch,-1)) 
        
        # src = self.ancestor_attention(a)
        src = a.reshape((bpttt,a.shape[1],self.ninp))* math.sqrt(self.ninp)

        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        features = output
        
        feature = output-feature
        output = torch.cat((output,feature),dim=2)
        #output = self.decoder1(self.act(self.decoder0(output)))
        output1 = self.act(self.decoder0(output))
        output2 = self.decoder3(self.act(self.decoder2(output)))
        output1 = torch.cat((output1,self.act(output2)),dim=2)
        output1 = self.decoder1(output1)
        return output1,output2 ,features


######################################################################
# ``PositionalEncoding`` module 
#

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

######################################################################
# Functions to generate input and target sequence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

def get_batch(source, i):
    seq_len = min(bptt, len(source)  - i-1)
    #zero=torch.zeros([1,32,4,6],dtype=torch.float).to(device)
    #source = torch.cat((source,zero),0)
    data = source[i:i+seq_len].clone()
    target = source[i+1:i+1+seq_len,:,-1,0].reshape(-1)
    data[:,:,0:-1,:] = source[i+1:i+seq_len+1,:,0:-1,:] # this moves the feat(octant,level) of current node to lastrow,        
    data[:,:,-1,1:3] = source[i+1:i+seq_len+1,:,-1,1:3]# which will be used as known feat
    return data[:,:,-levelNumK:,:], (target).long(),[]

def counter(arr):

    return Counter(arr)

def get_target(target):

    arr = target.to(device)+1
    x = torch.zeros([arr.shape[0],8]).to(device)

    for i in range(0,8):
        x[:,i] = arr% 2

        arr = torch.div(arr, 2, rounding_mode='trunc')

    return(x)

def get_num(target2):
    
    for i in occu1:
        target2[i,0]=1
    for i in  occu2:
        target2[i,1]=1

    for i in (occu3):
        target2[i,2]=1
    for i in (occu4):
        target2[i,3]=1
    for i in (occu5):
        target2[i,4]=1
    for i in (occu6):
        target2[i,5]=1
    for i in (occu7):
        target2[i,6]=1
    for i in (occu8):
        target2[i,7]=1
    return(target2)
######################################################################
# Run the model
# -------------
#
model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout).to(device)
if __name__=="__main__":
    import dataset
    import torch.utils.data as data
    import time
    import os
    freeze_layers = ("decoder0", "decoder1")
    for name, param in model.named_parameters():
        if name.split(".")[0] in freeze_layers:
            param.requires_grad = False


    #lable = torch.zeros(255,8).to(device)
    
    #lable = get_num(lable)
    epochs = 20 # The number of epochs
    best_model = None
    batch_size = 128
    TreePoint = bptt*16
    train_set = dataset.DataFolder(root=trainDataRoot, TreePoint=TreePoint,transform=None,dataLenPerFile= 297319.05) # you should run 'dataLenPerFile' in dataset.py to get this num (17456051.4)
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=4,drop_last=True) # will load TreePoint*batch_size at one time  
    
    # loger
    if not os.path.exists(checkpointPath):
        os.makedirs(checkpointPath)
    printl = CPrintl(expName+'/loss.log')
    writer = SummaryWriter('./log/'+expName)
    printl(datetime.datetime.now().strftime('\r\n%Y-%m-%d:%H:%M:%S'))
    # model_structure(model,printl)
    printl(expComment+' Pid: '+str(os.getpid()))
    log_interval = int(batch_size*TreePoint/batchSize/bptt)
    
    # learning
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.BCEWithLogitsLoss()
    lr = 1e-3 # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    best_val_loss = float("inf")
    idloss = 0

    # reload
    saveDic = None
    #saveDic = reload(2000120,checkpointPath)
    if saveDic:
        scheduler.last_epoch = saveDic['epoch'] - 1
        idloss = saveDic['idloss']
        best_val_loss = saveDic['best_val_loss']
        model.load_state_dict(saveDic['encoder'])
        
    def train(epoch):
        global idloss,best_val_loss
        model.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        total_loss_list = torch.zeros((1,7))
        
        for Batch, d in enumerate(train_loader): # there are two 'BATCH', 'Batch' includes batch_size*TreePoint/batchSize/bptt 'batch'es.
            batch = 0
            features = np.zeros([bptt, 32 ,552])

            features = torch.tensor(features,dtype=torch.float).to(device)

            train_data = d[0].reshape((batchSize,-1,4,6)).to(device).permute(1,0,2,3)   #shape [TreePoint*batch_size(data)/batch_size,batch_size,7,6]
            src_mask = model.generate_square_subsequent_mask(bptt).to(device)
            for index, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
                data, targets,dataFeat = get_batch(train_data, i)#data [35,20]

                target1 = get_target(targets) 

                optimizer.zero_grad()
                if data.size(0) != bptt:
                    src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
                output1, output2,features = model(data, src_mask, features, dataFeat)
                #output2 = torch.matmul(output,lable)
                #pading = torch.zeros(data.size(0),batchSize,247).to(device)              
                #output2 = torch.cat((output2,pading),dim=2)

                #loss = criterion(output1.view(-1, ntokens), targets)
                loss = criterion2(output2.view(-1, 8), target1)
                #loss = loss1+loss2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()
                batch = batch+1
                if batch % log_interval == 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                
                    total_loss_list = " - "
                    printl('| epoch {:3d} | Batch {:3d} | {:4d}/{:4d} batches | '
                        'lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | losslist  {} | ppl {:8.2f}'.format(
                            epoch, Batch, batch, len(train_data) // bptt, scheduler.get_last_lr()[0],
                            elapsed * 1000 / log_interval,
                            cur_loss,total_loss_list, math.exp(cur_loss)))
                    total_loss = 0
                    #print(loss2)
                    print(scheduler.get_last_lr()[0])
                    start_time = time.time()

                    writer.add_scalar('train_loss', cur_loss,idloss)
                    idloss+=1

            if Batch%10==0:
                save(epoch*100000+Batch,saveDict={'encoder':model.state_dict(),'idloss':idloss,'epoch':epoch,'best_val_loss':best_val_loss},modelDir=checkpointPath)
    
    # train
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(epoch)
        printl('-' * 89)
        scheduler.step()
        printl('-' * 89)
