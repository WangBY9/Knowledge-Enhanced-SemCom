import os
import sys

import torch
import torch.utils.data as data
import torch.optim as optim
import numpy as np

from preprocess import WebNLGTokenizer
from data_loader import WebNLG, collate_func
from model import make_model,subsequent_mask,make_std_mask,make_decoder
from utils import Channel, Crit, clip_gradient


os.chdir(sys.path[0])

_iscomplex = True
batch_size = 64
epochs = 50
learning_rate = 3e-4  
epoch_start = 1
time_penalty = 0.0001


save_model_path = "./ckpt/"
data_path = 'dataset'

if not os.path.exists(save_model_path): os.makedirs(save_model_path)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
data_parallel = False

train_loader_params = {'batch_size': batch_size,
                       'shuffle': True, 'num_workers':8,
                       'collate_fn': lambda x: collate_func(x),
                       'drop_last': True}

data_train = WebNLG(mode="train")
train_data_loader = data.DataLoader(data_train,**train_loader_params)

vocab_size = data_train.tokenizer.n_words

tmp_model = make_model(vocab_size,vocab_size,act1=True,act2=True).to(device)  

channel = Channel(_iscomplex=_iscomplex, _device=device)

_params = list(tmp_model.parameters())
optimizer = torch.optim.Adam(_params, lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [20,40], gamma = 0.5)
crit = Crit()

act1=True
act2=True

def train(model, device, train_loader, optimizer, epoch):

    model.train()
    if data_parallel: torch.cuda.synchronize()

    print('--------------------epoch: %d' % epoch)

    for batch_idx, (train_sents, _, len_batch) in enumerate(train_loader):
        train_sents = train_sents.to(device)  
        len_batch = len_batch.to(device) #cpu()
        optimizer.zero_grad()
        src = train_sents[:, 1:]
        trg = train_sents[:, :-1]
        trg_y = train_sents[:, 1:]
        src_mask = (src != 0).unsqueeze(-2).to(device)
        tgt_mask = make_std_mask(trg).to(device)
        if act1 == True:
            output,remainders1,n_updates1 = model.encode(src, src_mask)
            remainders1=torch.sum(remainders1)
            n_updates1=torch.sum(n_updates1)
            ponder_cost1=remainders1+n_updates1
        else:
            output= model.encode(src, src_mask)
        _snr1= np.random.randint(0, 10)
        
        output= channel.agwn(output, _snr=_snr1)
        output= model.from_channel_emb(output)
        if act2 == True:
            output,remainders2,n_updates2= model.decode(output, src_mask,trg, tgt_mask)
            remainders2=torch.sum(remainders2)
            n_updates2=torch.sum(n_updates2)
            ponder_cost2=remainders2+n_updates2
        else:
            output = model.decode(output, src_mask,trg, tgt_mask)
        output= model.generator.forward(output)

        loss = crit('xe', output, trg_y, len_batch)
        if act1 ==True:
            loss = loss + ponder_cost1*time_penalty*(1e-4)
        if act2 ==True:
            loss = loss + ponder_cost2*time_penalty*(3e-2)
        loss.backward()
        clip_gradient(optimizer, 0.1)
        optimizer.step()

        if batch_idx % 100==0:
            print('[%4d / %4d]    '%(batch_idx, epoch) , '    loss = ', loss.item())

    if epoch % 10 == 0: 
        torch.save(model.state_dict(), os.path.join(save_model_path, 'UT_Base.ckpt'))
        # print("Epoch {} model saved!".format(epoch))


# start training
for epoch in range(epoch_start, epoch_start + epochs):
    train(tmp_model, device, train_data_loader, optimizer, epoch)
    scheduler.step()


