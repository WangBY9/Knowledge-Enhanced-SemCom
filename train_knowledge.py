import os
import sys

import torch
import torch.utils.data as data
import torch.optim as optim
import numpy as np

from data_loader import WebNLG, collate_func
from preprocess import WebNLGTokenizer
import model
from model import make_model,subsequent_mask,make_std_mask,make_decoder
from utils import Channel, Crit, clip_gradient


os.chdir(sys.path[0])

_iscomplex = True
batch_size = 32
epochs = 100
learning_rate = 1e-4  
epoch_start = 1
weight_parameter = 0.02

# set path
save_model_path = "./ckpt/"
data_path = 'dataset'

if not os.path.exists(save_model_path): os.makedirs(save_model_path)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
data_parallel = False

# data loading parameters
train_loader_params = {'batch_size': batch_size,
                       'shuffle': True, 'num_workers':1,
                       'collate_fn': lambda x: collate_func(x),
                       'drop_last': True}

data_train = WebNLG(mode="train")
train_data_loader = data.DataLoader(data_train,**train_loader_params)

vocab_size = data_train.tokenizer.n_words

sc_model = make_model(vocab_size,vocab_size,act1=False,act2=False).to(device)
sc_model.load_state_dict(torch.load('ckpt/Transformer_Base.ckpt'))
knowledge_extractor = model.KnowledgeExtractor(embd_dim=128, output_dim=data_train.tokenizer.n_triples, num_layers=3, device=device)
knowledge_extractor.train()
channel = Channel(_iscomplex=_iscomplex)
optimizer = torch.optim.Adam(knowledge_extractor.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10,20,40], gamma = 0.5)

criterion = torch.nn.BCELoss(reduction="none")


for epoch in range(epoch_start, epoch_start + epochs):

    print('--------------------epoch: %d' % epoch)

    for batch_idx, (train_sents, train_triples, len_batch) in enumerate(train_data_loader):
        # distribute data to device
        train_sents = train_sents.to(device)  # with eos
        t_target = train_triples.float().to(device)
        #print(train_sents)############################
        # len_batch = len_batch.to(device) #cpu()
        
        src = train_sents[:, 1:]
        # trg = train_sents[:, :-1]
        # trg_y = train_sents[:, 1:]
        src_mask = (src != 0).unsqueeze(-2).to(device)
        # tgt_mask = make_std_mask(trg).to(device)
        _snr1 = np.random.randint(0, 10)

        with torch.no_grad():
            output= sc_model.encode(src, src_mask)
            output= channel.agwn(output, _snr=_snr1)
            output= sc_model.from_channel_emb(output)

        t_pred = knowledge_extractor(output)
       
        loss_weight = torch.ones(t_target.shape) * weight_parameter
        loss_weight[t_target > 0] = 1 - weight_parameter
        loss_weight = loss_weight.to(device)

        loss_original = criterion(t_pred, t_target)
        loss = torch.sum(loss_original * loss_weight) / batch_size
        
        optimizer.zero_grad()
        loss.backward()
        # clip_gradient(optimizer, 0.1) 
        optimizer.step()
        if batch_idx % 100==0:
           print('[%4d / %4d]    '%(batch_idx, epoch) , '    loss = ', loss.item())

    if epoch % 10 == 0:
        torch.save(knowledge_extractor.state_dict(),
                   os.path.join(save_model_path, 'Transformer_extractor.ckpt'))
    
    scheduler.step()
