import os
import json
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler, TensorDataset
from tqdm import tqdm

from sklearn.metrics import classification_report

import gc
import warnings
warnings.filterwarnings(action='ignore')

from transformers import ElectraModel, ElectraTokenizer, ElectraForMaskedLM #, ElectraForPreTraining
from loss import focal_loss
from util import pre_emotion
from data import load_dataset, ERC_ETRI_Dataset
from model import Electra_ETC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
discriminator = ElectraModel.from_pretrained("skplanet/dialog-koelectra-small-discriminator")
tokenizer = ElectraTokenizer.from_pretrained("skplanet/dialog-koelectra-small-discriminator")
base_model = ElectraModel.from_pretrained("skplanet/dialog-koelectra-small-discriminator")
generator = ElectraForMaskedLM.from_pretrained("skplanet/dialog-koelectra-small-generator")
generator.to(device)
for para in generator.parameters():
    para.requires_grad = False
    
pre_path = './pre_data'
train_path = os.path.join(pre_path, 'train.pkl')
valid_path = os.path.join(pre_path, 'valid.pkl')
# test_path = os.path.join(pre_path, 'test.pkl')
LabelEncoder_path = os.path.join(pre_path, 'LabelEncoder.pkl')

LabelEncoder = torch.load(LabelEncoder_path)
emotion2id = LabelEncoder['emotion2id']
id2emotion = LabelEncoder['id2emotion']

max_len = 128
train = True
CLS_token = tokenizer.vocab['[CLS]']
SEP_token = tokenizer.vocab['[SEP]']
MASK_token = tokenizer.vocab['[MASK]']
PAD_token = tokenizer.vocab['[PAD]']
num_classes = len(emotion2id)
utterance_limitation = 64

train_dialogues = load_dataset(train_path, emotion2id)
valid_dialogues = load_dataset(valid_path, emotion2id)
# test_dialogues = load_dataset(test_path, emotion2id)

config = {
    'tokenizer':tokenizer,
    'PAD_token':PAD_token,
    'max_len':max_len,
    'num_classes':num_classes,
    'utterance_limitation':utterance_limitation
}

train_dataset = ERC_ETRI_Dataset(train_dialogues, config)
valid_dataset = ERC_ETRI_Dataset(valid_dialogues, config)
# test_dataset = ERC_ETRI_Dataset(test_dialogues, config)


def encoding_text(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    return tokenizer.encode(tokens)

def save_model(epoch, model, opt, loss, PATH):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss,
        }, PATH + f'/{epoch}_model.pt')
    
def generate_utterance(utterance, top_k = 5):
    utterance = utterance[utterance.sum(axis=1)>0]
    if torch.randint(top_k+1,(1,))!=0:
        utterance_embed = utterance.clone()[:, :3]
        masked_utterance = utterance.clone()
        masked_utterance = torch.cat([masked_utterance[:, :1], masked_utterance[:, 3:]],dim=1)
        utterance_len = (masked_utterance>0).sum(axis=1)
        selection_ratio = (utterance_len * 0.15).to(torch.int8)
        selected_tokens = [torch.randint(low=1, high=ut-1, size=(sr,)) for ut, sr in zip(utterance_len, selection_ratio)]
        for i, st in enumerate(selected_tokens):
            masked_utterance[i,st] = MASK_token

        masked_index = masked_utterance==MASK_token
        out = generator(masked_utterance).logits
        masked_tokens = out[masked_index,:].squeeze()

        probs = masked_tokens.softmax(dim=-1)
        _, predictions = probs.topk(top_k)
        predictions = predictions.view(-1,top_k)
        selected_k = torch.randint(top_k,(predictions.size(0),))
        if not predictions.sum():
            return utterance
            
        selected_predictions = torch.stack([predictions[i][selected_k[i]] for i in range(predictions.size(0))])
        masked_utterance[masked_index] = selected_predictions
        generated_utterance = torch.cat([utterance_embed[:, :3], masked_utterance[:, 1:]],dim=1)
        return generated_utterance
    else:
        return utterance
    
class_count = torch.cat([pre_emotion(emotion) for emotion in train_dataset.emotions], dim=0).view(-1,7).sum(axis=0)
class_sum = train_dataset.__len__()
class_weights = class_sum/class_count

class_weights[4] = np.log(class_weights[4])
class_weights_all = torch.tensor([class_weights[pre_emotion(emotion)>0].mean() for emotion in train_dataset.emotions])


sampler = RandomSampler(
    train_dataset
)

batch_size = 1
train_dataloader = DataLoader(
    train_dataset,
    batch_size= batch_size,
    sampler=sampler,
    #num_workers = 0  # multiprocessing.cpu_count()
    drop_last=True
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size= batch_size*2,
    #num_workers = 0  # multiprocessing.cpu_count()
    shuffle=False
)

# test_dataloader = DataLoader(
#     test_dataset,
#     batch_size= batch_size*2,
#     #num_workers = 0  # multiprocessing.cpu_count()
#     shuffle=False
# )

def train():
    sigmoid = lambda x: 1/(1 +np.exp(-x))
    pred = lambda x: np.around(sigmoid(np.stack(x)))

    emotions = []
    for i, (data, emotion, valence, arousal) in enumerate(tqdm(train_dataloader)):
        emotions.append(torch.stack([pre_emotion(e) for e in emotion]))
    emotions = torch.cat(emotions)
    for emotion_count, key in zip(emotions.sum(axis=0), emotion2id.keys()):
        print(key,':',emotion_count)


    PATH = './20_weights/'
    if not os.path.isdir(PATH):
        os.mkdir(PATH)
        
    PATH = './20_weights/weights/'
    if not os.path.isdir(PATH):
        os.mkdir(PATH)

    BEST_PATH = './20_weights/best_weights/'
    if not os.path.isdir(BEST_PATH):
        os.mkdir(BEST_PATH)

    progress = False
    epohcs = 5
    start = 0
    
    # base_model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model = Electra_ETC(device=device,base_model=base_model, in_feats =256, hidden_feats=256, out_feats=num_classes)
    model.to(device)
    
    opt = torch.optim.AdamW(model.parameters())

    # criterion = nn.MultiLabelSoftMarginLoss()
    criterion = focal_loss

    best_f1 = 20
    history = {
        'train':[],
        'valid':[]
    }

    if progress:
        print('Progress Learning')
        with open("./20_weights/history.json", "r") as f:
            history = json.load(f)
        checkpoint = torch.load('./20_weights/weights/20_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start = checkpoint['epoch']+1
        
    end = start + epohcs
    for epoch in range(start, end):
        total_loss = []
        model.train()
        # for i, (data, emotion, valence, arousal) in enumerate(tqdm(train_dataloader, desc = f"epoch - {epoch}" )):
        train_dataloader_it = iter(train_dataloader)
        for i in tqdm(range(1000), desc = f"epoch - {epoch}" ):
            data, emotion, valence, arousal = next(train_dataloader_it)
            
            inputs = data.to(device)
            inputs = generate_utterance(inputs.squeeze(0))
            # outs = torch.cat([model(input_) for input_ in inputs], dim=0).squeeze(1)
            outs = model(inputs).squeeze(1)
            targets = emotion[emotion.sum(axis=2)>0].to(torch.float32).to(device)
            # targets = torch.cat([targets[:,:4], targets[:,5:]], dim=1)
            loss = criterion(outs, targets)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss.append(loss.item())
            
            gc.collect()
            torch.cuda.empty_cache()

        total_loss = np.mean(total_loss)
        history['train'].append(total_loss)
        print('epoch: % -3s loss: %s' % (epoch, total_loss))
        
        model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []
            for i, (data, emotion, valence, arousal) in enumerate(tqdm(valid_dataloader, desc = f"epoch - {epoch}" )):
                inputs = data.to(device)
                # outs = torch.cat([model(input_)[-1] for input_ in inputs], dim=0).squeeze(1).cpu().detach().numpy()
                outs = torch.cat([model(input_)[-1] for input_ in inputs], dim=0).squeeze(1).cpu().detach().numpy()
                targets = torch.stack([pre_emotion(e) for e in emotion]).to(torch.float32).cpu().detach().numpy()
                
                y_true.extend(targets)
                y_pred.extend(outs)
                
                gc.collect()
                torch.cuda.empty_cache()
        
        y_pred = pred(y_pred)
        # y_pred = np.concatenate([y_pred[:,:4], (y_pred.sum(axis=1)==0).astype('int').reshape(y_pred.shape[0],-1), y_pred[:,4:]],axis=1)
        his = classification_report(y_true, y_pred,
                                    target_names= list(emotion2id.keys()),
                                    output_dict=True)
        his['epoch'] = epoch
        history['valid'].append(his)
        file_path = "./20_weights/history.json"

        with open(file_path, 'w') as f:
            json.dump(history, f, indent=4)
            
        f1 = his['weighted avg']
        if best_f1 < f1['f1-score']:
            best_f1 = f1['f1-score']
            save_model('best', model, opt, criterion, BEST_PATH)
            
        save_model(epoch, model, opt, criterion, PATH)

        if epoch % 1 == 0:
            print(his)
            
if __name__ == '__main__':
    train()