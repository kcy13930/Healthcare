
import os
import json
import time

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report

import gc
import warnings
warnings.filterwarnings(action='ignore')

from tqdm import tqdm
from transformers import ElectraModel, ElectraTokenizer, ElectraForMaskedLM
from util import pre_emotion
from data import load_dataset, ERC_ETRI_Dataset
from loss import focal_loss
from model import Electra_ETC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
discriminator = ElectraModel.from_pretrained("skplanet/dialog-koelectra-small-discriminator")
tokenizer = ElectraTokenizer.from_pretrained("skplanet/dialog-koelectra-small-discriminator")
generator = ElectraForMaskedLM.from_pretrained("skplanet/dialog-koelectra-small-generator")
batch_size = 1

pre_path = './pre_data'
# train_path = os.path.join(pre_path, 'train.pkl')
# valid_path = os.path.join(pre_path, 'valid.pkl')
test_path = os.path.join(pre_path, 'test.pkl')
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

# train_dialogues = load_dataset(train_path, emotion2id)
# valid_dialogues = load_dataset(valid_path, emotion2id)
test_dialogues = load_dataset(test_path, emotion2id)

config = {
    'tokenizer':tokenizer,
    'PAD_token':PAD_token,
    'max_len':max_len,
    'num_classes':num_classes,
    'utterance_limitation':utterance_limitation
}

# train_dataset = ERC_ETRI_Dataset(train_dialogues, config)
# valid_dataset = ERC_ETRI_Dataset(valid_dialogues, config)
test_dataset = ERC_ETRI_Dataset(test_dialogues, config)

test_dataloader = DataLoader(
    test_dataset,
    batch_size= batch_size*8,
    #num_workers = 0  # multiprocessing.cpu_count()
    shuffle=False
)

with open("./20_weights/history.json", "r") as f:
    history = json.load(f)

i=3
history_df = pd.DataFrame(history['valid'])
crit = ['micro avg', 'macro avg', 'weighted avg', 'samples avg']
f1 = pd.DataFrame(list(history_df[crit[3]]))['f1-score']
best_f1 = f1[f1==f1.max()]
best_f1_index = best_f1.index[-1]
print(f'criterion : {crit[0]}')
print(f'f1:{best_f1}')
print(f'f1_index:{best_f1_index}')

BEST_PATH = f'./20_weights/weights/{best_f1_index}_model.pt'
# BEST_PATH = f'./20_weights/best_weights/gen_proposed_model.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_model = ElectraModel.from_pretrained("skplanet/dialog-koelectra-small-discriminator")
generator = ElectraForMaskedLM.from_pretrained("skplanet/dialog-koelectra-small-generator")
generator.to(device)
for para in generator.parameters():
    para.requires_grad = False

model = Electra_ETC(device=device,base_model=base_model, in_feats =256, hidden_feats=256, out_feats=num_classes)
model.to(device)

checkpoint = torch.load(BEST_PATH)
model.load_state_dict(checkpoint['model_state_dict'])

sigmoid = lambda x: 1/(1 +np.exp(-x))
pred = lambda x: np.around(sigmoid(np.stack(x)))
model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    for i, (data, emotion, valence, arousal) in enumerate(tqdm(test_dataloader, desc = "test" )):
        inputs = data.to(device)
        outs = torch.cat([model(input_)[-1] for input_ in inputs], dim=0).squeeze(1).cpu().detach().numpy()
        # outs = model(inputs.squeeze(0)).squeeze(1).cpu().detach().numpy()
        targets = torch.stack([pre_emotion(e) for e in emotion]).to(torch.float32).cpu().detach().numpy()
        y_true.extend(targets)
        y_pred.extend(outs)
        # y_pred = np.concatenate([y_pred[:,:4], (y_pred.sum(axis=1)==0).astype('int').reshape(y_pred.shape[0],-1), y_pred[:,4:]],axis=1) 
        gc.collect()
        torch.cuda.empty_cache()

y_pred = pred(y_pred)
his = classification_report(y_true, y_pred,
                            target_names= list(emotion2id.keys()))
print(his)