
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa

# get statics
def split_and_strip(x):
    xs = x.split(';')
    return [x.strip() for x in xs]

# sort_dict = lambda x: dict(sorted(x.items(), key= lambda item: item[1])[::-1])
def sort_dict(x):
    return dict(sorted(x.items(), key= lambda item: item[1])[::-1])

def get_value_counts(xs):
    value_counts = {}
    for x in xs:
        if not x in value_counts:
            value_counts[x]=1
        else:
            value_counts[x]+=1
    return sort_dict(value_counts)

def ratio(df, value_counts):
    total = np.sum(list(value_counts.values()))
    for key, value in value_counts.items():
        percent = (value)/(total+0.1e-8)*100
        print(f"{key} : {percent:2.2f}%")
        print()
        
def visual_ratio(value_counts):
    xlabel= value_counts.keys()
    y = value_counts.values()

    plt.figure(figsize=(7, 7))
    plt.axis('equal')
    plt.title('Emotion Categories')
    plt.pie(y, labels=xlabel,
            autopct='%2.2f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.savefig('ratio.png')
    # plt.show()
    
def visual_heatmap(annotation_df):
    y = annotation_df[' .1_Valence'].astype('float').values
    x = annotation_df[' .2_Arousal'].astype('float').values
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=(32,32))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Plot heatmap
    plt.clf()
    plt.title('VA')
    plt.ylabel('Valence')
    plt.xlabel('Arousal')
    plt.imshow(heatmap, extent=extent)
    # plt.show()
    
    plt.savefig('VA_heatmap.png')

def clean_text(text, pattern = ['c/', 'n/', 'N/', 'u/', 'l/', 'b/', '\*', '\+', '/', '\n' , 'o/' ]):
    return re.sub('|'.join(pattern), '', text).strip()

def encoding_text(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    return tokenizer.encode(tokens)

def get_mel_spec(wave, sr = 16000, n_fft = 2048, win_length = 2048, hop_length = 1024, n_mels = 128):
    stft = np.abs(librosa.stft(wave, n_fft=n_fft, win_length = win_length, hop_length=hop_length))
    mel_spec = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
    return mel_spec

def MinMaxScaler(X, min, max, range_min = 0, range_max = 1):
    X_std = (X - min) / (max - min)
    X_scaled = X_std * (range_max - range_min) + range_min
    return X_scaled

def MinMaxDecoder(X_scaled, min, max, range_min = 0, range_max = 1):
    X_std = (X_scaled - range_min) / (range_max - range_min)
    X = X_std * (max - min) + min
    return X

def pre_emotion(emotion):
    return emotion[emotion.sum(axis=1)>0][-1]

def one_hot_emotion(emotion, num_classes):
    return F.one_hot(torch.tensor(emotion), num_classes=num_classes).sum(axis=0).numpy()