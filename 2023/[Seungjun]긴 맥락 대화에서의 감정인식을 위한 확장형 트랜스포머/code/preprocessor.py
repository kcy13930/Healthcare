import os
import glob

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import ElectraTokenizer
from util import clean_text, split_and_strip, get_value_counts, ratio, visual_ratio, visual_heatmap

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--data_pth", dest="path_data", type=str, required=True,
                    help='data path')
# parser.add_argument("-pp", "--pre_data_pth", dest="pre_data", required=True,
#                     help='where to save the preprocessed path')
args = parser.parse_args()

# path_data = '../../data'
path_data = args.path_data
path_KEMDY19 = os.path.join(path_data, 'KEMDY19')
path_KEMDY20 = os.path.join(path_data, 'KEMDY20')
path_KEMDY20_1 = os.path.join(path_data, 'KEMDY20_v1_1')
tokenizer = ElectraTokenizer.from_pretrained("skplanet/dialog-koelectra-small-discriminator")

# pre_path = args.pre_data

class KEMDY20_Dataset():
    def __init__(self, path_KEMDY):
        self.path = {}
        
        get_path_file_list = lambda path : glob.glob(os.path.join(path,'**/*.csv'), recursive=True)
        get_path_txt_list = lambda path : glob.glob(os.path.join(path,'**/*.txt'), recursive=True)
        get_path_wav_list = lambda path : glob.glob(os.path.join(path,'**/*.wav'), recursive=True)
        self.path['annotation'] = get_path_file_list(os.path.join(path_KEMDY, 'annotation'))
        self.path['ECG'] = get_path_file_list(os.path.join(path_KEMDY, 'ECG'))
        self.path['EDA'] = get_path_file_list(os.path.join(path_KEMDY, 'EDA'))
        self.path['TEMP'] = get_path_file_list(os.path.join(path_KEMDY, 'TEMP'))
        self.path['wav'] = get_path_wav_list(os.path.join(path_KEMDY, 'wav'))
        self.path['text'] = get_path_txt_list(os.path.join(path_KEMDY, 'wav'))
        
        self.preprocessor = {
            'annotation' : self.preprocess_annotation, 
            #'ECG' : self.prerpocess_ECG
        }
        
        self.data = {}
        self.data['annotation'] = self.read_annotations(self.path['annotation'], self.preprocessor['annotation']).reset_index(drop=True)
        # self.data['ECG'] = self.read_sequence(self.path['ECG'], columns = ['sampling_order', 'Refit_ECG', 'Time_order', 'Segment ID']).reset_index(drop=True)
        # self.data['EDA'] = self.read_sequence(self.path['EDA'], columns = ['EDA_change', 'samplint_order', 'Time_order', 'Segment ID']).reset_index(drop=True)
        # self.data['TEMP'] = self.read_sequence(self.path['TEMP'], columns = ['Temp', 'sampling_order', 'Time_order', 'Segment ID']).reset_index(drop=True)
        self.data['text'] = self.read_text(self.path['text']).reset_index(drop=True)
        self.data['wav'] = self.read_wav(self.path['wav']).reset_index(drop=True)
        
    def read_annotations(self, path_file_list, preprocessor):
        df_list = []
        for path_file in tqdm(path_file_list, desc = 'downloading'):
            file_name = os.path.split(path_file)[-1]
            file_df = pd.read_csv(path_file)
            file_df = preprocessor(file_df)
            file_df['fname'] = file_name
            df_list.append(file_df)
        df = pd.concat(df_list, axis=0)
        return df
    
    def read_sequence(self, path_file_list, columns):
        df_list = []
        for path_file in tqdm(path_file_list, desc = 'downloading'):
            file_name = os.path.split(path_file)[-1].split('.')[0]
            file_df = pd.read_csv(path_file, names=columns, dtype={'Segment ID':object})
            df_list.append(file_df)
        df = pd.concat(df_list, axis=0)
        return df
    
    def read_text(self, path_file_list):
        data = {
            'fname':[],
            'text':[],
        }
        for path_file in tqdm(path_file_list, desc = 'downloading'):
            file_name = os.path.split(path_file)[-1].split('.')[0]
            f = open(path_file, mode = 'r', encoding='cp949')
            lines = f.readlines()
            f.close()
            
            lines = ''.join(lines)
            data['fname'].append(file_name)
            data['text'].append(lines)
        return pd.DataFrame(data)
    
    def read_wav(self, path_file_list):
        data = {
            'fname':[],
            'wav':[],
        }
        for path_file in tqdm(path_file_list, desc = 'downloading'):
            file_name = os.path.split(path_file)[-1].split('.')[0]
            data['fname'].append(file_name)
            data['wav'].append(path_file)
        return pd.DataFrame(data)
    
    def preprocess_annotation(self, df):
        columns = df.columns
        meta_columns = df.iloc[0,:]
        pre_column = columns[0]
        new_column = []
        for column, meta_column in zip(columns, meta_columns):
            if not column.startswith('Unnamed'):
                pre_column = column
            if pd.notna(meta_column):
                new_column.append(f'{pre_column}_{meta_column}')
            if pd.isna(meta_column):
                new_column.append(column)
        df.columns = new_column
        return df[df['Total Evaluation_Emotion']!='Emotion']



def create_dataset(df, special = ['c/', 'n/', 'N/', 'u/', 'l/', 'b/', '\*', '\+', '/', '\n' , 'o/' ]):
    data_dict = {
        'dialogues':[]
        }
    for section_dialogue in df.dialogue.unique():
        utterances = df[df.dialogue==section_dialogue].reset_index(drop=True)
        utterances_ = []
        for i in range(utterances.shape[0]):
            utterance = utterances.iloc[i,:]
            utterance_id = utterance.utterance
            speaker = utterance_id.split('_')[-2][-1]
            text = utterance.text
            text = clean_text(text, pattern = special)
            # tokens = encoding_text(text, tokenizer = tokenizer) if text != '' else text
            tokens = tokenizer.tokenize(text)
            emotion = utterance['Total Evaluation_Emotion']
            valence = utterance[' .1_Valence']
            arousal = utterance[' .2_Arousal']

            
            utterance_ = {
                'utterance_id':utterance_id,
                'speaker':speaker,
                'transcript':text,
                'tokens':tokens,
                'emotion':emotion.split(';'),
                'valence':valence,
                'arousal':arousal
            }
            utterances_.append(utterance_)
            
        data_dict['dialogues'].append({
            'dialogue_id':section_dialogue,
            'utterances':utterances_
            })
        
    return data_dict


def preprocess_KEMDY20_1(annotation_df, pre_path = './pre_data'):
    annotation_df = KEMDY20_1.data['annotation'].copy()
    text_df = KEMDY20_1.data['text'].copy()
    wav_df = KEMDY20_1.data['wav'].copy()

    annotation_df['Section'] = annotation_df['Segment ID_ '].map(lambda x: x.split('_')[0])
    annotation_df['dialogue'] = annotation_df['Segment ID_ '].map(lambda x: '_'.join(x.split('_')[:2]))
    annotation_df['utterance'] = annotation_df['Segment ID_ '].map(lambda x: x)
    annotation_df['gender'] = annotation_df['utterance'].map(lambda x: x.split('_')[-2][-1])
    annotation_df_ = annotation_df[annotation_df['gender'] == annotation_df['utterance'].map(lambda x: x.split('_')[-2][-1])].sort_values(by=['Section','Numb_ ']).reset_index(drop=True)
    annotation_df_['Numb_ '] = annotation_df_['Numb_ '].astype('int32')
    annotation_df_ = annotation_df_.sort_values(by=['Section','Numb_ ']).reset_index(drop=True)

    text_df.columns = ['Segment ID_ ', 'text']
    wav_df.columns = ['Segment ID_ ', 'wav']

    df = pd.merge(left = annotation_df_, right = text_df, how = 'outer', on = 'Segment ID_ ')
    df = pd.merge(left = df, right = wav_df, how = 'outer', on = 'Segment ID_ ')

    df = df.rename(columns={'Numb_ ':'Numb', 'Segment ID_ ':'Segment ID', 'WAV_start':'Wav_start',' _end':'Wav_end'})
    df = df[df.Numb.notna()]

    # EDA Section
    # print Missing unit
    feature = []
    for x in ['Wav']:
        feature.append(f'{x}_start')
        feature.append(f'{x}_end')
    modal = df[feature]
    
    print('Missing Values unit %')
    print('='*25)
    print(f'{modal.isna().sum()/modal.shape[0]*100}')

    # print ratio
    emotions = []
    for x in df['Total Evaluation_Emotion'].map(split_and_strip).values:
        emotions+=x

    print(df.shape)
    value_counts = get_value_counts(emotions)

    print(value_counts)
    ratio(df, value_counts)
    visual_ratio(value_counts)
    visual_heatmap(annotation_df)

    print('Section value counts')
    print(df['Section'].value_counts().describe())
    print()
    print('Dialogue value counts')
    print(df['dialogue'].value_counts().describe())

    # plt.style.use('default')
    # plt.rcParams['figure.figsize'] = (10, 10)
    # plt.rcParams['font.size'] = 12

    # fig, ax = plt.subplots()

    # ax.boxplot([df['dialogue'].value_counts()])
    # ax.set_xticks([])
    # ax.set_xlabel('dialogue')
    # ax.set_ylabel('count')

    # Preprocess
    target_feature = ['Total Evaluation_Emotion',' .1_Valence', ' .2_Arousal']
    target = KEMDY20_1.data['annotation'][target_feature]
    target.head()

    Sections = df.Section.unique()
    ratio_ = np.array([0.7, 0.15, 0.15]) * len(Sections) # train valid test ratio
    ratio_ = ratio_.astype('int')
    np.random.seed(2023)
    Sections = np.random.permutation(Sections)

    train = Sections[:ratio_[0]]
    valid =  Sections[ratio_[0]:-ratio_[2]]
    test = Sections[-ratio_[2]:]
    train, valid, test

    # save preprocessed data
    if not os.path.isdir(pre_path):
        os.mkdir(pre_path)
        
    train_path = os.path.join(pre_path, 'train.pkl')
    valid_path = os.path.join(pre_path, 'valid.pkl')
    test_path = os.path.join(pre_path, 'test.pkl')

    train_df = df.query(f'Section in {list(train)}')
    valid_df = df.query(f'Section in {list(valid)}')
    test_df = df.query(f'Section in {list(test)}')


    train_dict = create_dataset(train_df)
    valid_dict = create_dataset(valid_df)
    test_dict = create_dataset(test_df)

    torch.save(train_dict, train_path)
    torch.save(valid_dict, valid_path)
    torch.save(test_dict, test_path)
    
    # save LabelEncoder
    emotions = []
    for emotion in train_df['Total Evaluation_Emotion'].map(lambda x: x.split(';')):
        emotions+=emotion
    emotions = sorted(list(set(emotions)))

    # emotions.remove('neutral')
    
    LabelEncoder = {}
    emotion2id = {}
    id2emotion = {}
    for i, emotion in enumerate(emotions):
        emotion2id[emotion]=i
        id2emotion[i]=emotion

    LabelEncoder = {
        'emotion2id':emotion2id,
        'id2emotion':id2emotion
    }

    LabelEncoder_path = os.path.join(pre_path, 'LabelEncoder.pkl') 
    torch.save(LabelEncoder, LabelEncoder_path)

if __name__ == '__main__':
    print(path_KEMDY20_1)
    KEMDY20_1 = KEMDY20_Dataset(path_KEMDY20_1)
    preprocess_KEMDY20_1(KEMDY20_1)