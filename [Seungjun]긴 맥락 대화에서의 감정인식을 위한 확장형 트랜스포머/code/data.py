import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

def load_dataset(path, emotion2id):
    dialogues = torch.load(path)['dialogues']
    dialogues_ = []
    for dialogue in dialogues:
        utterances = dialogue['utterances']
        dialogue_ = []
        for utterance in utterances:
            utterance_ = {
                'speaker': utterance['speaker'],
                'text': utterance['transcript'],
                'emotion': [emotion2id[emotion] for emotion in utterance['emotion']],
                'valence': utterance['valence'],
                'arousal': utterance['arousal'],
            }
            dialogue_.append(utterance_)
        dialogues_.append(dialogue_)
    return dialogues_


def pad_to_len(list_data, max_len, pad_value):
    list_data = list_data[-max_len:]
    len_to_pad = max_len - len(list_data)
    pads = [pad_value] * len_to_pad
    list_data.extend(pads)
    return list_data

class ERC_ETRI_Dataset(Dataset):
    def __init__(self, dialogues, config):
        self.data = self.build_dataset(dialogues,
                                        config['tokenizer'],
                                        config['PAD_token'],
                                        config['max_len'],
                                        config['num_classes'],
                                        utterance_limitation = config['utterance_limitation']
                                        )
        self.utterances, self.emotions, self.valences, self.arousals = self.data
    def __len__(self):
        return len(self.utterances)
    
    def __getitem__(self, index):
        # return self.utterances[index], self.emotions[index], self.valences[index], self.arousals[index]
       
        return self.utterances[index], self.emotions[index], self.valences[index], self.arousals[index]
    
    def build_dataset(self, dialogues,
                      tokenizer,
                      PAD_token,
                      max_len,
                      num_classes,
                      utterance_limitation
                      ):
        utterances = []
        emotions = []
        valences = []
        arousals = []
        
        for dialogue in dialogues:
            emotion_ids = []
            utterance_ids = []
            for idx, turn_data in enumerate(dialogue):
                text_with_speaker = turn_data['speaker'] + ':' + turn_data['text']
                token_ids = tokenizer(text_with_speaker)['input_ids']
                utterance_ids.append(token_ids)
                emotion_ids.append(F.one_hot(torch.tensor(dialogue[idx]['emotion']), num_classes= num_classes).sum(axis=0))
                
                utterance_ids = [pad_to_len(utterance_id, max_len, PAD_token) for utterance_id in utterance_ids]
                
                if len(utterance_ids) > utterance_limitation:
                    utterance_ids = utterance_ids[-utterance_limitation:]
                    emotion_ids = emotion_ids[-utterance_limitation:]
                    
                full_context = torch.full((utterance_limitation, max_len), PAD_token)
                full_context[:len(utterance_ids)] = torch.tensor(utterance_ids)
                utterances.append(full_context)
                
                full_emotion = torch.full((utterance_limitation, num_classes),0)
                full_emotion[:len(utterance_ids)] = torch.stack(emotion_ids).to(torch.float)
                emotions.append(full_emotion)
                
                valences.append(dialogue[idx]['valence'])
                arousals.append(dialogue[idx]['arousal'])

            
        return utterances, emotions, valences, arousals