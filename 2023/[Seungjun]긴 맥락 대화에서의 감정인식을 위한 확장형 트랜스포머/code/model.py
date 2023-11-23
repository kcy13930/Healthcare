import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_extended_attention_mask(attention_mask):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"attention_mask (shape {attention_mask.shape})"
        )
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class GlobalEncoder(nn.Module):
    def __init__(self, in_feats, nhead=8, num_layers=11):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_feats, dim_feedforward=in_feats*4,  nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        out = self.transformer_encoder(x)
        return out

class Small_Electra_ETC(nn.Module):
    def __init__(self, base_model, in_feats, hidden_feats, out_feats):
        super().__init__()
        self.local_model = base_model
        # self.embedding = self.local_model.electra.embeddings
        # self.embedding_project = self.local_model.electra.embeddings_project
        self.embedding = self.local_model.embeddings
        self.local_layers = [layer for layer in self.local_model.encoder.layer]  
        
    def forward(self, xs):
        embed_xs = self.embedding(xs)
        embed_xs = self.embedding_project(embed_xs)
        return embed_xs
    
class Electra_ETC(nn.Module): # global + long2long + g2l/l2g
    def __init__(self, device, base_model, in_feats, hidden_feats, out_feats):
        super().__init__()
        
        self.device = device
        self.local_model = base_model
        self.hidden_feats = hidden_feats
        # self.embedding = self.local_model.electra.embeddings
        # self.local_layers = [layer for layer in self.local_model.electra.encoder.layer]
        self.embedding = self.local_model.embeddings
        self.embedding_project = self.local_model.embeddings_project
        self.local_layers = [layer for layer in self.local_model.encoder.layer]  
        
        self.global_position = PositionalEncoding(self.hidden_feats)
        self.global_model = GlobalEncoder(in_feats= in_feats, num_layers = len(self.local_layers))
        self.global_layers = [layer for layer in self.global_model.transformer_encoder.layers]
        
        self.classifier = nn.Linear(hidden_feats, out_feats) # 768
        
    def forward(self, xs):
        xs = xs[xs.sum(axis=1)>0]
        # last = xs[x.sum(axis=1)>0].shape[0]
        attention_mask = (xs[:,1:]>0).to(self.device)
        embed_xs = self.embedding(xs)
        embed_xs = self.embedding_project(embed_xs)
        
        global_tokens = embed_xs[:,0:1,:]
        local_tokens = embed_xs[:,1:,:]
        local_tokens = [local_token[mask] for local_token, mask in zip(local_tokens, attention_mask)]
        
        attention_mask_lens = attention_mask.sum(axis=1)
        long_tokens_mask = torch.zeros((attention_mask.shape[0], int(attention_mask_lens.sum()))).to(self.device)
        pre_index = 0
        for i, attention_mask_len in enumerate(attention_mask_lens):
            pre_index += int(attention_mask_len)
            long_tokens_mask[i][:pre_index] = 1
        attention_mask_accu_lens = long_tokens_mask.sum(axis=1)
        long_tokens_mask = get_extended_attention_mask(long_tokens_mask)
        # print('attention', attention_mask.shape, attention_mask_lens.shape, attention_mask_accu_lens.shape, long_tokens_mask.shape)
        
        global_attention_mask = torch.tril(torch.ones((global_tokens.size(0),global_tokens.size(0)))).to(self.device)
        
        for layer_index in range(len(self.local_layers)):
            # local_tokens = self.local_layers[layer_index](local_tokens)[0]
            # local_tokens_without_pad = torch.cat([local_token[x_i] for local_token, x_i in zip(local_tokens, x_index)], axis=0)
            local_tokens = [self.local_layers[layer_index](local_token.view(1,-1,self.hidden_feats))[0] for local_token in local_tokens]
            local_tokens_without_pad = torch.cat([local_token for local_token in local_tokens], dim=1)
            long_tokens = torch.cat([
                global_tokens,
                torch.cat([local_tokens_without_pad for i in range(embed_xs.size(0))], dim=0)
            ], dim=1)
            
            #l2g
            hooks = Hook(self.local_layers[layer_index], l2g = True)
            local_tokens_ = self.local_layers[layer_index](long_tokens, attention_mask = long_tokens_mask.permute(0, 1, 3, 2))[0]
            for i, local_token in enumerate(local_tokens_):
                start = int(attention_mask_accu_lens[i]-attention_mask_lens[i])
                end = int(attention_mask_accu_lens[i])
                local_tokens[i][:int(attention_mask_lens[i])] = local_token[start:end]
            hooks.close()
            # print(long_tokens.shape)
            
            #g2l
            hooks = Hook(self.local_layers[layer_index], l2g = False)
            global_tokens = self.local_layers[layer_index](long_tokens, attention_mask = long_tokens_mask)[0]
            hooks.close()
            # print(global_tokens.shape)
            
            # global_tokens =  global_tokens.view(-1,1,self.hidden_feats)
            if layer_index == 0:
                global_tokens = self.global_position(global_tokens)
                
                
            global_tokens = self.global_layers[layer_index](global_tokens, src_mask = global_attention_mask)
            
            # global_tokens = global_tokens.view(-1,1,self.hidden_feats)
            
            
        out = self.classifier(global_tokens[:,:,:])
        
        return out
    
class Hook():
    def __init__(self, layer, l2g = True):
        if l2g:
            self.hook_query = layer.attention.self.query.register_forward_pre_hook(self.long_pre_hook)
            self.hook_key = layer.attention.self.key.register_forward_pre_hook(self.global_pre_hook)
            self.hook_value = layer.attention.self.value.register_forward_pre_hook(self.global_pre_hook)
            self.hook_norm = layer.attention.output.register_forward_pre_hook(self.norm_pre_hook)
        if not l2g:
            self.hook_query = layer.attention.self.query.register_forward_pre_hook(self.global_pre_hook)
            self.hook_key = layer.attention.self.key.register_forward_pre_hook(self.long_pre_hook)
            self.hook_value = layer.attention.self.value.register_forward_pre_hook(self.long_pre_hook)
            self.hook_norm = layer.attention.output.register_forward_pre_hook(self.norm_pre_hook)
        
    def norm_pre_hook(self, module, in_):
        if in_[0].size(1) == [1]:
            return in_[0], in_[1][:,1:,:]
        if in_[0].size(1) != [1]:
            return in_[0], in_[1][:,0:1,:]
        
    def long_pre_hook(self, module, in_):
        return in_[0][:,1:,:]

    def global_pre_hook(self, module, in_):
        return in_[0][:,0:1,:]
    
    def long_hook(self, module, in_, out_):
        return out_[:,1:,:]
    
    def global_hook(self, module, in_, out_):
        return out_[:,0:1,:]
    
    def close(self):
        self.hook_query.remove()
        self.hook_key.remove()
        self.hook_value.remove()
        self.hook_norm.remove()

pre_emotion = lambda emotion : emotion[emotion.sum(axis=1)>0][-1]