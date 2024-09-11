
import asyncio
import base64
import io
import json
import os
import random
import string
import time
import aiohttp
import aiohttp.http_writer
from vocos import Vocos
from einops import reduce, repeat
from transformers import PreTrainedTokenizerFast
import torch
torch.autograd.set_detect_anomaly(True)
model_state_dict = "exp.pt"

import json
import csv
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
layers =  12*2
# from datasets import load_dataset

# # load model and processor

# whisperprocessor = WhisperProcessor.from_pretrained("openai/whisper-small")

# whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# whisper.config.forced_decoder_ids = None

# load dummy dataset and read audio files



def read_csv(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

# read the csv file
data = read_csv('CMU.in.IPA.txt')

# print(data)

# remove [\t] from start of each line
data = [(line[0],line[1].replace('\t', '')) for line in data if line.__len__() > 1]

# print(data)

# create a dictionary with key as the first element and value as the second element
ipa_dict = {}
for line in data:

    # remove trailing (x) where x is 1,2,3, or 4
    key = line[0].split('(')[0]

    if key not in ipa_dict:
        ipa_dict[key] = []

    ipa_dict[key].append(line[1])

# print(ipa_dict)

# get a list of all characters that appear in the key values
all_chars = []
for key in ipa_dict:
    for val in ipa_dict[key]:
        for char in val:
            if char not in all_chars:
                all_chars.append(char)

# print(all_chars)


# create a two step greedy tokenizer
# first step: tokenize from english text to ipa
# second step: tokenize from ipa to all_chars

class ipa_tokenizer():
    def __init__(self, ipa_dict, all_chars):
        self.ipa_dict = ipa_dict
        self.all_chars = all_chars
        self.max_ipa_len = max([len(val) for key in ipa_dict for val in ipa_dict[key]])
        self.context_size = all_chars.__len__() + 256

        self.all_char_to_token = {
            char: i for i, char in enumerate(all_chars)
        }

    

    def encode(self, text):
        # first step: tokenize from english text to ipa
        ipa_text = self.tokenize_to_ipa(text.lower())
        # print(ipa_text)

        # second step: tokenize from ipa to all_chars
        all_chars_text = self.tokenize_to_all_chars(ipa_text)
        # print(all_chars_text)
        return all_chars_text
    
    def add_pronunciation(self, text, equiv):
        ipa_text = "".join([self.tokenize_to_ipa(e) for e in equiv])
        self.ipa_dict[text] = [ipa_text]

    def tokenize_to_ipa(self, text):
        ipa_text = ''
        i = 0
        temptext = text
        while len(temptext) > 0:
            curslice = self.max_ipa_len
            for j in range(curslice, 0, -1):
                if temptext[:j] in self.ipa_dict:
                    # print(temptext[:j])
                    ipa_text += self.ipa_dict[temptext[:j]][0]
                    temptext = temptext[j:]
                    break
                elif j == 1:
                    ipa_text += temptext[:j]
                    temptext = temptext[j:]

        return ipa_text

    def tokenize_to_all_chars(self, text):
        all_char_tokens = []
        for o in text:
            if o in self.all_chars:
                all_char_tokens.append(self.all_char_to_token[o])
            else:
                # get bytes of the character
                char_bytes = o.encode('utf-8')
                for byte in char_bytes:
                    all_char_tokens.append(byte + all_chars.__len__())

        return all_char_tokens

tokenizer = ipa_tokenizer(ipa_dict, all_chars)

vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz")
from transformers import EncodecModel, AutoProcessor
EncodecModel = EncodecModel.from_pretrained("facebook/encodec_24khz")
EncodecProcessor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

def topk_sampling(seq, k=1, temp=1., first=0, banned=[]):
    # temp = temp/pow(pos+1,2)
    k = [(k if i < first else 1) for i in range(seq.shape[0] )]
    topk = torch.stack([torch.topk(seq[i], k[i], dim=-1).values[-1] for i in range(k.__len__())])
    logits = seq 
    # print(logits.shape, topk.shape)
    mask = logits < topk.unsqueeze(1)
    logits[mask]  = -float('Inf')
    if banned.__len__() > 0:
        # print(banned)
        logits[2] = 0.0;
    probs = torch.softmax(logits , dim=-1)** (1/temp)
    return torch.multinomial(probs, num_samples=1)

# def topk_sampling(
#     logits, k=1, temp=1
# ):
#     out = logits
#     if temp == 0.0 or k == 1:
#         out = torch.argmax(out, -1, keepdim=True)
#         return out
    
#     logits = logits.norm(dim=-1, keepdim=True)
    
#     min = out.min(-1).values
#     max = out.max(-1).values

#     dart = torch.rand(1, device=out.device)
#     dart = 1-temp*(dart*(1-dart))
#     dart = (dart*(max-min)+min)

#     out = (out - dart).abs()
#     out = torch.argmin(out, -1, keepdim=True)

#     return out


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from functools import partial


from typing import List, Optional, Tuple

from fla.layers.gla import GatedLinearAttention

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
from torch import nn

class MultiEmbedding(nn.Module):
    """
    Stacks multiple homogeneous embedding (same size, same embedding dim.) into one single weight.
    """

    def __init__(self, n_level, n_emb, d_emb, padding_idx=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(n_level, n_emb, d_emb, requires_grad=True)
        )
        self.n_level = n_level
        self.padding_idx = padding_idx
        nn.init.normal_(self.weight)

    def forward(self, idx):
        emb_fn = partial(nn.functional.embedding, padding_idx=self.padding_idx)
        return torch.vmap(emb_fn)(idx, self.weight)



def exists(x):
    return x is not None


def scaled_dot_product_attention(query, key, value, mask=None):
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    if exists(mask):
        attn_weight.masked_fill_(~mask, -torch.finfo(attn_weight.dtype).max)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value, attn_weight


class SinPos(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        exp = torch.arange(self.dim // 2, device=x.device)
        exp = 2 * exp / (self.dim)
        exp = rearrange(exp, "e -> 1 e")
        x = rearrange(x, "p -> p 1")
        pos = x * torch.pow(10000, -exp)
        pos = torch.cat((pos, pos + math.pi / 2), dim=1)
        pos = torch.sin(pos)
        return pos


class BlindCrossAttention(nn.Module):
    def __init__(
        self,
        q_dim,
        k_dim,
        att_dim,
        heads,
        pos_net,
        dropout=0.0,
        pos_dim=64,
        rotary=True,
    ):
        super().__init__()
        self.q = nn.Linear(q_dim, att_dim)
        self.k = nn.Linear(k_dim, att_dim)
        self.v = nn.Linear(k_dim, att_dim)
        self.pos_net = pos_net
        self.pos_embed = SinPos(pos_dim)
        assert att_dim % heads == 0
        self.ln_q = nn.LayerNorm(att_dim)
        self.ln_k = nn.LayerNorm(att_dim)
        self.ln_v = nn.LayerNorm(att_dim)
        # self.rotary = RotaryEmbedding((att_dim // heads) // 2) if rotary else None
        self.dropout_att = nn.Dropout(dropout)

    def forward(
        self,
        q,
        k,
        mask=None,
        time_step=None,
        **kwargs,
    ):
        q = self.ln_q(self.q(q))
        v = self.ln_v(self.v(k))
        k = self.ln_k(self.k(k))

        q, k, v = map(lambda x: rearrange(x, "b n d -> b 1 n d"), (q, k, v))
        b, h, j, d = k.shape
        pos = self.pos_embed(torch.arange(j, device=k.device))
    
        if mask is not None:
            mask = mask.masked_fill(~mask, -torch.finfo(q.dtype).max)
        pos = pos.to(q.dtype)
        k = k.to(q.dtype)
        v = v.to(q.dtype)
        x, att1 = scaled_dot_product_attention(q, k, pos, mask=mask)
        x = rearrange(x, "b 1 n d -> b n d")
        x,kwargs = self.pos_net((x, kwargs))
        x = rearrange(x, "b n d -> b 1 n d")
        pos = rearrange(pos, "n d -> 1 1 n d")
        x, att2 = scaled_dot_product_attention(x, pos, v, mask=mask)
        att = torch.cat((att1, att2), dim=1)
        
        x = rearrange(x, "b 1 n d -> b n d")
        return x, att, kwargs

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, rotary=True):
        super().__init__()
        self.qkv = nn.Linear(dim, 3*dim)
        assert dim % heads == 0
        self.heads = heads
        self.rotary = RotaryEmbedding((dim // heads) // 2) if rotary else None
    def forward(self, x, mask=None):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )
        if self.rotary is not None:
            q, k = map(self.rotary.rotate_queries_or_keys, (q, k))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = rearrange(y, "b h n d -> b n (h d)")
        return y

class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.p_in = nn.Linear(d_model, (d_model * 4 // 3) * 2)
        self.p_out = nn.Linear(d_model * 4 // 3, d_model)
        self.Silu = nn.SiLU()

    def forward(self, x):
        xx = self.p_in(x)
        *shape, last = xx.shape
        xx = xx.view(*shape, 2, last // 2)
        a,b = self.Silu(xx[:, :, 0]), xx[:, :, 1]
        return self.p_out(a*b)




class MixingBlock(torch.nn.Module):
    def __init__(self, tmix: GatedLinearAttention, cmix: Callable, norm: Callable, dropout=0.):
        super().__init__()
        self.tmix = tmix()
        self.cmix = cmix()
        self.norm1 = norm()
        self.norm2 = norm()
        self.drop = nn.Dropout(dropout)

    def forward(self, xkwargs):
        x, kwargs = xkwargs
        ax = self.tmix(self.norm1(x), **kwargs)
        if(type(ax) == tuple):
            outx,_,_ = ax
        else:
            outx = ax
        x = outx + x
        x = self.cmix(self.norm2(x)) + x
        x = self.drop(x)
        
        return x, kwargs
    

def exists(x):
    return x is not None




def undelay_rvq(extended_code):
    q, _, n = extended_code.shape
    extended_code = torch.nn.functional.pad(extended_code.transpose(0,1).reshape(_,-1),[0,q,0,0]).reshape(_,q,-1).transpose(0,1)
    return extended_code

def delay_rvq(extended_code):
    q, _, n = extended_code.shape
    extended_code = torch.nn.functional.pad(extended_code.transpose(0,1).reshape(_,-1),[0,-q,0,0]).reshape(_,q,-1).transpose(0,1)
    
    return extended_code


class AttentiveGLA(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_context: int,
        heads: int,
        dropout_att: float=0.0,
        dropout: float=0.,
        d_blind: int=None,
        max_batch: int = 8
    ):
        super().__init__()
        if d_blind is None:
            d_blind=d_model
        block = lambda d, h, i: MixingBlock(lambda: GatedLinearAttention(hidden_size=d, num_heads=h, mode="fused_recurrent", expand_k=1, expand_v=2, layer_idx=i),
                                                lambda: SwiGLU(d),
                                                lambda: nn.LayerNorm(d),
                                                    dropout=dropout,
                                                )
        self.encoder = nn.Sequential(*[block(d_model, heads, i) for i in range(n_layer)])
        self.decoder = nn.Sequential(*[block(d_model, heads, i + n_layer) for i in range(n_layer)])
        self.cross_att = BlindCrossAttention(
                d_model, d_context, d_model, 1, block(d_blind, heads, n_layer*2), dropout_att, pos_dim=d_blind)
            


    

    def forward(self, x, x_enc, time_step, state):

        x,kwargs = self.encoder((x, {"past_key_values":state, "use_cache":True}))
        
        y, att, kwargs = self.cross_att(x, x_enc, time_step=time_step, **kwargs)
        y = y+ x
        y,kwargs = self.decoder((y, kwargs))
        

        return y, att, state




class TextEncoder(torch.nn.Module):
    def __init__(
            self,
            dim:int,
            heads:int,
            n_layers=4,
            dropout=0.0,
            ):
        super().__init__()
        self.sa = torch.nn.Sequential(
                *[
                    MixingBlock(lambda: SelfAttention(dim, heads),
                                lambda: SwiGLU(dim),
                                lambda: torch.nn.LayerNorm(dim),
                                dropout)
                    for _ in range(n_layers)
                    ]
                )

    def forward(self, x, mask=None):
        
        x,_ = self.sa((x, {"mask":mask}))
        return x
    
jobs = []


class DecodeJob():
    def __init__(self, embed, enc, state, callback, k, temp, first_greedy, stop, bid,n_special_token_in, guided = None):
        self.embed = embed
        self.callback = callback
        self.enc = enc
        self.t = 0
        self.qs = []
        self.k = k
        self.temp = temp
        self.first_greedy = first_greedy
        self.stop_token = stop
        self.finished = False
        self.batchid = bid
        self.n_special_token_in = n_special_token_in
        self.buffer = []
        self.guided = guided
        print("state", state)
        self.state = 0
        if state is not None:
            mystate = torch.load(state)
            from tqdm import tqdm
            for i in tqdm(range(layers + 1)):
                globalstate[i][0][self.state] = mystate[i].to("cuda")

            globalstate[layers][0][self.state]*=0.0
        else:
            for i in range(layers + 1):
                globalstate[i][0][self.state] *= 0.0

        # self.state = [s.detach() for s in self.state]
        # self.state[-1] *= 0.0

        # for i in range(6):
        #     self.state[i] *= 0

    def time_step(self):
        self.t += 1
        return self.t-1
    

    async def finish(self):
        self.finished = True
        
        # torchaudio.save(f"./test{self.batchid}.wav", wav, 24000)
        if(self.guided is not None):
            self.callback(self.state)
            gd = undelay_rvq(self.guided)[:,0]

            # print(gd.shape)
            wav = vocos.decode(vocos.codes_to_features(rvq.cpu()), bandwidth_id=torch.tensor(1))

            # print(wav.shape)
            torchaudio.save(f"./test{self.batchid}.wav", wav, 24000, format="wav")
        if self.buffer.__len__() != 0:
            rvq = torch.stack(self.buffer,dim=1)
            
            
            wav = vocos.decode(vocos.codes_to_features(rvq.cpu()), bandwidth_id=torch.tensor(1))
            # audio = Audio(wav, rate=24000)
            # print(wav.shape)
            self.callback.write_audio_chunk(0, wav.t())

        torch.save({i:globalstate[i][0][self.state].clone() for i in range(layers + 1)}, f"last.voice")
        
    

    async def sample(self, logits, emb,rvqmix):

       
        q_sampled = topk_sampling(logits, k=self.k, temp=self.temp, first=self.first_greedy, banned =self.stop_token if self.t < 8 else [])
        
        for i in range(q_sampled.shape[0]):
            if(self.qs.__len__() == i):
                self.qs += [[]]
            if(i < self.t):
                self.qs[i].append(q_sampled[i])
        if(q_sampled.shape[0] <= self.t):
            self.buffer .append(((torch.stack([r.pop(0) for r in self.qs], dim=0).squeeze(-1)  - self.n_special_token_in).clamp_min(0)))

            if(self.buffer.__len__() > 16):
                rvq = torch.stack(self.buffer,dim=1)
        
                print(rvq.shape)
                wav = vocos.decode(vocos.codes_to_features(rvq.cpu()), bandwidth_id=torch.tensor(1))
                # audio = Audio(wav, rate=24000)
                print(wav.shape)
                self.callback.write_audio_chunk(0, wav.t())
                self.buffer = self.buffer[16:]
        # print(q_sampled.shape, self.stop_token.shape)
        if (q_sampled.squeeze() == self.stop_token.squeeze()).prod(dim=0):
            await self.finish()

        y_embd = emb(q_sampled)
        y_embd = rearrange(y_embd, "q n d -> n (q d)")
        y_embd = rvqmix(y_embd)
        self.embed = y_embd.unsqueeze(0)

class CacheObject():
    def __init__(self, state, store = True):
        self.state = state
        self.store = store
    def update(self, item, position, _):
        if(self.store):
            self.state[position][0][:item[-1].shape[0]] = item[-1]

    def apply_grads(self, loss):
        for i in range(layers + 1):
            self.state[i] = ((self.state[i][0] - self.state[i][0].grad * loss).detach().requires_grad_(True),)
    # *unpack
    def __iter__(self):
        return iter(self.state)
    
    
    # subscript
    def __getitem__(self, key):
        return self.state[key]
    
    
jobbuffera = [*[(torch.zeros(16,8,128, 256, device="cuda", requires_grad=False),) for i in range(layers)],  (torch.zeros(16,8,32,64, device="cuda", requires_grad=False),)]


globalstate = CacheObject(jobbuffera, store=True)

    
class Lina(nn.Module):
    def __init__(
        self,
        d_model: int,
        quant_layer: List[int],
        n_codebook: int,
        n_special_token_in: int,
        n_special_token_out: int,
        n_txt_vocab: int,
        tie_embed: bool = False,
        d_context: Optional[int] = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.999),
        max_batch: int = 8
    ):
        super(Lina, self).__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas

        self.n_quant = len(quant_layer)
        self.n_codebook = n_codebook
        self.n_special_token_in = n_special_token_in
        self.n_special_token_out = n_special_token_out
        self.n_txt_vocab = n_txt_vocab
        self.n_target_vocab = n_codebook + n_special_token_out  # no padding token

        self.txt_encoder = TextEncoder(**{
            "dim": 1024,
            "heads": 8,
            "n_layers": 4,
            "dropout": 0
        })

        self.txt_embed = nn.Embedding(
            n_txt_vocab,
            d_context if exists(d_context) else d_model,
            padding_idx=0,
        )
        self.rvq_embed = MultiEmbedding(self.n_quant, n_codebook + n_special_token_in, d_model, padding_idx=0) 
        self.rvqmix = nn.Linear(d_model*4, d_model, bias=False)

        self.attentive_rnn = AttentiveGLA(**{
        "d_model": 1024,
        "d_context": 1024,
        "heads": 8,
        "dropout_att": 0,
        "dropout": 0.,
        "n_layer": 12,
        "d_blind": 256,
        
        })

        self.logits_head = nn.Sequential(
            nn.Linear(d_model, self.n_quant*self.n_target_vocab, bias=False),
            nn.Sigmoid(),
            nn.Linear(self.n_quant*self.n_target_vocab, self.n_quant*self.n_target_vocab, bias=False)
        )
        if tie_embed:
            self.logits_head.weight = self.rvq_embed.weight

        self.jobs = []


    async def decodeJob(self):

        self.jobs = [j for j in self.jobs if j.finished == False]
        
        if(self.jobs.__len__()==0):
            return

        emb = [j.embed for j in self.jobs]
        enc = [j.enc for j in self.jobs]
        ts = [torch.tensor(j.time_step(), dtype=torch.long, device=emb[0].device) for j in self.jobs]

        emb = torch.cat(emb,0)

        maxenc = max([e.shape[1] for e in enc])
        enc = [torch.nn.functional.pad(e,[0,0,0,maxenc-e.shape[1]]) for e in enc]
        # print(*[e.shape for e in enc])
        enc = torch.cat(enc,0)
        ts = torch.stack(ts)

        


        y_embd, att, st = self.attentive_rnn(emb, enc, time_step=ts, state=globalstate)

        # st = list(zip(*st))

        # for i, j in enumerate(self.jobs):
        #     j.state = st[i]
        logits = self.logits_head(y_embd)
        b, n, l = logits.shape
        logits = logits.view(b, n, self.n_quant, -1)
        logits = logits.squeeze(1)
        # print(logits.shape)
        jjobs = []
        for i, q in enumerate(logits):   
            # print(q.shape, i)  
            
            jjobs = jjobs + [self.jobs[i].sample(logits[i], self.rvq_embed, self.rvqmix)]
            
        
        await asyncio.gather(*jjobs)

        # if exists(prompt) and t < p_len:
        #     y_embd = prompt[:, [t]]
        # else:
    

    
    def generate_batch(
        self,
        x: torch.Tensor,
        batch_size: int=1,
        device: str = "cpu",
        k: int = 100,
        first_greedy_quant: int = 1,
        temp: float = 1.0,
        callback = None,
        voice = None,
        guided = None
    ):
        
        x = repeat(x, "n -> b n", b=batch_size).to(device)
        stop_token = torch.ones(self.n_quant, 1, 1, device=device) * 2
        y_start = torch.ones(self.n_quant, batch_size, 1, device=device).long()

        x_embd = self.txt_embed(x)
        y_embd = self.rvq_embed(y_start)
        y_embd = rearrange(y_embd, "q b n d -> b n (q d)")
        y_embd = self.rvqmix(y_embd)


        x_mask = torch.ones_like(x, device=device).bool()

        x_enc = self.txt_encoder(x_embd, x_mask) 

        
        job = DecodeJob(y_embd,x_enc,voice,callback,k,temp, first_greedy=first_greedy_quant,stop=stop_token, bid=self.jobs.__len__(), n_special_token_in=self.n_special_token_in, guided=guided)
        self.jobs.append(job)
        
        return job

    def state_backwards(self, x, guided, sto):
        x_embed = self.txt_embed(x.to(self.txt_embed.weight.device)).unsqueeze(0)
        y_embed = self.rvq_embed(guided)
        
        y_embed = rearrange(y_embed, "q b n d -> b n (q d)")
        y_embed = self.rvqmix(y_embed)
        x_mask = torch.ones_like(x).bool().to(x_embed.device)
        x_enc = self.txt_encoder(x_embed, x_mask) if exists(self.txt_encoder) else x_embed
        
       
        self.attentive_rnn.train()
        
        y, att, state = self.attentive_rnn(y_embed, x_enc, time_step=0, state=sto)

        lt = self.logits_head(y)



        # print(lt.shape, guided.shape)

        lt = lt.view(-1, 4, self.n_target_vocab)[:-1,1:].transpose(0,1).reshape(-1,self.n_target_vocab)

        loss = F.cross_entropy(lt, guided[1:,:,1:].reshape(-1), reduction="mean")
        loss.backward()
        print("Loss", loss.item())
        return sto, loss




model = Lina(
    **{
        "n_codebook": 1024,
        "n_special_token_in": 3,
        "n_special_token_out": 3,
        "n_txt_vocab": 305,
        "d_context": 1024,
        "d_model": 1024,
        "quant_layer": [0, 1, 2, 3],
        "max_batch" : 8
    }
)

modelfile = torch.load("./exp.pt", map_location="cpu")
modelfile = modelfile
print(modelfile.keys())

model.load_state_dict(modelfile)



model = model.float().cuda()
# model.attentive_rnn = torch.compile(model.attentive_rnn)

import torchaudio


async def runloop():
    print("starting runloop...")
    while True:

        await model.decodeJob()
        # let other tasks run
        await asyncio.sleep(0.00001)



import asyncio
from aiohttp import web



async def handleGet(request, query=None):
    with torch.inference_mode():
        if request.path == "/":
            # open test.html
            # replace VOICES with [<option >v</option> for v in files ending in voice]
            with open("test.html") as f:
                text = f.read()
                text = text.replace("VOICES", f"""
                                        {
                                            "".join([f'<option value="{v}">{v}</option>' for v in os.listdir() if v.endswith("voice")])
                                        }
                                    """)
            return web.Response(text=text, content_type="text/html")
        else:
            # get from query
            query = request.query if query is None else query
            prompt = query["prompt"] if "prompt" in query else base64.encode("The quick brown fox jumps over the lazy dog").decode("utf-8")
            voice = query.get("voice", None)
            if(voice == "random"):
                voice = None
            # voice = None
            temp = query.get("temp", 1.0)
            temp = float(temp)
            print(prompt)
            #atob
            txt = base64.b64decode(prompt).decode("utf-8")

            basicstreambuffer = io.BytesIO()

            s = torchaudio.io.StreamWriter(basicstreambuffer, "wav")
            s.add_audio_stream(
                        sample_rate=24000,
                        num_channels=1,
                    )
            s.open()
            jobs = []
            
            for t in txt.split("."):
                txta = "\x16" + t + "\x17"
                txta = torch.LongTensor(tokenizer.encode(txta))
                # print(request)
                headers = {
                    'Content-Type': 'audio/x-wav',
                }

                jobs.append([txta, temp, voice, headers, s])
            job = None
            def jobstart(jobtostart):
                txta, temp, _, headers, s = jobtostart
                
                return model.generate_batch(
                                txta,
                                k=32, #topk sampling
                                temp=temp, #temperature sampling
                                first_greedy_quant=1, #first residual quantizer to be greedy-sampled
                                device="cuda",
                                callback=s,
                                voice=voice)

            job = jobstart(jobs.pop(0))
            # while job.finished == False:
            #     # sleep for 1 second
            #     await asyncio.sleep(1)

            res = web.StreamResponse(headers=headers, status=200)
            await res.prepare(request)

            frompos = 0
            while True:

                if job.finished:
                    if jobs.__len__() == 0:
                        break
                    job = jobstart(jobs.pop(0))
                buffer = basicstreambuffer.getvalue()
                print(buffer.__len__(), frompos)
                try:
                    await res.write(buffer[frompos:])
                    frompos = basicstreambuffer.tell()
                except:
                    print("Error")

                await asyncio.sleep(0.001)


            
            return res
        # return web.Response(text="OK")
from aiohttp.abc import Request
from datasets import Audio
def saveState(state):
        global ret
        # create a psuedo random file name
        fname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        torch.save(state, f"state{fname}.voice")
        ret = web.Response(text=f"state{fname}.voice")

async def cloneVoice(request:Request):
    
    # get form data file from request
    file = (await request.multipart())
    file = await file.next()
    data = await file.read()
    # save data as wav file
    mymodel = model.train(True)
    # audiodata = "temp.wav"
    data = io.BytesIO(data).getvalue()
    # audiodata = Audio(sampling_rate=24000).encode_example(data)
    
    data = torchaudio.load(io.BytesIO(data), frame_offset=0, num_frames=-1, normalize=True)
    # with open("temp.wav", "wb") as f:
    #     f.write(data)

    # get current audio sample rate
    current_sample_rate = data[1]

    curdata = data[0]  
    
    st = [*[(torch.rand(1,8,128, 256, device="cuda", requires_grad=True),) for i in range(layers)], (torch.rand(1,8,128,256, device="cuda", requires_grad=True),)]
    sto = CacheObject(st, store=False)
    loss = 9 
    optimizer = torch.optim.Adam([s[0] for s in sto], lr=5)

    # cut the audio into 5 second segments
    pieces = curdata.shape[1] // (current_sample_rate * 5)
    piecearray = torch.split(curdata, current_sample_rate * 5, dim=1)

    codearray = []
    txtarray = []
    from tqdm import tqdm
    for piece in tqdm(piecearray):

        dataw,*_ = torchaudio.transforms.Resample(current_sample_rate, 16000)(piece)
        data,*_ = torchaudio.transforms.Resample(current_sample_rate, 24000)(piece)
        # audio_sample = audiodata
        codes = delay_rvq(EncodecModel.encode(**EncodecProcessor(raw_audio=data, sampling_rate=24000, return_tensors="pt"), bandwidth=3).audio_codes[0].transpose(0,1)[:4]).cuda() + 3
        txt = torch.LongTensor(tokenizer.encode("\x16"+whisperprocessor.batch_decode(whisper.generate(whisperprocessor(dataw, sampling_rate=16000, return_tensors="pt").input_features ), skip_special_tokens=True)[0]+"\x17"))
        codearray.append(codes)
        txtarray.append(txt)
    # print(transcription)

        
  
    while loss > 2.5:
        # start = random.randint(0, max(codes.shape[2],codes.shape[2]-512))
        index = random.randint(0, len(codearray)-2)
        codes = codearray[index]
        txt = txtarray[index]
        sto, loss = mymodel.state_backwards(txt, codes, sto)

        optimizer.step()

        optimizer.zero_grad()

        


    saveState(sto)
    codes = undelay_rvq(codes)
    wav = vocos.decode(vocos.codes_to_features(codes.cpu()-3), bandwidth_id=torch.tensor(1))
    # # load wav file and convert to tensor

    # headers = {
    #     'Content-Type': 'audio/x-wav',
    # }

    # res = web.StreamResponse(headers=headers)
    # await res.prepare(request)

    # outbytes = io.BytesIO()
    torchaudio.save("example.wav", wav, 24000, format="wav")
    # # send the audio file back
    # await res.write(outbytes.getvalue())
    # return res
    while ret is None:
        await asyncio.sleep(0.1)
    return ret

async def handlePost(request):
    body = await request.read()
    bodyjson = json.loads(body)
    print(bodyjson)
    text = bodyjson["input"]
    voice = bodyjson.get("voice", None)
    query = {"prompt":base64.b64encode(text.encode("utf-8")).decode("utf-8"), "voice":voice, "temp": bodyjson.get("temp", 1.0)}

    return await handleGet(request, query)

app = web.Application()

async def getModels(request):
    return web.Response(text=json.dumps({"models":[
        os.path.basename(f) for f in os.listdir() if f.endswith("voice")
    ]}))

app.router.add_route("GET",'/v1/audio/speech/', handleGet)
app.router.add_route("GET",'/', handleGet)
app.router.add_route("POST",'/v1/audio/voice/clone', cloneVoice)
app.router.add_route("POST",'/v1/audio/speech', handlePost)
app.router.add_route("GET",'/v1/audio/random_voice', handleGet)
app.router.add_route("GET",'/v1/models', getModels)

async def run():
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8085)
    print("Server started at http://0.0.0.0:8085")

    # run loop
    task = runloop()
    webo = site.start()

    # do both simultaneously
    tog = asyncio.gather(task, webo)

    await tog


asyncio.run(run())