from TCN.tcn import TemporalConvNet
from dataloader import KTData
from torch import nn
import numpy as np
import torch
from torch.nn import functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

#Storage = []
#Embeddings = {}

class SSAKT(nn.Module):
    def __init__(self, skill_num, embed_dim, num_channels, nhead, kernel_size, d_k, d_v, d_inner=2048,
    problem_num=0,is_pmodel=False ,qa_binary=False,dropout=0.1, emb_dropout=0.1):
        super(SSAKT, self).__init__()
        self.problem_num = problem_num
        self.is_pmodel = is_pmodel
        self.qa_binary = qa_binary
        self.skill_num = skill_num
        self.qa_encoder = nn.Embedding(skill_num * 2 + 1, embed_dim)
        self.q_encoder = nn.Embedding(skill_num + 1, embed_dim)
        # qa_seperate TODO
        if problem_num > 0 and is_pmodel:
            self.p_emb = nn.Embedding(problem_num + 1, embed_dim)
            self.p_enc = EncoderLayer(embed_dim, d_inner, 1, d_k, d_v, dropout)
            self.problem_encoder = nn.Embedding(problem_num + 1, embed_dim)
            self.qa_diff = nn.Embedding(skill_num * 2 + 1, embed_dim)
            self.q_diff = nn.Embedding(skill_num + 1, embed_dim)
            trf_input_dim = embed_dim + embed_dim
            self.qalinear = nn.Linear(trf_input_dim, embed_dim)
            self.qlinear = nn.Linear(trf_input_dim, embed_dim)
            trf_input_dim = embed_dim
        else:
            trf_input_dim = embed_dim
        self.tcn = TemporalConvNet(trf_input_dim, num_channels, kernel_size, dropout=dropout)
        self.q_tcn = TemporalConvNet(embed_dim, num_channels, kernel_size, dropout=dropout)
        self.activation0 = nn.ReLU()
        self.lstm = nn.LSTM(trf_input_dim, num_channels[-1])
        self.q_lstm = nn.LSTM(trf_input_dim, num_channels[-1])
        self.context_lstm = nn.LSTM(trf_input_dim, num_channels[-1])
        self.linear = nn.Linear(num_channels[-1], embed_dim)
        self.linear1 = nn.Linear(num_channels[-1], embed_dim)
        self.linear2 = nn.Linear(num_channels[-1], embed_dim)
        self.drop = nn.Dropout(emb_dropout)
        self.tcn_input_linear = nn.Linear(trf_input_dim + trf_input_dim, trf_input_dim)
        #self.qa_enc = MultiHeadAttention(trf_input_dim, nhead, d_k, d_v)
        self.qa_enc = EncoderLayer(embed_dim, d_inner, nhead, d_k, d_v, dropout)
        self.q_enc = EncoderLayer(embed_dim, d_inner, nhead, d_k, d_v, dropout)
        self.out_enc = DecoderLayer(embed_dim, d_inner, nhead, d_k, d_v, dropout)
        # self.activation1 = nn.ReLU()
        self.theta = nn.Parameter(torch.rand(1))
        self.outlayer = nn.Sequential(nn.Linear(embed_dim + trf_input_dim, 1024), nn.ReLU(), nn.Dropout(dropout), 
                                      nn.Linear(1024, 256),  nn.ReLU(), nn.Dropout(dropout), nn.Linear(256,1))
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
    def forward(self, q, qa, problem, Q, test=False):
        if self.qa_binary:
            qa = qa//(self.skill_num+1)
        reg_loss = 0
        q_emb = self.drop(self.q_encoder(q))
        qa_emb = self.drop(self.qa_encoder(qa))
        if self.qa_binary:
            qa_emb +=  q_emb
        if self.problem_num > 0 and self.is_pmodel:
            p_embedding = self.problem_encoder(problem)
            p_emb = self.p_emb(problem)
            #p_enc_emb, p_attn = self.p_enc(p_emb)
            #p_attn = p_attn.squeeze(1)
            q_diff = self.q_diff(q)
            qa_diff = self.qa_diff(qa)
            qa_emb = p_embedding * qa_diff + qa_emb#self.qalinear(torch.cat([qa_embedding, p_embedding], -1))
            q_emb = p_embedding * q_diff + q_emb#self.qlinear(torch.cat([q_embedding, p_embedding], -1))
        tcn_input = qa_emb
        
        y, (h_n) = self.lstm(tcn_input.transpose(0,1))
        y = y.transpose(0,1)
        y = self.activation0(self.linear(y))
        qa_embedding, _ = self.qa_enc(y)

        context, _ = self.context_lstm(qa_embedding.transpose(0,1))
        context = context.transpose(0,1)
        #context = self.tcn(qa_embedding.transpose(1,2)).transpose(1,2)
        context = self.activation0(self.linear2(context))

        #q_embedding, _ = self.q_enc(q_emb)
        #if self.problem_num > 0 and self.is_pmodel:
        #    qa_embedding += torch.matmul(p_attn, qa_emb)
        '''tcn_input = torch.cat([torch.zeros((qa_embedding.shape[0],1,qa_embedding.shape[2])).to(device), 
                                qa_embedding[:,:-1,:]], 1)
        tcn_input = torch.cat([q_embedding,tcn_input], -1)
        tcn_input = self.tcn_input_linear(tcn_input)'''
        #query = self.tcn(q_embedding.transpose(1,2)).transpose(1,2)
        
        #q_out = self.q_tcn(q_embedding.transpose(1,2)).transpose(1,2)
        #tcnout = self.tcn(tcn_input.transpose(1,2)).transpose(1,2)
        #y = self.activation0(self.linear(tcnout))
        #q_seq_emb = q_emb
        q_seq_emb , _ = self.q_lstm(q_emb.transpose(0,1))
        q_seq_emb = q_seq_emb.transpose(0,1)
        q_seq_emb = self.activation0(self.linear1(q_seq_emb)) + q_emb
        t_out, attn1, attn2 = self.out_enc(q_seq_emb, qa_embedding)
        # y.shape -> batch_size * seqlen * embed_dim
        
        y = torch.cat([torch.zeros((y.shape[0],1,y.shape[2])).to(device), y[:,:-1,:]], 1)
        context = torch.cat([torch.zeros((context.shape[0],1,context.shape[2])).to(device), context[:,:-1,:]], 1)
        ks = t_out + context
        y = torch.cat([q_seq_emb, ks], -1)
        # y = self.activation1(y)
        y = self.outlayer(y)
        if test:
            pass
        return y, reg_loss

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, 0)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, dec_output, enc_output, 1)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, current=0):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, current=current)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, d_k, d_v, dropout=0.1,bias=True):
        super().__init__()
        self.bias = bias
        self.d_k, self.d_v, self.nhead = d_k, d_v, nhead
        self.w_q = nn.Linear(d_model, nhead * d_k, bias=bias)
        self.w_k = nn.Linear(d_model, nhead * d_k, bias=bias)
        self.w_v = nn.Linear(d_model, nhead * d_v, bias=bias)
        self.fc = nn.Linear(nhead * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.attention = ScaledDotProductAttention(d_k**0.5)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, q, k, v, current=1):
        '''
        input N * seqlen * d_model
        q -> query, key
        qa -> value
        '''
        residual = q
        d_k, d_v, n_head = self.d_k, self.d_v, self.nhead
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.w_q(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_k(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_v(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # batch_size * nhead * seq_len * d_model
        mask = torch.tril(torch.ones((len_q, len_q)),diagonal=-current).to(device)
        q, attn = self.attention(q, k, v, current,mask=mask)# batch_size * nhead * seq_len * d_v

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        # sz_b * len_q * (nhead * d_v) -> sz_b * len_q * d_model

        q = q + residual
        #q = torch.cat([q, residual], 2)

        q = self.layernorm(q)
        return q, attn # 


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature,n_heads=8,attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.gammas = nn.Parameter(torch.rand(1))

    def attention(self, q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
        """
        This is called by Multi-head atention object to find the values.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / \
            d_k  # BS, 8, seqlen, seqlen
        bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

        x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
            scores_ = scores_ * mask.float().to(device)
            distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
            disttotal_scores = torch.sum(
                scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1
            position_effect = torch.abs(
                x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen
            # bs, 8, sl, sl positive distance
            dist_scores = torch.clamp(
                (disttotal_scores-distcum_scores)*position_effect, min=0.)
            dist_scores = dist_scores.sqrt().detach()
        m = nn.Softplus()
        gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1
        # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
        total_effect = torch.clamp(torch.clamp(
            (dist_scores*gamma).exp(), min=1e-5), max=1e5)
        scores = scores * total_effect

        scores.masked_fill_(mask == 0, -1e32)
        scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
        if zero_pad:
            pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
            scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
        scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output, scores

    def forward(self, q, k, v, current, mask=None):
        #return self.attention(q, k, v,self.temperature, mask,self.dropout,current==1,self.gammas)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        '''x1 = torch.arange(q.size(-2)).float().unsqueeze(1).to(device)
        x2 = x1.transpose(0, 1).contiguous()
        time_decay = (x1-x2 + 1).masked_fill(mask==0, 1)
        time_decay = torch.log(time_decay)
        attn = attn - self.gammas * time_decay'''
        attn = attn.masked_fill(mask == 0, -1e32).to(device)
        attn = self.dropout(F.softmax(attn.masked_fill(mask==0,-1e32), dim=-1))
        if current == 1:
            pad_zero = torch.zeros(attn.size(0), attn.size(1), 1, attn.size(-1)).to(device)
            attn = torch.cat([pad_zero, attn[:, :, 1:, :]], dim=-2)
        attn = attn * mask
        output = torch.matmul(attn, v)

        return output, attn