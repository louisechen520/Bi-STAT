import torch
import torch.nn as nn
from models.common_layer import AttentionLayer, LayerNorm, _gen_embedding


class AdaptiveTransformer(nn.Module):
    """
    An adaptive Transformer module. 

    """
    def __init__(self, epsilon, flag, hidden_size, max_step, num_heads, seq_length, input_dropout=0.0, layer_dropout=0.0, 
                 relu_dropout=0.0, dhm=False):
                         
        super(AdaptiveTransformer, self).__init__()
        
        self.position_embedding = _gen_embedding(seq_length, hidden_size)
        self.recurrent_embedding = _gen_embedding(max_step, hidden_size)

        self.max_step = max_step
        self.dhm = dhm

        self.att = AttentionLayer(flag, hidden_size, num_heads, layer_dropout, relu_dropout)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        if(self.dhm):
            self.dhm_fn = DHM_basic(epsilon, hidden_size)

    def forward(self, inputs, STE):

        #Add input dropout
        x = self.input_dropout(inputs)

        if(self.dhm):
            x, (remainders,n_updates) = self.dhm_fn(x, inputs, STE, self.att, self.position_embedding, self.recurrent_embedding, self.max_step)
            return x, (remainders,n_updates)
        else:
            for l in range(self.recurrent_step):
                x += self.position_embedding[:, :inputs.shape[1], :].type_as(inputs.data)
                x += self.recurrent_embedding[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
                x = self.enc(x)
            return x, None


class DHM_basic(nn.Module):
    def __init__(self, epsilon, hidden_size):
        super(DHM_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size,1)  
        self.p.bias.data.fill_(1) 
        self.threshold = 1 - epsilon

    def forward(self, state, inputs, STE, fn, pos_enc, recur_enc, max_step, encoder_output=None):
        B,T,N,D = state.shape[0],state.shape[1],state.shape[2],state.shape[3]
        state = state.view(B*T,N,D)
        inputs = inputs.view(B*T,N,D)
        # init
        halting_probability = torch.zeros(B*T,N).cuda()
        remainders = torch.zeros(B*T,N).cuda()
        n_updates = torch.zeros(B*T,N).cuda()
        previous_state = torch.zeros_like(inputs).cuda()
        step = 0
        # Dynamic Halting Module
        while( ((halting_probability<self.threshold) & (n_updates < max_step)).byte().any()):
            
            state = state + pos_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            state = state + recur_enc[:, step, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
            
            p = self.sigma(self.p(state)).squeeze(-1).cuda()

            still_running = (halting_probability < 1.0).float().cuda()

            new_halted = (halting_probability + p * still_running > self.threshold).float().cuda() * still_running

            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            halting_probability = halting_probability + p * still_running

            remainders = remainders + new_halted * (1 - halting_probability)

            halting_probability = halting_probability + new_halted * remainders

            n_updates = n_updates + still_running + new_halted

            update_weights = p * still_running + new_halted * remainders

            state = state.view(B,T,N,D)

            if(encoder_output):
                state, _ = fn((state,encoder_output))
            else:
                state = fn(state, STE, None)

            state = state.view(B*T,N,D)

            previous_state = ((state.cuda() * update_weights.unsqueeze(-1).cuda()) + (previous_state.cuda() * (1 - update_weights.unsqueeze(-1)).cuda()))
            
            step+=1

        return previous_state.view(B,T,N,D), (remainders,n_updates)
