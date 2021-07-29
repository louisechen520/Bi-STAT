import torch
import torch.nn.functional as F
from models.spatial_adaptive_transformer import AdaptiveTransformer as SpatialAdaptiveTransformerModel
from models.temporal_adaptive_transformer import AdaptiveTransformer as TemporalAdaptiveTransformerModel
from models.common_layer import SpatialTransformerModel, TemporalTransformerModel

class STEmbModel(torch.nn.Module):
    def __init__(self, SEDims, TEDims, OutDims, device):
        super(STEmbModel, self).__init__()
        self.TEDims = TEDims
        self.FC_se1 = torch.nn.Linear(SEDims, OutDims)
        self.FC_se2 = torch.nn.Linear(OutDims, OutDims)
        self.FC_te1 = torch.nn.Linear(TEDims, OutDims)
        self.FC_te2 = torch.nn.Linear(OutDims, OutDims)
        self.device = device


    def forward(self, SE, TE):
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.FC_se2(F.relu(self.FC_se1(SE)))
        dayofweek = F.one_hot(TE[..., 0], num_classes = 7)
        timeofday = F.one_hot(TE[..., 1], num_classes = self.TEDims-7)
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(2).type(torch.FloatTensor).to(self.device)
        TE = self.FC_te2(F.relu(self.FC_te1(TE)))
        sum_tensor = torch.add(SE, TE)
        return sum_tensor


class EntangleModel(torch.nn.Module):
    def __init__(self, K, d):
        super(EntangleModel, self).__init__()
        D = K*d
        self.FC_xs = torch.nn.Linear(D, D)
        self.FC_xt = torch.nn.Linear(D, D)
        self.FC_h1 = torch.nn.Linear(D, D)
        self.FC_h2 = torch.nn.Linear(D, D)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = self.sigmoid(torch.add(XS, XT))
        H = torch.add((z* HS), ((1-z)* HT))
        H = self.FC_h2(F.relu(self.FC_h1(H)))
        return H


class STATModel(torch.nn.Module):
    def __init__(self, K, d, epsilon, hidden_size, max_step, T, N, dhm):
        super(STATModel, self).__init__()
        self.spatialAdaptiveTransformer = SpatialAdaptiveTransformerModel(epsilon, 'S', hidden_size, max_step, K, 
        N, input_dropout=0.0, layer_dropout=0.0, relu_dropout=0.0, dhm=dhm)

        self.temporalAdaptiveTransformer = TemporalAdaptiveTransformerModel(epsilon, 'T', hidden_size, max_step, K, 
        T, input_dropout=0.0, layer_dropout=0.0, relu_dropout=0.0, dhm=dhm)

        self.entangle = EntangleModel(K, d)

    def forward(self, X, STE, mask):
        HS, dhm_S = self.spatialAdaptiveTransformer(X, STE)
        HT, dhm_T = self.temporalAdaptiveTransformer(X, STE, mask)
        H = self.entangle(HS, HT)
        return torch.add(X, H), dhm_S, dhm_T

class STTModel(torch.nn.Module):
    def __init__(self, K, d):
        super(STTModel, self).__init__()
        self.spatialTransformer = SpatialTransformerModel(K, d)
        self.temporalTransformer = TemporalTransformerModel(K, d)
        self.entangle = EntangleModel(K, d)

    def forward(self, X, STE, mask):
        HS = self.spatialTransformer(X, STE, mask)
        HT = self.temporalTransformer(X, STE, mask)
        H = self.entangle(HS, HT)
        return torch.add(X, H)


class CrossAttentionModel(torch.nn.Module):
    def __init__(self, K, d):
        super(CrossAttentionModel, self).__init__()
        D = K * d
        self.FC_Q_Cross = torch.nn.Linear(D, D)
        self.FC_K_Cross = torch.nn.Linear(D, D)
        self.FC_V_Cross = torch.nn.Linear(D, D)
        self.FC_Out1 = torch.nn.Linear(D, D)
        self.FC_Out2 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X, STE_P, STE_Q):
        query = F.relu(self.FC_Q_Cross(STE_Q))
        key = F.relu(self.FC_K_Cross(STE_P))
        value = F.relu(self.FC_V_Cross(X))
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        query = torch.transpose(query, 2, 1)
        key = torch.transpose(torch.transpose(key, 1, 2), 2, 3)
        value = torch.transpose(value, 2, 1)
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = self.softmax(attention)
        X = torch.matmul(attention, value)
        X = torch.transpose(X, 2, 1)
        X = torch.cat(torch.split(X, X.shape[0]//self.K, dim=0), dim=-1)
        X = self.FC_Out2(F.relu(self.FC_Out1(X)))
        return X


class BISTAT(torch.nn.Module):
    def __init__(self, K, d, SEDims, TEDims, P, F, H, N, L, episilon, hidden_size, max_hop, dhm, device):
        super(BISTAT, self).__init__()
        D = K*d
        self.P = P
        self.F = F
        self.H = H
        self.L = L
        self.FC_1 = torch.nn.Linear(1, D)
        self.FC_2 = torch.nn.Linear(D, D)
        self.STEmb = STEmbModel(SEDims, TEDims, K*d, device)
        self.STATBlockEnc = torch.nn.ModuleList([STATModel(K, d, episilon, hidden_size, max_hop, P, N, dhm) for _ in range(self.L)])
        self.STATBlockDec1 = torch.nn.ModuleList([STATModel(K, d, episilon, hidden_size, max_hop, P, N, dhm) for _ in range(self.L)])
        self.STATBlockDec2 = torch.nn.ModuleList([STTModel(K, d) for _ in range(self.L)])

        self.CrossAttention1 = CrossAttentionModel(K, d)
        self.CrossAttention2 = CrossAttentionModel(K, d)

        self.FC_dec1_1 = torch.nn.Linear(D, D)
        self.FC_dec1_2 = torch.nn.Linear(D, 1)

        self.FC_dec2_1 = torch.nn.Linear(D, D)
        self.FC_dec2_2 = torch.nn.Linear(D, 1)

    def forward(self, X, SE, TE, flag):
        # input
        X = X.unsqueeze(3)
        X = self.FC_2(F.relu(self.FC_1(X)))
        
        # STE for the Historical, Present and Future condition
        STE = self.STEmb(SE, TE)
        STE_H = STE[:, : self.H]
        STE_P = STE[:, self.H: self.H + self.P]
        STE_F = STE[:, self.H + self.P: self.H + self.P + self.F]

        # dhm0 and dhm1 represent the remainders and the number of recurrent step in DHM 
        dhm0_S_enc = []
        dhm1_S_enc = []
        dhm0_T_enc = []
        dhm1_T_enc = []

        # output for the layer1 in the encoder, which is used for the present-historical cross-attention
        X_enc_out_L1, dhm_S, dhm_T = self.STATBlockEnc[0](X, STE_P, mask=True)
        dhm0_S_enc.append(dhm_S[0])
        dhm1_S_enc.append(dhm_S[1])
        dhm0_T_enc.append(dhm_T[0])
        dhm1_T_enc.append(dhm_T[1])

        # output from the last layers in the encoder, which is used for the future-present cross-attention
        for l in range(1, len(self.STATBlockEnc)):            
            X, dhm_S, dhm_T = self.STATBlockEnc[l](X, STE_P, mask=True)
            dhm0_S_enc.append(dhm_S[0])
            dhm1_S_enc.append(dhm_S[1])
            dhm0_T_enc.append(dhm_T[0])
            dhm1_T_enc.append(dhm_T[1])
        X_enc_out = X

        dhm0_S_enc = torch.stack(dhm0_S_enc, dim=0)
        dhm1_S_enc = torch.stack(dhm1_S_enc, dim=0)
        dhm0_T_enc = torch.stack(dhm0_T_enc, dim=0)
        dhm1_T_enc = torch.stack(dhm1_T_enc, dim=0)

        # the Future-Present Cross-Attention
        X_cross1_out = self.CrossAttention1(X_enc_out, STE_P, STE_F)

        # Future Decoder with DHM
        dhm0_S_dec = []
        dhm1_S_dec = []
        dhm0_T_dec = []
        dhm1_T_dec = []
        for net in self.STATBlockDec1:
            X_dec1_out, dhm_S, dhm_T = net(X_cross1_out, STE_F, mask=True)

            dhm0_S_dec.append(dhm_S[0])
            dhm1_S_dec.append(dhm_S[1])
            dhm0_T_dec.append(dhm_T[0])
            dhm1_T_dec.append(dhm_T[1])

        dhm0_S_dec = torch.stack(dhm0_S_dec, dim=0)
        dhm1_S_dec = torch.stack(dhm1_S_dec, dim=0)
        dhm0_T_dec = torch.stack(dhm0_T_dec, dim=0)
        dhm1_T_dec = torch.stack(dhm1_T_dec, dim=0)

        dhm0_S = dhm0_S_enc + dhm0_S_dec
        dhm1_S = dhm1_S_enc + dhm1_S_dec
        dhm0_T = dhm0_T_enc + dhm0_T_dec
        dhm1_T = dhm1_T_enc + dhm1_T_dec

        X_dec1_out = self.FC_dec1_2(F.relu(self.FC_dec1_1(X_dec1_out)))

        # Present-Past Cross Attention and the Past Decoder without DHM in the training and validation
        if flag=='train' or flag=='val':
            X_cross2_out = self.CrossAttention2(X_enc_out_L1, STE_P, STE_H)
            X_dec2_out = self.STATBlockDec2[0](X_cross2_out, STE_H, mask=True)
            X_dec2_out = self.FC_dec2_2(F.relu(self.FC_dec2_1(X_dec2_out)))
            
            return X_dec1_out.squeeze(3), X_dec2_out.squeeze(3), dhm0_S, dhm1_S, dhm0_T, dhm1_T
        
        if flag=='test':
            
            return X_dec1_out.squeeze(3), dhm0_S, dhm1_S, dhm0_T, dhm1_T


def mae_loss(pred, label, device):
    mask = (label != 0)
    mask = mask.type(torch.FloatTensor).to(device)
    mask /= torch.mean(mask)
    mask[mask!=mask] = 0
    loss = torch.abs(pred - label)
    loss *= mask
    loss[loss!=loss] = 0
    loss = torch.mean(loss)
    return loss
