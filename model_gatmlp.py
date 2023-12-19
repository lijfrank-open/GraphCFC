import torch
from model_gcn import GATv2Conv_Layer
import torch.nn as nn
from torch.nn import functional as F

def _get_activation_fn(activation):
    if activation == "relu":
        return torch.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "sigmoid":
        return torch.sigmoid
    elif activation == "" or "None":
        return None
    raise RuntimeError("activation should be relu/gelu/tanh/sigmoid, not {}".format(activation))

class GATMLP(torch.nn.Module):
    def __init__(self, encoder_layer, num_layers, hidesize):
        super(GATMLP, self).__init__()
        layer = []
        for l in range(num_layers):
            layer.append(encoder_layer)
        self.layers = nn.ModuleList(layer)
    def forward(self, features, edge_index, rel_emb_vector):
        output = features
        for mod in self.layers:
            output = mod(output, edge_index, rel_emb_vector)
        return output

class GATMLPLayer(torch.nn.Module):
    def __init__(self, hidesize, dropout=0.5, num_heads=5, agg_type = '', use_relation=True, use_residual=True, activation='', norm_first=False, no_cuda=False):
        super(GATMLPLayer, self).__init__()
        self.use_relation = use_relation
        self.no_cuda = no_cuda
        if agg_type == '' or 'None':
            self.agg_type = None
        else:
            self.agg_type = agg_type
        self.hidesize = hidesize
        self.use_residual = use_residual
        self.norm = nn.LayerNorm(hidesize)
        self.norm1 = nn.LayerNorm(hidesize)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidesize, 2*hidesize)
        self.fc1 = nn.Linear(2*hidesize, hidesize)

        if self.use_relation:
            if self.agg_type is None:
                self.convs = GATv2Conv_Layer(hidesize, hidesize, heads=num_heads, add_self_loops=True, edge_dim=hidesize, concat=False)
            else:
                self.convs = GATv2Conv_Layer(hidesize, hidesize, heads=num_heads, add_self_loops=False, edge_dim=hidesize, concat=False)
        else:
            if self.agg_type is None:
                self.convs = GATv2Conv_Layer(hidesize, hidesize, heads=num_heads, add_self_loops=True, concat=False)
            else:
                self.convs = GATv2Conv_Layer(hidesize, hidesize, heads=num_heads, add_self_loops=False, concat=False)
        if self.agg_type == 'GRU':
            self.gru = nn.GRUCell(hidesize, hidesize)
        elif self.agg_type == 'BiGRU':
            self.gru = nn.GRUCell(hidesize, hidesize)
            self.gru1 = nn.GRUCell(hidesize, hidesize)
        elif self.agg_type == 'LSTM':
            self.lstm = nn.LSTMCell(hidesize, hidesize)
        elif self.agg_type == 'BiLSTM':
            self.lstm = nn.LSTMCell(hidesize, hidesize)
            self.lstm1 = nn.LSTMCell(hidesize, hidesize)
        elif self.agg_type == 'sum':
            self.sumpro = nn.Linear(hidesize, hidesize)
            self.sumpro1 = nn.Linear(hidesize, hidesize)
        elif self.agg_type == 'concat':
            self.sumpro2 = nn.Linear(2*hidesize, hidesize)
        elif self.agg_type is None:
            pass
        else:
            raise RuntimeError("not {}".format(agg_type))
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
        self.norm_first = norm_first

    def forward(self, features, edge_index, rel_emb_vector):
        x = features
        if self.norm_first:
            if self.use_residual:
                x = x + self.graph_conv(self.norm(x), edge_index, rel_emb_vector)
            else:
                x = self.graph_conv(self.norm(x), edge_index, rel_emb_vector)
        else:
            if self.use_residual:
                x = self.norm(x + self.graph_conv(x, edge_index, rel_emb_vector))
            else:
                x = self.norm(self.graph_conv(x, edge_index, rel_emb_vector))
        
        if self.norm_first:
            if self.use_residual:
                x = x + self.full_con(self.norm1(x))
            else:
                x = self.full_con(self.norm1(x))
        else:
            if self.use_residual:
                x = self.norm1(x + self.full_con(x))
            else:
                x = self.norm1(self.full_con(x))
        return x

    def graph_conv(self, x, edge_index, rel_emb_vector):
        if self.use_relation:
            if self.agg_type is None:
                x = self.convs(x, edge_index, rel_emb_vector)
            else:
                con_f = self.convs(x, edge_index, rel_emb_vector)
        else:
            if self.agg_type is None:
                x = self.convs(x, edge_index)
            else:
                con_f = self.convs(x, edge_index)
        if self.agg_type == 'GRU':
            x = self.gru(x, con_f)
        elif self.agg_type == 'BiGRU':
            gru_f = self.gru(x, con_f)
            gru1_f = self.gru1(con_f, x)
            x = gru1_f + gru_f
        elif self.agg_type == 'LSTM':
            x, _ = self.lstm(x, (con_f,con_f))
        elif self.agg_type == 'BiLSTM':
            lstm_f, _ = self.lstm(x, (con_f,con_f))
            lstm1_f, _ = self.lstm1(con_f, (x, x))
            x = lstm1_f + lstm_f
        elif self.agg_type == 'sum':
            x = self.sumpro(x) + self.sumpro1(con_f)
        elif self.agg_type == 'concat':
            x = self.sumpro2(torch.cat((x, con_f), dim=-1))
        elif self.agg_type is None:
            pass
        return x

    def full_con(self, x):
        if self.activation is None:
            x = self.dropout1(self.fc1(self.dropout(self.fc(x))))
        else:
            x = self.dropout1(self.fc1(self.dropout(self.activation(self.fc(x)))))
        return x