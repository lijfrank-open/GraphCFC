import torch
import torch.nn as nn
from model_dens import DensNet, DensNetLayer
from model_fusion import ConcatFusion, FiLM, GatedFusion, SumFusion
from model_gatmlp import GATMLP, GATMLPLayer
from model_mlp import MLP
from model_utils import batch_graphify, simple_batch_graphify

class GNNModel(nn.Module):

    def __init__(self, args, D_m_a, D_m_v, D_m, num_speakers, n_classes):
        
        super(GNNModel, self).__init__()
        self.base_model = args.base_model
        self.no_cuda = args.no_cuda
        self.num_speakers = num_speakers
        self.return_feature = True
        self.dropout = args.dropout
        self.modals = [x for x in args.modals]
        self.ratio_modal = args.ratio_modal
        self.use_relation = args.use_relation
        self.multi_modal = args.multi_modal
        n_relations = int((num_speakers * (num_speakers + 1))/2)
        self.window_past = args.windowp
        self.window_future = args.windowf
        self.hidesize = args.hidesize
        self.list_mlp = args.list_mlp
        self.ratio_speaker = args.ratio_speaker

        if 'a' in self.modals:
            if self.base_model[0] == 'LSTM':
                self.linear_audio = nn.Linear(D_m_a, args.base_size[0])
                self.rnn_audio = nn.LSTM(input_size=args.base_size[0], hidden_size=args.hidesize, num_layers=args.base_nlayers[0], batch_first=True, dropout=args.dropout, bidirectional=True)
                self.linear_audio_ = nn.Linear(2*args.hidesize, args.hidesize)
            elif self.base_model[0] == 'GRU':
                self.linear_audio = nn.Linear(D_m_a, args.base_size[0])
                self.rnn_audio = nn.GRU(input_size=args.base_size[0], hidden_size=args.hidesize, num_layers=args.base_nlayers[0], batch_first=True, dropout=args.dropout, bidirectional=True)
                self.linear_audio_ = nn.Linear(2*args.hidesize, args.hidesize)
            elif self.base_model[0] == 'Transformer':
                self.linear_audio = nn.Linear(D_m_a, args.hidesize)
                encoder_layer_audio = nn.TransformerEncoderLayer(d_model=args.hidesize, nhead=4, dropout=args.dropout, batch_first=True)
                self.transformer_encoder_audio = nn.TransformerEncoder(encoder_layer_audio, num_layers=args.base_nlayers[0])
            elif self.base_model[0] == 'Dens':
                self.linear_audio = nn.Linear(D_m_a, args.hidesize)
                encoder_layer_audio = DensNetLayer(hidesize=args.hidesize, dropout=self.dropout, activation='tanh', no_cuda=self.no_cuda)
                self.dens_audio = DensNet(encoder_layer_audio, num_layers=args.base_nlayers[0])
            elif self.base_model[0] == 'None':
                self.linear_audio = nn.Linear(D_m_a, args.hidesize)
            else:
                print ('Base model must be one of .')
                raise NotImplementedError 

        if 'v' in self.modals:
            if self.base_model[1] == 'LSTM':
                self.linear_visual = nn.Linear(D_m_v, args.base_size[1])
                self.rnn_visual = nn.LSTM(input_size=args.base_size[1], hidden_size=args.hidesize, num_layers=args.base_nlayers[1], batch_first=True, dropout=args.dropout, bidirectional=True)
                self.linear_visual_ = nn.Linear(2*args.hidesize, args.hidesize)
            elif self.base_model[1] == 'GRU':
                self.linear_visual = nn.Linear(D_m_v, args.base_size[1])
                self.rnn_visual = nn.GRU(input_size=args.base_size[1], hidden_size=args.hidesize, num_layers=args.base_nlayers[1], batch_first=True, dropout=args.dropout, bidirectional=True)
                self.linear_visual_ = nn.Linear(2*args.hidesize, args.hidesize)
            elif self.base_model[1] == 'Transformer':
                self.linear_visual = nn.Linear(D_m_v, args.hidesize)
                encoder_layer_visual = nn.TransformerEncoderLayer(d_model=args.hidesize, nhead=4, dropout=args.dropout, batch_first=True)
                self.transformer_encoder_visual = nn.TransformerEncoder(encoder_layer_visual, num_layers=args.base_nlayers[1])
            elif self.base_model[1] == 'Dens':
                self.linear_visual = nn.Linear(D_m_v, args.hidesize)
                encoder_layer_visual = DensNetLayer(hidesize=args.hidesize, dropout=self.dropout, activation='tanh', no_cuda=self.no_cuda)
                self.dens_visual = DensNet(encoder_layer_visual, num_layers=args.base_nlayers[1])
            elif self.base_model[1] == 'None':
                self.linear_visual = nn.Linear(D_m_v, args.hidesize)
            else:
                print ('Base model must be one of .')
                raise NotImplementedError 

        if 'l' in self.modals:
            if self.base_model[2] == 'LSTM':
                self.linear_text = nn.Linear(D_m, args.base_size[2])
                self.rnn_text = nn.LSTM(input_size=args.base_size[2], hidden_size=args.hidesize, num_layers=args.base_nlayers[2], batch_first=True, dropout=args.dropout, bidirectional=True)
                self.linear_text_ = nn.Linear(2*args.hidesize, args.hidesize)
            elif self.base_model[2] == 'GRU':
                self.linear_text = nn.Linear(D_m, args.base_size[2])
                self.rnn_text = nn.GRU(input_size=args.base_size[2], hidden_size=args.hidesize, num_layers=args.base_nlayers[2], batch_first=True, dropout=args.dropout, bidirectional=True)
                self.linear_text_ = nn.Linear(2*args.hidesize, args.hidesize)
            elif self.base_model[2] == 'Transformer':
                self.linear_text = nn.Linear(D_m, args.hidesize)
                encoder_layer_text = nn.TransformerEncoderLayer(d_model=args.hidesize, nhead=4, dropout=args.dropout, batch_first=True)
                self.transformer_encoder_text = nn.TransformerEncoder(encoder_layer_text, num_layers=args.base_nlayers[2])
            elif self.base_model[2] == 'Dens':
                self.linear_text = nn.Linear(D_m, args.hidesize)
                encoder_layer_text = DensNetLayer(hidesize=args.hidesize, dropout=self.dropout, activation='tanh', no_cuda=self.no_cuda)
                self.dens_text = DensNet(encoder_layer_text, num_layers=args.base_nlayers[2])
            elif self.base_model[2] == 'None':
                self.linear_text = nn.Linear(D_m, args.hidesize)
            else:
                print ('Base model must be one of .')
                raise NotImplementedError 

        if args.ratio_speaker > 0:
            self.speaker_embeddings = nn.Embedding(num_speakers, args.hidesize)
        if len(self.modals)==1:
            self.rel_embeddings = nn.Embedding(n_relations, args.hidesize)
        if len(self.modals)>1:
            self.rel_embeddings1 = nn.Embedding(n_relations + 1, args.hidesize) 
        
        if 'a' in self.modals:
            densnetlayer_audio = DensNetLayer(hidesize=args.hidesize, dropout=args.dropout, activation='gelu')
            self.densnet_audio = DensNet(densnetlayer_audio, num_layers=args.unimodal_nlayers[0])
        if 'v' in self.modals:    
            densnetlayer_visual = DensNetLayer(hidesize=args.hidesize, dropout=args.dropout, activation='gelu')
            self.densnet_visual = DensNet(densnetlayer_visual, num_layers=args.unimodal_nlayers[1])
        if 'l' in self.modals:
            densnetlayer_text = DensNetLayer(hidesize=args.hidesize, dropout=args.dropout, activation='gelu')
            self.densnet_text = DensNet(densnetlayer_text, num_layers=args.unimodal_nlayers[2])
        
        if len(self.modals) > 1:
            densnetlayer_share = DensNetLayer(hidesize=args.hidesize, dropout=args.dropout, activation='gelu')
            self.densnet_share = DensNet(densnetlayer_share, num_layers=args.unimodal_nlayers[3])
            if len(self.modals) == 3:
                self.fc_share = nn.Linear(3*args.hidesize, args.hidesize)
            if len(self.modals) == 2:
                self.fc_share = nn.Linear(2*args.hidesize, args.hidesize)

        if args.list_mlp != []:
            self.mlp_audio = MLP(args.hidesize, args.list_mlp, args.dropout, activation='gelu')
            self.mlp_visual = MLP(args.hidesize, args.list_mlp, args.dropout, activation='gelu')
            self.mlp_text = MLP(args.hidesize, args.list_mlp, args.dropout, activation='gelu')
            self.mlp_share = MLP(args.hidesize, args.list_mlp, args.dropout, activation='gelu')
            self.smax_fc_audio = nn.Linear(args.list_mlp[-1], n_classes)
            self.smax_fc_visual = nn.Linear(args.list_mlp[-1], n_classes)
            self.smax_fc_text = nn.Linear(args.list_mlp[-1], n_classes)
            self.smax_fc_share = nn.Linear(args.list_mlp[-1], n_classes)
        else:
            if 'a' in self.modals:
                self.smax_fc_audio = nn.Linear(args.hidesize, n_classes)
            if 'v' in self.modals:
                self.smax_fc_visual = nn.Linear(args.hidesize, n_classes)
            if 'l' in self.modals:
                self.smax_fc_text = nn.Linear(args.hidesize, n_classes)
            if len(self.modals)>1:
                self.smax_fc_share = nn.Linear(args.hidesize, n_classes)

        if args.ratio_modal > 0:
            self.modal_embeddings = nn.Embedding(3, args.hidesize)
        
        if len(self.modals) ==1:
            gatmlplayer = GATMLPLayer(args.hidesize, dropout=args.dropout, num_heads=args.nheads, agg_type = args.agg_type, use_relation=args.use_relation, use_residual=args.use_residual, activation='gelu', norm_first=args.norm_first[1], no_cuda=args.no_cuda)
            self.gatmlp = GATMLP(gatmlplayer, num_layers=args.multimodal_nlayers[0], hidesize=args.hidesize)
        if len(self.modals) ==2:
            if 'a' in self.modals and 'v' in self.modals:
                gatmlplayer_av = GATMLPLayer(args.hidesize, dropout=args.dropout, num_heads=args.nheads, agg_type = args.agg_type, use_relation=args.use_relation, use_residual=args.use_residual, activation='gelu', norm_first=args.norm_first[1], no_cuda=args.no_cuda)
                self.gatmlp_av = GATMLP(gatmlplayer_av, num_layers=args.multimodal_nlayers[0], hidesize=args.hidesize)
                if args.fusion_method == 'sum':
                    self.fusion_av = SumFusion(input_dim=args.hidesize, output_dim=args.hidesize)
                elif args.fusion_method == 'concat':
                    self.fusion_av = ConcatFusion(input_dim=2*args.hidesize, output_dim=args.hidesize)
                elif args.fusion_method == 'film':
                    self.fusion_av = FiLM(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_film=True)
                elif args.fusion_method == 'gated':
                    self.fusion_av = GatedFusion(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_gate=True)
                else:
                    raise NotImplementedError('Incorrect fusion method: {}!'.format(args.fusion_method))
                
                gatmlplayer_av_share = GATMLPLayer(args.hidesize, dropout=args.dropout, num_heads=args.nheads, agg_type = args.agg_type, use_relation=args.use_relation, use_residual=args.use_residual, activation='gelu', norm_first=args.norm_first[1], no_cuda=args.no_cuda)
                self.gatmlp_av_share = GATMLP(gatmlplayer_av_share, num_layers=args.multimodal_nlayers[2], hidesize=args.hidesize)
                if args.fusion_method == 'sum':
                    self.fusion_av_share = SumFusion(input_dim=args.hidesize, output_dim=args.hidesize)
                elif args.fusion_method == 'concat':
                    self.fusion_av_share = ConcatFusion(input_dim=2*args.hidesize, output_dim=args.hidesize)
                elif args.fusion_method == 'film':
                    self.fusion_av_share = FiLM(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_film=True)
                elif args.fusion_method == 'gated':
                    self.fusion_av_share = GatedFusion(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_gate=True)
                else:
                    raise NotImplementedError('Incorrect fusion method: {}!'.format(args.fusion_method))

            if 'a' in self.modals and 'l' in self.modals:
                gatmlplayer_al = GATMLPLayer(args.hidesize, dropout=args.dropout, num_heads=args.nheads, agg_type = args.agg_type, use_relation=args.use_relation, use_residual=args.use_residual, activation='gelu', norm_first=args.norm_first[1], no_cuda=args.no_cuda)
                self.gatmlp_al = GATMLP(gatmlplayer_al, num_layers=args.multimodal_nlayers[0], hidesize=args.hidesize)
                if args.fusion_method == 'sum':
                    self.fusion_al = SumFusion(input_dim=args.hidesize, output_dim=args.hidesize)
                elif args.fusion_method == 'concat':
                    self.fusion_al = ConcatFusion(input_dim=2*args.hidesize, output_dim=args.hidesize)
                elif args.fusion_method == 'film':
                    self.fusion_al = FiLM(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_film=True)
                elif args.fusion_method == 'gated':
                    self.fusion_al = GatedFusion(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_gate=True)
                else:
                    raise NotImplementedError('Incorrect fusion method: {}!'.format(args.fusion_method))

                gatmlplayer_al_share = GATMLPLayer(args.hidesize, dropout=args.dropout, num_heads=args.nheads, agg_type = args.agg_type, use_relation=args.use_relation, use_residual=args.use_residual, activation='gelu', norm_first=args.norm_first[1], no_cuda=args.no_cuda)
                self.gatmlp_al_share = GATMLP(gatmlplayer_al_share, num_layers=args.multimodal_nlayers[2], hidesize=args.hidesize)
                if args.fusion_method == 'sum':
                    self.fusion_al_share = SumFusion(input_dim=args.hidesize, output_dim=args.hidesize)
                elif args.fusion_method == 'concat':
                    self.fusion_al_share = ConcatFusion(input_dim=2*args.hidesize, output_dim=args.hidesize)
                elif args.fusion_method == 'film':
                    self.fusion_al_share = FiLM(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_film=True)
                elif args.fusion_method == 'gated':
                    self.fusion_al_share = GatedFusion(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_gate=True)
                else:
                    raise NotImplementedError('Incorrect fusion method: {}!'.format(args.fusion_method))

            if 'v' in self.modals and 'l' in self.modals:
                gatmlplayer_vl = GATMLPLayer(args.hidesize, dropout=args.dropout, num_heads=args.nheads, agg_type = args.agg_type, use_relation=args.use_relation, use_residual=args.use_residual, activation='gelu', norm_first=args.norm_first[1], no_cuda=args.no_cuda)
                self.gatmlp_vl = GATMLP(gatmlplayer_vl, num_layers=args.multimodal_nlayers[0], hidesize=args.hidesize)
                if args.fusion_method == 'sum':
                    self.fusion_vl = SumFusion(input_dim=args.hidesize, output_dim=args.hidesize)
                elif args.fusion_method == 'concat':
                    self.fusion_vl = ConcatFusion(input_dim=2*args.hidesize, output_dim=args.hidesize)
                elif args.fusion_method == 'film':
                    self.fusion_vl = FiLM(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_film=True)
                elif args.fusion_method == 'gated':
                    self.fusion_vl = GatedFusion(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_gate=True)
                else:
                    raise NotImplementedError('Incorrect fusion method: {}!'.format(args.fusion_method))

                gatmlplayer_vl_share = GATMLPLayer(args.hidesize, dropout=args.dropout, num_heads=args.nheads, agg_type = args.agg_type, use_relation=args.use_relation, use_residual=args.use_residual, activation='gelu', norm_first=args.norm_first[1], no_cuda=args.no_cuda)
                self.gatmlp_vl_share = GATMLP(gatmlplayer_vl_share, num_layers=args.multimodal_nlayers[2], hidesize=args.hidesize)
                if args.fusion_method == 'sum':
                    self.fusion_vl_share = SumFusion(input_dim=args.hidesize, output_dim=args.hidesize)
                elif args.fusion_method == 'concat':
                    self.fusion_vl_share = ConcatFusion(input_dim=2*args.hidesize, output_dim=args.hidesize)
                elif args.fusion_method == 'film':
                    self.fusion_vl_share = FiLM(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_film=True)
                elif args.fusion_method == 'gated':
                    self.fusion_vl_share = GatedFusion(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_gate=True)
                else:
                    raise NotImplementedError('Incorrect fusion method: {}!'.format(args.fusion_method))

        if len(self.modals) ==3:
            gatmlplayer_av = GATMLPLayer(args.hidesize, dropout=args.dropout, num_heads=args.nheads, agg_type = args.agg_type, use_relation=args.use_relation, use_residual=args.use_residual, activation='gelu', norm_first=args.norm_first[1], no_cuda=args.no_cuda)
            self.gatmlp_av = GATMLP(gatmlplayer_av, num_layers=args.multimodal_nlayers[0], hidesize=args.hidesize)
            if args.fusion_method == 'sum':
                self.fusion_av = SumFusion(input_dim=args.hidesize, output_dim=args.hidesize)
            elif args.fusion_method == 'concat':
                self.fusion_av = ConcatFusion(input_dim=2*args.hidesize, output_dim=args.hidesize)
            elif args.fusion_method == 'film':
                self.fusion_av = FiLM(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_film=True)
            elif args.fusion_method == 'gated':
                self.fusion_av = GatedFusion(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_gate=True)
            else:
                raise NotImplementedError('Incorrect fusion method: {}!'.format(args.fusion_method))
        
            gatmlplayer_av_l = GATMLPLayer(args.hidesize, dropout=args.dropout, num_heads=args.nheads, agg_type = args.agg_type, use_relation=args.use_relation, use_residual=args.use_residual, activation='gelu', norm_first=args.norm_first[1], no_cuda=args.no_cuda)
            self.gatmlp_av_l = GATMLP(gatmlplayer_av_l, num_layers=args.multimodal_nlayers[1], hidesize=args.hidesize)
            if args.fusion_method == 'sum':
                self.fusion_av_l = SumFusion(input_dim=args.hidesize, output_dim=args.hidesize)
            elif args.fusion_method == 'concat':
                self.fusion_av_l = ConcatFusion(input_dim=2*args.hidesize, output_dim=args.hidesize)
            elif args.fusion_method == 'film':
                self.fusion_av_l = FiLM(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_film=True)
            elif args.fusion_method == 'gated':
                self.fusion_av_l = GatedFusion(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_gate=True)
            else:
                raise NotImplementedError('Incorrect fusion method: {}!'.format(args.fusion_method))

            gatmlplayer_av_l_share = GATMLPLayer(args.hidesize, dropout=args.dropout, num_heads=args.nheads, agg_type = args.agg_type, use_relation=args.use_relation, use_residual=args.use_residual, activation='gelu', norm_first=args.norm_first[1], no_cuda=args.no_cuda)
            self.gatmlp_av_l_share = GATMLP(gatmlplayer_av_l_share, num_layers=args.multimodal_nlayers[2], hidesize=args.hidesize)
            if args.fusion_method == 'sum':
                self.fusion_av_l_share = SumFusion(input_dim=args.hidesize, output_dim=args.hidesize)
            elif args.fusion_method == 'concat':
                self.fusion_av_l_share = ConcatFusion(input_dim=2*args.hidesize, output_dim=args.hidesize)
            elif args.fusion_method == 'film':
                self.fusion_av_l_share = FiLM(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_film=True)
            elif args.fusion_method == 'gated':
                self.fusion_av_l_share = GatedFusion(input_dim=args.hidesize, dim=args.hidesize, output_dim=args.hidesize, x_gate=True)
            else:
                raise NotImplementedError('Incorrect fusion method: {}!'.format(args.fusion_method))

        edge_type_mapping = {} 
        for j in range(num_speakers):
            for k in range(j, num_speakers):
                edge_type_mapping[str(j) + str(k)] = len(edge_type_mapping)     
        if len(self.modals)>1:
            edge_type_mapping['modal'] = len(edge_type_mapping)
        self.edge_type_mapping = edge_type_mapping
        if args.list_mlp != []:
            self.mlp = MLP(args.hidesize, args.list_mlp, args.dropout, activation='gelu')
            self.smax_fc = nn.Linear(args.list_mlp[-1], n_classes)
        else:
            self.smax_fc = nn.Linear(args.hidesize, n_classes)

    def forward(self, U, qmask, umask, seq_lengths, max_seq_length, U_a=None, U_v=None):
        if 'a' in self.modals:
            if self.base_model[0] == 'LSTM':
                U_a = self.linear_audio(U_a)
                U_a = nn.utils.rnn.pack_padded_sequence(U_a, seq_lengths.cpu(), batch_first=True, enforce_sorted=False) # 压紧
                self.rnn_audio.flatten_parameters()
                emotions_a, hidden_a = self.rnn_audio(U_a)
                emotions_a, _ = nn.utils.rnn.pad_packed_sequence(emotions_a, batch_first=True) # pad
                emotions_a = self.linear_audio_(emotions_a)
            elif self.base_model[0] == 'GRU':
                U_a = self.linear_audio(U_a)
                U_a = nn.utils.rnn.pack_padded_sequence(U_a, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
                self.rnn_audio.flatten_parameters()
                emotions_a, hidden_a = self.rnn_audio(U_a)
                emotions_a, _ = nn.utils.rnn.pad_packed_sequence(emotions_a, batch_first=True) # pad
                emotions_a = self.linear_audio_(emotions_a)
            elif self.base_model[0] == 'Transformer':
                U_a = self.linear_audio(U_a)
                emotions_a = self.transformer_encoder_audio(U_a, src_key_padding_mask=umask)
            elif self.base_model[0] == 'Dens':
                U_a = self.linear_audio(U_a)
                emotions_a = self.dens_audio(U_a)
            elif self.base_model[0] == 'None':
                emotions_a = torch.tanh(self.linear_audio(U_a))
        
        if 'v' in self.modals:
            if self.base_model[1] == 'LSTM':
                U_v = self.linear_visual(U_v)
                U_v = nn.utils.rnn.pack_padded_sequence(U_v, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
                self.rnn_visual.flatten_parameters()
                emotions_v, hidden_v = self.rnn_visual(U_v)
                emotions_v, _ = nn.utils.rnn.pad_packed_sequence(emotions_v, batch_first=True)
                emotions_v = self.linear_visual_(emotions_v)
            elif self.base_model[1] == 'GRU':
                U_v = self.linear_visual(U_v)
                U_v = nn.utils.rnn.pack_padded_sequence(U_v, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
                self.rnn_visual.flatten_parameters()
                emotions_v, hidden_v = self.rnn_visual(U_v)
                emotions_v, _ = nn.utils.rnn.pad_packed_sequence(emotions_v, batch_first=True)
                emotions_v = self.linear_visual_(emotions_v)
            elif self.base_model[1] == 'Transformer':
                U_v = self.linear_visual(U_v)
                emotions_v = self.transformer_encoder_visual(U_v, src_key_padding_mask=umask)
            elif self.base_model[1] == 'Dens':
                U_v = self.linear_visual(U_v)
                emotions_v = self.dens_visual(U_v)
            elif self.base_model[1] == 'None':
                emotions_v = torch.tanh(self.linear_visual(U_v))
        
        if 'l' in self.modals:
            if self.base_model[2] == 'LSTM':
                U = self.linear_text(U)
                U = nn.utils.rnn.pack_padded_sequence(U, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
                self.rnn_text.flatten_parameters()
                emotions_l, hidden_l = self.rnn_text(U)
                emotions_l, _ = nn.utils.rnn.pad_packed_sequence(emotions_l, batch_first=True)
                emotions_l = self.linear_text_(emotions_l)
            elif self.base_model[2] == 'GRU':
                U = self.linear_text(U)
                U = nn.utils.rnn.pack_padded_sequence(U, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
                self.rnn_text.flatten_parameters()
                emotions_l, hidden_l = self.rnn_text(U)
                emotions_l, _ = nn.utils.rnn.pad_packed_sequence(emotions_l, batch_first=True) # pad
                emotions_l = self.linear_text_(emotions_l)
            elif self.base_model[2] == 'Transformer':
                U = self.linear_text(U)
                emotions_l = self.transformer_encoder_text(U, src_key_padding_mask=umask)
            elif self.base_model[2] == 'Dens':
                U = self.linear_text(U)
                emotions_l = self.dens_text(U)
            elif self.base_model[2] == 'None':
                emotions_l = torch.tanh(self.linear_text(U))
        if len(self.modals) == 3:
            features_a, edge_index, edge_type, edge_index_lengths, edge_index1, edge_type1 = batch_graphify(emotions_a, qmask, seq_lengths, self.window_past, self.window_future, self.edge_type_mapping, self.no_cuda)
            features_v = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
            features_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
        if len(self.modals) == 2:
            if 'a' in self.modals and 'v' in self.modals:
                features_a, edge_index, edge_type, edge_index_lengths, edge_index1, edge_type1 = batch_graphify(emotions_a, qmask, seq_lengths, self.window_past, self.window_future, self.edge_type_mapping, self.no_cuda)
                features_v = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
            if 'a' in self.modals and 'l' in self.modals:
                features_a, edge_index, edge_type, edge_index_lengths, edge_index1, edge_type1 = batch_graphify(emotions_a, qmask, seq_lengths, self.window_past, self.window_future, self.edge_type_mapping, self.no_cuda)
                features_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
            if 'v' in self.modals and 'l' in self.modals:
                features_v, edge_index, edge_type, edge_index_lengths, edge_index1, edge_type1 = batch_graphify(emotions_v, qmask, seq_lengths, self.window_past, self.window_future, self.edge_type_mapping, self.no_cuda)
                features_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
        if len(self.modals) == 1:
            if 'a' in self.modals:
                features_a, edge_index, edge_type, edge_index_lengths, edge_index1, edge_type1 = batch_graphify(emotions_a, qmask, seq_lengths, self.window_past, self.window_future, self.edge_type_mapping, self.no_cuda)
            if 'v' in self.modals:
                features_v, edge_index, edge_type, edge_index_lengths, edge_index1, edge_type1 = batch_graphify(emotions_v, qmask, seq_lengths, self.window_past, self.window_future, self.edge_type_mapping, self.no_cuda)
            if 'l' in self.modals:
                features_l, edge_index, edge_type, edge_index_lengths, edge_index1, edge_type1 = batch_graphify(emotions_l, qmask, seq_lengths, self.window_past, self.window_future, self.edge_type_mapping, self.no_cuda)
        if len(self.modals) == 3:
            if self.ratio_modal > 0:
                emb_idx = torch.LongTensor([0, 1, 2]).cuda()
                emb_vector = self.modal_embeddings(emb_idx)
                features_a = features_a + self.ratio_modal*emb_vector[0].reshape(1, -1).expand(features_a.shape[0], features_a.shape[1])
                features_v = features_v + self.ratio_modal*emb_vector[1].reshape(1, -1).expand(features_v.shape[0], features_v.shape[1])
                features_l = features_l + self.ratio_modal*emb_vector[2].reshape(1, -1).expand(features_l.shape[0], features_l.shape[1])
        if self.ratio_speaker > 0:
            qmask_ = torch.cat([qmask[i,:x,:] for i,x in enumerate(seq_lengths)],dim=0)
            spk_idx = torch.argmax(qmask_, dim=-1).cuda() if not self.no_cuda else torch.argmax(qmask_, dim=-1)
            spk_emb_vector = self.speaker_embeddings(spk_idx)
            if 'a' in self.modals:
                features_a = features_a + self.ratio_speaker*spk_emb_vector
            if 'v' in self.modals:
                features_v = features_v + self.ratio_speaker*spk_emb_vector
            if 'l' in self.modals:
                features_l = features_l + self.ratio_speaker*spk_emb_vector
        if 'a' in self.modals:
            features_single_a = self.densnet_audio(features_a)
        if 'v' in self.modals:
            features_single_v = self.densnet_visual(features_v)
        if 'l' in self.modals:
            features_single_l = self.densnet_text(features_l)

        if len(self.modals)>1:
            if 'a' in self.modals:
                features_share_a = self.densnet_share(features_a)
            if 'v' in self.modals:
                features_share_v = self.densnet_share(features_v)
            if 'l' in self.modals:
                features_share_l = self.densnet_share(features_l)
        if len(self.modals) == 3:
            features_share = self.fc_share(torch.cat([features_share_a, features_share_v, features_share_l],dim=-1))
        if len(self.modals) == 2:
            if 'a' in self.modals and 'v' in self.modals:
                features_share = self.fc_share(torch.cat([features_share_a, features_share_v],dim=-1))
            if 'a' in self.modals and 'l' in self.modals:
                features_share = self.fc_share(torch.cat([features_share_a, features_share_l],dim=-1))
            if 'v' in self.modals and 'l' in self.modals:
                features_share = self.fc_share(torch.cat([features_share_v, features_share_l],dim=-1))
        
        if self.list_mlp != []:
            features_single_pa = self.mlp_audio(features_single_a)
            features_single_pv = self.mlp_visual(features_single_v)
            features_single_pl = self.mlp_text(features_single_l)
            features_share_p = self.mlp_share(features_share)
            prob_a = self.smax_fc_audio(features_single_pa)
            prob_v = self.smax_fc_visual(features_single_pv)
            prob_l = self.smax_fc_text(features_single_pl)
            prob_share = self.smax_fc_share(features_share_p)
        else:
            if 'a' in self.modals:
                prob_a = self.smax_fc_audio(features_single_a)
            if 'v' in self.modals:
                prob_v = self.smax_fc_visual(features_single_v)
            if 'l' in self.modals:
                prob_l = self.smax_fc_text(features_single_l)
            if len(self.modals)>1:
                prob_share = self.smax_fc_share(features_share)

        if len(self.modals)==1:
            rel_emb_vector = self.rel_embeddings(edge_type)
        if len(self.modals)>1:
            rel_emb_vector1 = self.rel_embeddings1(edge_type1)
        
        if len(self.modals) ==1:
            if 'a' in self.modals:
                features_cross_a = self.gatmlp(features_single_a, edge_index, rel_emb_vector)
            if 'v' in self.modals:
                features_cross_v = self.gatmlp(features_single_v, edge_index, rel_emb_vector)
            if 'l' in self.modals:
                features_cross_l = self.gatmlp(features_single_l, edge_index, rel_emb_vector)
        if len(self.modals) ==2:
            if 'a' in self.modals and 'v' in self.modals:
                features_single_av = torch.cat([features_single_a, features_single_v], dim=0)
                features_cross_av = self.gatmlp_av(features_single_av, edge_index1, rel_emb_vector1)
                features_cross_a, features_cross_v = torch.chunk(features_cross_av, 2, dim=0)
                features_av = self.fusion_av(features_cross_a, features_cross_v)

                features_single_av_share = torch.cat([features_av, features_share], dim=0)
                features_cross_av_share = self.gatmlp_av_share(features_single_av_share, edge_index1, rel_emb_vector1)
                features_cross_av_, features_cross_share = torch.chunk(features_cross_av_share, 2, dim=0)
                features_av_share = self.fusion_av_share(features_cross_av_, features_cross_share)
            if 'a' in self.modals and 'l' in self.modals:
                features_single_al = torch.cat([features_single_a, features_single_l], dim=0)
                features_cross_al = self.gatmlp_al(features_single_al, edge_index1, rel_emb_vector1)
                features_cross_a, features_cross_l = torch.chunk(features_cross_al, 2, dim=0)
                features_al = self.fusion_al(features_cross_a, features_cross_l)

                features_single_al_share = torch.cat([features_al, features_share], dim=0)
                features_cross_al_share = self.gatmlp_al_share(features_single_al_share, edge_index1, rel_emb_vector1)
                features_cross_al_, features_cross_share = torch.chunk(features_cross_al_share, 2, dim=0)
                features_al_share = self.fusion_al_share(features_cross_al_, features_cross_share)
            if 'v' in self.modals and 'l' in self.modals:
                features_single_vl = torch.cat([features_single_v, features_single_l], dim=0)
                features_cross_vl = self.gatmlp_vl(features_single_vl, edge_index1, rel_emb_vector1)
                features_cross_v, features_cross_l = torch.chunk(features_cross_vl, 2, dim=0)
                features_vl = self.fusion_vl(features_cross_v, features_cross_l)

                features_single_vl_share = torch.cat([features_vl, features_share], dim=0)
                features_cross_vl_share = self.gatmlp_vl_share(features_single_vl_share, edge_index1, rel_emb_vector1)
                features_cross_vl_, features_cross_share = torch.chunk(features_cross_vl_share, 2, dim=0)
                features_vl_share = self.fusion_vl_share(features_cross_vl_, features_cross_share)

        if len(self.modals) ==3:
            features_single_av = torch.cat([features_single_a, features_single_v], dim=0)
            features_cross_av = self.gatmlp_av(features_single_av, edge_index1, rel_emb_vector1)
            features_cross_a, features_cross_v = torch.chunk(features_cross_av, 2, dim=0)
            features_av = self.fusion_av(features_cross_a, features_cross_v)

            features_single_av_l = torch.cat([features_av, features_single_l], dim=0)
            features_cross_av_l = self.gatmlp_av_l(features_single_av_l, edge_index1, rel_emb_vector1)
            features_cross_av_, features_cross_l = torch.chunk(features_cross_av_l, 2, dim=0)
            features_av_l = self.fusion_av_l(features_cross_av_, features_cross_l)

            features_single_av_l_share = torch.cat([features_av_l, features_share], dim=0)
            features_cross_av_l_share = self.gatmlp_av_l_share(features_single_av_l_share, edge_index1, rel_emb_vector1)
            features_cross_av_l_, features_cross_share = torch.chunk(features_cross_av_l_share, 2, dim=0)
            features_av_l_share = self.fusion_av_l_share(features_cross_av_l_, features_cross_share)

        if self.list_mlp != []:
            prob = self.smax_fc(self.mlp(features_av_l_share))
        else:
            if len(self.modals) ==3:
                prob = self.smax_fc(features_av_l_share)
            if len(self.modals) ==2:
                if 'a' in self.modals and 'v' in self.modals:
                    prob = self.smax_fc(features_av_share)
                if 'a' in self.modals and 'l' in self.modals:
                    prob = self.smax_fc(features_al_share)
                if 'v' in self.modals and 'l' in self.modals:
                    prob = self.smax_fc(features_vl_share)
            if len(self.modals) ==1:
                if 'a' in self.modals:
                    prob = self.smax_fc(features_cross_a)
                if 'v' in self.modals:
                    prob = self.smax_fc(features_cross_v)
                if 'l' in self.modals:
                    prob = self.smax_fc(features_cross_l)
        if len(self.modals)==1:
            if 'a' in self.modals:
                return prob_a, prob
            if 'v' in self.modals:
                return prob_v, prob
            if 'l' in self.modals:
                return prob_l, prob
        if len(self.modals)==2:
            if 'a' in self.modals and 'v' in self.modals:
                return prob_a, prob_v, prob_share, prob
            if 'a' in self.modals and 'l' in self.modals:
                return prob_a, prob_l, prob_share, prob
            if 'v' in self.modals and 'l' in self.modals:
                return prob_v, prob_l, prob_share, prob
        if len(self.modals)==3:    
            return prob_a, prob_v, prob_l, prob_share, prob
