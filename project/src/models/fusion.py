# Import libraries
import torch_geometric.nn as nngc
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

# Import parent class and constants
from models.graph_base import GraphBase
import constants


class LinearNodeEmbedder(nn.Module):
    def __init__(self,input_dim_fft=354, emb_dim=512, emb_hidden_dim=256, dropout=0.55):
        super().__init__()
        self.embedding_dim = emb_dim
        self.hidden_dim = emb_hidden_dim
        

        self.hidden = nn.Linear(input_dim_fft, self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.output = nn.Linear(self.hidden_dim, self.embedding_dim)

    def forward(self, data):
        x = data.x  # shape: (num_nodes, input_dim)
        #print(f"Input shape fft: {x.shape}")
        x = self.hidden(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

class LSTMEmbedder(nn.Module):
    def __init__(self, 
        input_dim_lstm: int,  # NOTE: will now be 1 for univariate per-node time series
        hidden_dim_lstm: int, 
        emb_dim: int, 
        num_layers_lstm: int = 2, 
        dropout: float = 0.3, 
        bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=1,  # each node has 1D input at each time step (not 3000-D)
            hidden_size=hidden_dim_lstm,
            num_layers=num_layers_lstm,
            batch_first=True,
            dropout=dropout if num_layers_lstm > 1 else 0,
            bidirectional=bidirectional
        )
        
        proj_input_dim = 2 * hidden_dim_lstm if bidirectional else hidden_dim_lstm
        self.proj = nn.Linear(proj_input_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, data):
        time = data.time  # (num_nodes, seq_len) = (1216, 3000)
        time = time.unsqueeze(-1)  # (num_nodes, seq_len, 1)

        _, (h_n, _) = self.lstm(time)  # h_n: (num_layers * num_directions, num_nodes, hidden_dim)

        if self.bidirectional:
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (num_nodes, 2*hidden_dim)
        else:
            last_hidden = h_n[-1]  # (num_nodes, hidden_dim)

        out = self.proj(last_hidden)  # (num_nodes, emb_dim)
        return self.norm(out)


class EmbeddedGCN(GraphBase): # Fourier features
    def __init__(self, hidden_channels: int, emb_dim: int = 64, emb_hidden_dim: int = 32):
        super().__init__()
        self.device = constants.DEVICE

        self.embedder = LinearNodeEmbedder(emb_dim=emb_dim, emb_hidden_dim=emb_hidden_dim)

        self.conv1 = nngc.GCNConv(emb_dim, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)

        self.conv2 = nngc.GCNConv(hidden_channels, hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        self.lin = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(0.5)

        self.leaky_relu_slope = 0.1
        self._init_weights()
        self.to(self.device)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)

        for conv in [self.conv1, self.conv2]:
            nn.init.xavier_uniform_(conv.lin.weight)
            if conv.lin.bias is not None:
                nn.init.zeros_(conv.lin.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.embedder(data)

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)

        x_res = x

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = x + x_res
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)

        x = nngc.global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.lin(x)

    @staticmethod
    def from_config(model_cfg):
        return EmbeddedGCN(**model_cfg)

class FusionBlock(nn.Module):
    def __init__(self, 
        input_dim_lin: int,
        input_dim_lstm: int, 
        hidden_dim_lstm: int, 
        emb_dim: int, 
        num_layers_lstm: int = 2, 
        dropout: float = 0.3, 

        hidden_dim_lin: int = 64,

    ):
        super().__init__()
       
        self.fft_embedder = LinearNodeEmbedder(input_dim_fft=input_dim_lin, emb_dim=emb_dim, emb_hidden_dim=hidden_dim_lin, dropout=dropout)
        self.time_embedder = LSTMEmbedder(input_dim_lstm=input_dim_lstm, 
                                            hidden_dim_lstm=hidden_dim_lstm, 
                                            emb_dim=emb_dim,
                                            dropout=dropout, 
                                            num_layers_lstm=num_layers_lstm)

        self.time_weight = nn.Parameter(torch.ones(1), requires_grad=True)
        self.fft_weight = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, data):
        fft_embedded = self.fft_embedder(data)         # (N, out_channels)
        time_embedded = self.time_embedder(data)  # (batch_size, out_channels)
        
        print(f"FFT embedded shape: {fft_embedded.shape}")  # debug
        print(f"Time embedded shape: {time_embedded.shape}")  # debug
        x = torch.cat([
            self.time_weight * time_embedded,
            self.fft_weight * fft_embedded
        ], dim=-1)  # (N, 2*out_channels)

        return x

class FusionGCN(GraphBase):
    def __init__(self, 
        input_dim_lin: int = 354, 
        input_dim_lstm: int = 3000, 
        hidden_dim_lstm: int = 256, 
        emb_dim: int = 128, 
        num_layers_lstm: int = 2, 
        dropout: float = 0.3, 
        hidden_dim_lin: int = 64,
        hidden_gcn: int = 64,
    ):
        super().__init__()
        self.embedder = FusionBlock(
            input_dim_lin,
            input_dim_lstm,
            hidden_dim_lstm,
            emb_dim,
            num_layers_lstm,
            dropout,
            hidden_dim_lin,
        )
        self.device = constants.DEVICE
        self.conv1 = nngc.GCNConv(2 * emb_dim, hidden_gcn)
        self.norm1 = nn.LayerNorm(hidden_gcn)

        self.conv2 = nngc.GCNConv(hidden_gcn, hidden_gcn)
        self.norm2 = nn.LayerNorm(hidden_gcn)

        self.lin = nn.Linear(hidden_gcn, 1)
        self.dropout = nn.Dropout(0.5)

        self.leaky_relu_slope = 0.1

        # Don't call _init_weights here yet — we do it after embedder is initialized

    def _init_weights(self):
        def init_func(m):
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

        self.apply(init_func)

    def forward(self, data):

        self._init_weights()


        x = self.embedder(data)  # (N, 2*out_channels)
        x = self.conv1(x, data.edge_index)
        x = self.norm1(x)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)

        x_res = x
        x = self.conv2(x, data.edge_index)
        x = self.norm2(x)
        x = x + x_res
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)

        x = global_mean_pool(x, data.batch)
        x = self.dropout(x)
        return self.lin(x)

    @staticmethod
    def from_config(model_cfg):
        return FusionGCN(**model_cfg)

class EmbeddedGCN(GraphBase): # Fourier features
    def __init__(self, hidden_channels: int, emb_dim: int = 64, emb_hidden_dim: int = 32):
        super().__init__()
        self.device = constants.DEVICE

        self.embedder = LinearNodeEmbedder(emb_dim=emb_dim, emb_hidden_dim=emb_hidden_dim)

        self.conv1 = nngc.GCNConv(emb_dim, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)

        self.conv2 = nngc.GCNConv(hidden_channels, hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        self.lin = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(0.5)

        self.leaky_relu_slope = 0.1
        self._init_weights()
        self.to(self.device)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)

        for conv in [self.conv1, self.conv2]:
            nn.init.xavier_uniform_(conv.lin.weight)
            if conv.lin.bias is not None:
                nn.init.zeros_(conv.lin.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.embedder(data)

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)

        x_res = x

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = x + x_res
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)

        x = nngc.global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.lin(x)

    @staticmethod
    def from_config(model_cfg):
        return EmbeddedGCN(**model_cfg)

class EmbeddedLSTMGCN(GraphBase):  # Fourier features
    def __init__(self, emb_dim, emb_hidden_dim, hidden_channels_gcn, hidden_channels_lstm):
        super().__init__()
        self.device = constants.DEVICE

        # Node embedder
        self.embedder = LinearNodeEmbedder(emb_dim=emb_dim, emb_hidden_dim=emb_hidden_dim)

        # GCN layers
        self.gcn1 = nngc.GCNConv(emb_dim, hidden_channels_gcn)
        self.gcn2 = nngc.GCNConv(hidden_channels_gcn, hidden_channels_gcn)

        # LSTM to process GCN layer outputs as a sequence
        self.lstm = nn.LSTM(
            input_size=hidden_channels_gcn,
            hidden_size=hidden_channels_lstm,
            num_layers=1,
            batch_first=False
        )

        # Final prediction layer
        self.fc = nn.Linear(hidden_channels_lstm, 1)

        self.to(self.device)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Embed node features
        x = self.embedder(data)

        # First GCN layer
        x1 = self.gcn1(x, edge_index)
        x1 = F.relu(x1)

        # Second GCN layer
        x2 = self.gcn2(x1, edge_index)
        x2 = F.relu(x2)

        # Stack the two GCN outputs as a sequence (like time steps)
        sequence = torch.stack([x1, x2], dim=0)  # [2, num_nodes, hidden_channels_gcn]

        # Pass sequence through LSTM
        lstm_out, _ = self.lstm(sequence)  # [2, num_nodes, hidden_channels_lstm]

        # Take the last output from the LSTM for each node
        node_embeddings = lstm_out[-1]  # [num_nodes, hidden_channels_lstm]

        # Global mean pooling to graph embedding
        graph_embeddings = nngc.global_mean_pool(node_embeddings, batch)  # [batch_size, hidden_channels_lstm]

        # Final linear classifier
        logits = self.fc(graph_embeddings)  # [batch_size, 1]

        return logits

    @staticmethod
    def from_config(model_cfg):
        return EmbeddedLSTMGCN(**model_cfg)


class CnnNodeEmbedder(nn.Module):


    def __init__(self, emb_dim=512, emb_hidden_dim=256, dropout=0.3):
        super().__init__()
        self.embedding_dim = emb_dim
        self.hidden_dim = emb_hidden_dim

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),  # → [32, 1500]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # → [64, 750]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), # → [128, 375]
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → [128, 1]
        )

        self.mlp = nn.Sequential(
            nn.Linear(128, emb_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(emb_hidden_dim, emb_dim)
        )

    def forward(self, data):
        x = data.x  # [num_nodes, seq_len]
        x = x.unsqueeze(1)  # [num_nodes, 1, seq_len]
        features = self.conv(x).squeeze(-1)  # [num_nodes, 128]
        out = self.mlp(features)  # [num_nodes, emb_dim]
        return out

