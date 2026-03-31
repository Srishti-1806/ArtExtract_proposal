import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, global_max_pool

class GCNBackbone(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNBackbone, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn2 = BatchNorm(out_channels)
        self.lrelu = nn.LeakyReLU()
        # self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.lrelu(x)
        # x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.lrelu(x)
        # x = self.dropout(x)
        
        graph_embedding = global_max_pool(x, batch) 
        
        return graph_embedding # return graph-level embedding

class GCNSiameseNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNSiameseNetwork, self).__init__()
        self.gcn = GCNBackbone(in_channels, hidden_channels, out_channels)

    def forward(self, graph1, graph2):
        # Forward pass for the first graph
        embedding1 = self.gcn(graph1)
        
        # Forward pass for the second graph
        embedding2 = self.gcn(graph2)
        
        return embedding1, embedding2

class GATBackbone(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GATBackbone, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=heads)
        self.conv3 = GATConv(out_channels * heads, out_channels, heads=heads)
        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.lrelu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.lrelu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.lrelu(x)
        x = self.dropout(x)

        graph_embedding = global_max_pool(x, batch) 
        
        return graph_embedding, x, edge_index

class GATSiameseNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GATSiameseNetwork, self).__init__()
        self.gat = GATBackbone(in_channels, hidden_channels, out_channels, heads)

    def forward(self, graph1, graph2):
        # Forward pass for the first graph
        emb1, node1, edge1 = self.gat(graph1)
        
        # Forward pass for the second graph
        emb2, node2, edge2 = self.gat(graph2)
        
        return emb1, emb2, node1, node2, edge1, edge2