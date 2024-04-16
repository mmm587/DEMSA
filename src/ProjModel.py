import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLayer(nn.Module):
    def __init__(self, args):
        super(ClassificationLayer, self).__init__()
        self.cross_attention_1 = nn.MultiheadAttention(ConCat_dim, Heads)
        self.dropout = nn.Dropout(dropouts)
        self.LayerNorm = nn.LayerNorm(ConCat_dim)
        self.fc1 = nn.Linear(ConCat_dim, Hidden_dim) 
        self.fc2 = nn.Linear(Hidden_dim, args.output_dim)  

    def forward(self, denoised_text, text):
        text_de, _ = self.cross_attention_1(text, denoised_text, denoised_text)
        text_de = self.dropout(text_de)
        text_de = self.LayerNorm(text_de + text)
        x = F.relu(self.fc1(text_de))
        x = self.fc2(x)
        return x
