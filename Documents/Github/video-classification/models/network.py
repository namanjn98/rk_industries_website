import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

from models.network_helper import *

class VideoModel(nn.Module):
    def __init__(self, num_classes):
        """
            encoder(InceptionNet) + decoder(LSTM)
        """
        super(VideoModel, self).__init__()
        self.encoder = InceptionCNNEncoder(fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300)
        self.decoder = DecoderRNN(CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=num_classes)

    def forward(self, x_3d):
        encoded_sequences = self.encoder(x_3d)
        decoded_sequences = self.decoder(encoded_sequences)
        return(decoded_sequences)