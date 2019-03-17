import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule

class CNN(BasicModule):
    """
    CNN for sentences classification.
    """
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        class_num = self.config.label_num   #different for each dataset
        self.dropout = nn.Dropout(0.5)
        self.kernel_num = 100
        self.kernel_sizes = [3,4,5]
        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (kernel, self.config.word_embedding_dimension),padding=(kernel-1,0))
                                    for kernel in self.kernel_sizes])
        self.linear_out = nn.Linear(len(self.kernel_sizes) * self.kernel_num, class_num)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        xs = []
        for conv in self.convs:  # Apply a convolution + max pool layer for each window size
            x2 = F.relu(conv(x))  # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2)) # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)  # [B, F, window]
        x = x.view(x.size(0), -1)  # [B, F * window]
        x = self.dropout(x)
        logits = self.linear_out(x)  # [B, class]
        return logits

    def predict(self, x):
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        self.train(mode=True)
        return predict_labels
