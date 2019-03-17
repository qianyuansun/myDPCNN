import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class DPCNN(BasicModule):
    """
    DPCNN for sentences classification.
    """
    def __init__(self, config):
        super(DPCNN, self).__init__()
        input_dropout_p = float(0.05)
        self.config = config
        class_num = self.config.label_num  #different for each dataset
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.channel_size = 250
        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, self.config.word_embedding_dimension), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(self.channel_size, class_num)

    def forward(self, x):
        batch = x.shape[0]
        x = self.input_dropout(x)
        batch, width, height = x.shape
        x = x.view((batch, 1, width, height))
        #x = x.view(x.size(0), 1, x.size(1), x.size(2))

        # Region embedding
        try:
            x = self.conv_region_embedding(x)        # [batch_size, channel_size, length, 1]
        except:
            print(x.shape)

        x = self.padding_conv(x)                      # pad保证等长卷积，先通过激活函数再卷积
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        count = 0
        while count < 6:
            while x.size()[-2] >= 2:
                x = self._block(x)
            count += 1


        x = x.view(batch, self.channel_size)
        x = self.linear_out(x)

        return x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x
    '''

    def predict(self, x):
        #self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        print(predict_labels)
        #self.train(mode=True)
        return predict_labels
        '''

