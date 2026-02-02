import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import math

# 数据集定义
class CIFAR10Dataset(Dataset):
    def __init__(self, data_path, labels_path=None):
        self.data_path = data_path
        
        self.data_name = []
        self.data_labels = []
        if labels_path:
            self.labels = True
            csv = pd.read_csv(labels_path)
            self.data_name = [f'{name}.png' for name in csv['id'].tolist()]
            self.data_labels = csv['label'].tolist()
            self.labels_set = list(set(self.data_labels))
            self.labels_map = {label: idx for idx, label in enumerate(self.labels_set)}
            self.data_labels = [self.labels_map[label] for label in self.data_labels]
        else:
            self.labels = False
            self.data_name = sorted([name for name in os.listdir(data_path) if name.endswith('.png')])

        self.len = len(self.data_name)

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        path = os.path.join(self.data_path, self.data_name[index])
        img = Image.open(path).convert('RGB')

        img = self.transform(img)

        return (img, self.data_labels[index]) if self.labels else img
    
    def get_labels_set(self):
        return self.labels_set

# 点积注意力
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)

# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        output = self.attention(queries, keys, values)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

# 前馈神经网络
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
    
# 残差连接与层归一化
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
    
# Encoder 块
class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X):
        Y = self.addnorm1(X, self.attention(X, X, X))
        return self.addnorm2(Y, self.ffn(Y))
    
# 图片处理
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, num_hiddens):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(in_chans, num_hiddens, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
    
# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = PatchEmbedding(img_size, patch_size, in_chans, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module('blocks'+str(i), EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias))


    def forward(self, X):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
    
# MLP 统计答案
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 整体网络定义
class Net(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, num_hiddens, num_heads, num_layers, num_types, dropout, use_bias=False):
        super(Net, self).__init__()
        sentence_lens = (img_size // patch_size) ** 2
        self.encoder = TransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            key_size=num_hiddens,
            query_size=num_hiddens,
            value_size=num_hiddens,
            num_hiddens=num_hiddens,
            norm_shape=[num_hiddens],
            ffn_num_input=num_hiddens,
            ffn_num_hiddens=num_hiddens*4,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_bias=False
        )
        self.mlp = MLP(num_hiddens * sentence_lens, num_hiddens, num_types)

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = self.mlp(x)
        return x
    
# 数据加载
batch_size = 50
training_epochs = 200

train_dataset = CIFAR10Dataset(data_path='data/kaggle_cifar10_tiny/train', labels_path='data/kaggle_cifar10_tiny/trainLabels.csv')
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
test_dataset = CIFAR10Dataset(data_path='data/kaggle_cifar10_tiny/test')
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# 超参数定义
img_size = 32
patch_size = 4
in_chans = 3
num_hiddens = 256
num_heads = 4
num_layers = 2
num_types = len(train_dataset.get_labels_set())
dropout = 0.1

# 模型及训练测试流程
model = Net(img_size, patch_size, in_chans, num_hiddens, num_heads, num_layers, num_types, dropout)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, data in enumerate(train_loader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 5 == 4:
            print('[%d, %4d] loss: %.6f' % (epoch + 1, (batch_idx + 1) * batch_size, running_loss / 5))
            running_loss = 0.0

def test():
    model.eval()
    epoch = 0

    with torch.no_grad():
        for data in test_loader:
            imgs = data.to(device)
            labels = model(imgs)
            for label in labels.argmax(dim=1).cpu().numpy():
                epoch += 1
                print(f'#{epoch}: {train_dataset.get_labels_set()[label]}')

if __name__ == '__main__':
    print('--- Training ---')
    for epoch in range(training_epochs):
        train(epoch)
    
    print()
    
    print('--- Testing ---')
    test()