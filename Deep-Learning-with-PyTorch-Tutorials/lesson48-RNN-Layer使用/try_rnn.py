import torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F

#from 'hello' to 'ohlol'
#对hello进行one-hot
#e\h\l\o分别对应0、1、2、3四个类别，变为分类问题求交叉熵
input_size = 4
hidden_size = 4
output_size = 1
batch_size = 1
lr=0.1

idx2char = ['e','h','l','o']
x_data = [1,0,2,2,3]
y_data = [3,1,2,3,2]
one_hot_lookup = [[1,0,0,0],[0,1, 0,0], [0,0,1,0], [0,0,0,1]]
x_one_hot= [one_hot_lookup[x] for x in x_data]
inputs = torch.Tensor(x_one_hot).view(-1,batch_size,input_size)
# labels = torch.LongTensor(y_data).view(-1, 1)
labels = torch.LongTensor(y_data)
print(inputs.shape,labels.shape,labels[0])

## RNN_cell
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model,self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size,hidden_size=self.hidden_size)
    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden
    def init_hidden(self):
        return torch.zeros(self.batch_size,self.hidden_size)
net = Model(input_size, hidden_size, batch_size)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(),lr)


## RNN

class Model1(torch.nn.Module):
    def __init__(self,input_size,hidden_size, batch_size,num_layers):
        super(Model1,self).__init__()
        self. num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size,hidden_size=self.hidden_size, num_layers=num_layers)
    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.batch_size, self. hidden_size)
        out, h = self.rnn(input, hidden)
        print(self.rnn)
        print(out.shape,h.shape)
        return out.view(-1, self.hidden_size)
net1 = Model1(input_size,hidden_size,batch_size, num_layers=1)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net1.parameters(),lr=0.05)
for epoch in range(15):
    optimizer.zero_grad()
    outputs = net1(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _,idx = outputs.max(dim=1)
    idx = idx.data.numpy()

    print(''.join([idx2char[x] for x in idx]), end='')
    print(',Epoch [%d/15] loss=%.3f' % (epoch+1, loss.item()))