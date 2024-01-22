from torchvision import transforms
import torchvision
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import optim


#UINT 8
train_path = '/home/michael/Downloads/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train'
transform = transforms.Compose(
    [transforms.Resize((224, 224)),transforms.ToTensor()]
)
imagenet_data = torchvision.datasets.ImageFolder(train_path, transform=transform)
data_loader = torch.utils.data.DataLoader(
    imagenet_data,
    batch_size=64,
    shuffle=True,
    num_workers=0
)


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 96, (11, 11), 4)
        self.relu1 = torch.nn.ReLU()
        self.maxPool1 = torch.nn.MaxPool2d((3, 3), stride=2)
        self.bn1 = torch.nn.BatchNorm2d(96)
        self.conv2 = torch.nn.Conv2d(96, 256, (5,5), 2, padding=2)
        self.relu2 = torch.nn.ReLU()
        self.maxPool2 = torch.nn.MaxPool2d((3, 3), stride=2)
        self.bn2 = torch.nn.BatchNorm2d(256)
        self.conv3 = torch.nn.Conv2d(256, 384, (3,3), padding=1)
        self.relu3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(384, 256, (3, 3), padding=1)
        self.relu4 = torch.nn.ReLU()
        self.maxPool3 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=2)
        self.bn3 = torch.nn.BatchNorm2d(256)
        #reshape
        self.linear1 = torch.nn.Linear(1024, 1024),
        self.dropout1 = torch.nn.Dropout1d(),
        self.relu5 = torch.nn.ReLU(),
        self.linear2 = torch.nn.Linear(1024, 1000)

    def forward(self, x:Tensor):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxPool1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxPool2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxPool3(x)
        x = self.bn3(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.relu5(x)
        x = self.linear2(x)
        return x
device=torch.device("cuda:2")
net = Model().to(device)
net.train()
optimizer = optim.Adadelta(net.parameters(), lr=0.001)
i = 0
for x, y in data_loader:
    i += 1
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    output = net(x)
    loss = F.nll_loss(output, y)
    loss.backward()
    optimizer.step()
    print("Step:{i} Loss:{loss}")