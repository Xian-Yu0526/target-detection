import torch.nn as nn
import torch

class Pnet(nn.Module):
    def __init__(self):
        super(Pnet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1),
            nn.BatchNorm2d(10),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.pre_layer2 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
        )
        self.pre_layer3 = nn.Sequential(
            nn.Conv2d(32, 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(4)
        )

    def forward(self, x):
        out1 = self.pre_layer(x)
        cond = torch.sigmoid(self.pre_layer2(out1))
        offset = self.pre_layer3(out1)
        return cond, offset



# class Pnet(nn.Module):
#     def __init__(self):
#         super(Pnet, self).__init__()
#         self.pre_layer = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=10, kernel_size=1, stride=1),
#             nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1),
#             nn.Conv2d(in_channels=10, out_channels=3, kernel_size=1, stride=1),
#             nn.BatchNorm2d(3),
#             nn.PReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1),
#             nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
#             nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1),
#             nn.BatchNorm2d(3),
#             nn.PReLU(),
#
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
#             nn.BatchNorm2d(32),
#             nn.PReLU(),
#         )
#         self.pre_layer2 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1),
#             nn.BatchNorm2d(1),
#         )
#         self.pre_layer3 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1),
#             nn.BatchNorm2d(4)
#         )
#
#     def forward(self, x):
#         out1 = self.pre_layer(x)
#         cond = torch.sigmoid(self.pre_layer2(out1))
#         offset = self.pre_layer3(out1)
#         return cond, offset


class R_net(nn.Module):
    def __init__(self):
        super(R_net, self).__init__()
        self.pre_layer1 = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),
            nn.BatchNorm2d(28),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(28, 48, kernel_size=3, stride=1),
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 64, kernel_size=2, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.pre_layer2 = nn.Sequential(
            nn.Linear(in_features=64*3*3, out_features=128),
            nn.BatchNorm1d(128),
            nn.PReLU())
        self.pre_layer3 = nn.Sequential(
            nn.Linear(in_features=128, out_features=1),
            nn.BatchNorm1d(1)
        )
        self.pre_layer4 = nn.Sequential(
            nn.Linear(in_features=128, out_features=4),
            nn.BatchNorm1d(4)
        )

    def forward(self, x):
        out1 = self.pre_layer1(x)
        out1 = out1.view(out1.size(0), -1)
        out2 = self.pre_layer2(out1)
        cond = torch.sigmoid(self.pre_layer3(out2))
        offset = self.pre_layer4(out2)
        return cond, offset


class O_net(nn.Module):
    def __init__(self):
        super(O_net, self).__init__()
        self.pre_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, stride=1, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2,padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.Conv2d(kernel_size=2, stride=1, in_channels=64, out_channels=128),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.pre_layer2 = nn.Sequential(
            nn.Linear(in_features=128*3*3, out_features=256),
            nn.BatchNorm1d(256),
            nn.PReLU()
        )
        self.pre_layer3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=1),
            nn.BatchNorm1d(1)
        )
        self.pre_layer4 = nn.Sequential(
            nn.Linear(in_features=256, out_features=4),
            nn.BatchNorm1d(4),
        )

    def forward(self, x):
        out1 = self.pre_layer1(x)
        out1 = out1.view(out1.size(0), -1)
        out2 = self.pre_layer2(out1)
        cond = torch.sigmoid(self.pre_layer3(out2))
        offset = self.pre_layer4(out2)
        return cond, offset