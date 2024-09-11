import torch
import torch.nn as nn


class NTK(nn.Module):

    def __init__(self, input_size, hidden_size_1, output_size):
        super(NTK, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = 0
        self.output_size = output_size

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        self.relu = torch.nn.ReLU()
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size_1, elementwise_affine=False)
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.output_size)

    def forward(self, x):
        x = torch.flatten(x, 0)
        hidden = self.fc1(x)
        hidden = self.relu(hidden)
        hidden = self.layer_norm(hidden)
        output = self.fc2(hidden)

        return output


class LeNet5(nn.Module):

    def __init__(self, nodesNum1=200, nodesNum2=100, nodesFc1=50, nodesFc2=1):
        super(LeNet5, self).__init__()

        self.nodesNum2 = nodesNum2

        self.c1 = nn.Conv2d(1, nodesNum1, 5)
        self.s2 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(nodesNum1)
        self.c3 = nn.Conv2d(nodesNum1, nodesNum2, 5)
        self.s4 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(nodesNum2)
        self.c5 = nn.Linear(nodesNum2 * 4 * 4, nodesFc1)
        self.f6 = nn.Linear(nodesFc1, nodesFc2)
        self.out7 = nn.Linear(nodesFc2, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        output = self.c1(x)
        output = f.relu(self.s2(output))
        output = self.bn1(output)
        output = self.c3(output)
        output = f.relu(self.s4(output))
        output = self.bn2(output)
        output = output.view(-1, self.nodesNum2 * 4 * 4)
        output = self.c5(output)
        output = self.f6(output)
        output = self.out7(output)  # remove for 99.27 and 90.04 models

        return output
