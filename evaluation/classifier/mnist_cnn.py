from torch import nn
import torch
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, img_dim=(1,28,28), num_classes=10):
        super(CNN, self).__init__()
        assert img_dim[1] in [28,32]
        self.fe = torch.nn.Sequential(
            nn.Conv2d(img_dim[0], 32,3,1),  #26, 28
            nn.MaxPool2d(2,2),  # 13, 14
            nn.ReLU(),
            nn.Dropout(0.5), #0.5
            nn.Conv2d(32,64,3,1),   # 11, 12
            nn.MaxPool2d(2, 2), # 5, 6
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, 3, 1),   #3, 4
            nn.ReLU(),
            nn.Flatten()
        )
        if img_dim[1] == 28:
            self.cla = torch.nn.Sequential(
                nn.Linear(1152, 128),
                nn.ReLU(),
                #nn.Dropout2d(0.5),
                nn.Linear(128, num_classes),
            )
        elif img_dim[1] == 32:
            self.cla = torch.nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(),
                #nn.Dropout2d(0.5),
                nn.Linear(128, num_classes),

            )

    def forward(self, x):
        fes = self.fe(x)
        pred = self.cla(fes)
        return F.log_softmax(pred, dim=1)

    def pred(self, x):
        return F.softmax(self.cla(self.fe(x)), dim=1)