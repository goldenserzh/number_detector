import torch.nn as nn
import pandas

class ConvModel(nn.Module):
  def __init__(self, verbose=False):
    super(ConvModel, self).__init__()
    self.layer_1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding = 2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer_2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    self.drop_out = nn.Dropout()
    self.fc1 = nn.Linear(7*7*64, 1000)
    self.fc2 = nn.Linear(1000, 10)
    self.verbose = verbose

  def forward(self, x):
    out = self.layer_1(x)
    out = self.layer_2(out)
    out = out.view(out.size(0), -1)
    out = self.drop_out(out)
    out = self.fc1(out)
    out = self.fc2(out)

    return out