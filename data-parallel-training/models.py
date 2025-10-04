# models.py
import torch.nn as nn

class LinearNet(nn.Module):
    def __init__(self, hidden_layer_sizes=[128, 256, 128]):
        super().__init__()
        layers = []
        for i, hidden_size in enumerate(hidden_layer_sizes):
            if i == 0:
                layers += [nn.Flatten()]
                layers += [nn.Linear(93, hidden_size)]   # 93 = census feature dim
                layers += [nn.Tanh()]
            else:
                layers += [nn.Linear(hidden_layer_sizes[i - 1], hidden_size)]
                layers += [nn.Tanh()]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_layer_sizes[-1], 2)

    def forward(self, x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)
