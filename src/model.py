import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        """DNNの層を定義
        """
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=16, kernel_size=3, padding=2, stride=2, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=16, kernel_size=3, padding=2, stride=2, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.LazyLinear(out_features=num_classes)
        
    
    def forward(self, x):
        """DNNの入力から出力までの計算
        Args:
            x: torch.Tensor whose size of
               (batch size, # of channels, # of freq. bins, # of time frames)
        Return:
            y: torch.Tensor whose size of
               (batch size, # of classes)
        """
        x = self.net(x)
        y = self.classifier(x.view(x.size(0), -1))
        return y
