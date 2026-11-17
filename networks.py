import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, in_ch, growth, n):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_ch
        for _ in range(n):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch, growth, 3, padding=1)
                )
            )
            ch += growth
        self.out_ch = ch

    def forward(self, x):
        feats = [x]
        for l in self.layers:
            feats.append(l(torch.cat(feats, 1)))
        return torch.cat(feats, 1)


class DenseUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ===== Encoder =====
        self.db1 = DenseBlock(3, 16, 3)
        self.td1 = nn.MaxPool2d(2)

        self.db2 = DenseBlock(self.db1.out_ch, 16, 4)
        self.td2 = nn.MaxPool2d(2)

        # ===== Bottleneck =====
        self.mid = DenseBlock(self.db2.out_ch, 16, 5)

        # ===== Decoder =====
        self.up2 = nn.ConvTranspose2d(self.mid.out_ch, self.db2.out_ch, 2, 2)
        self.db3 = DenseBlock(self.db2.out_ch * 2, 16, 4)

        self.up1 = nn.ConvTranspose2d(self.db3.out_ch, self.db1.out_ch, 2, 2)
        self.db4 = DenseBlock(self.db1.out_ch * 2, 16, 3)

        # ===== Output =====
        self.out = nn.Conv2d(self.db4.out_ch, 1, 1)

    def forward(self, x):
        e1 = self.db1(x)
        e2 = self.db2(self.td1(e1))
        b  = self.mid(self.td2(e2))

        d2 = self.up2(b)
        d2 = self.db3(torch.cat([d2, e2], 1))

        d1 = self.up1(d2)
        d1 = self.db4(torch.cat([d1, e1], 1))

        return self.out(d1)
