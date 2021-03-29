import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):
    def __init__(self, in_planes, kernel_size=3, k_up=5, up_factor=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.k_up = k_up
        self.up_factor = up_factor
        self.down = nn.Conv2d(in_planes, in_planes // 2, kernel_size=1)
        self.encoder = nn.Conv2d(in_planes // 2, self.up_factor ** 2 * self.k_up ** 2,
                                 kernel_size=kernel_size, padding=self.kernel_size // 2)
        self.out = nn.Conv2d(in_planes, in_planes, kernel_size=1)

        self._init_weight()

    def forward(self, input_tensor):
        N, C, H, W = input_tensor.size()

        # kernel prediction module
        kernel_tensor = self.down(input_tensor)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.k_up ** 2, H, W, self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        input_tensor = F.pad(input_tensor, pad=[self.k_up // 2, self.k_up // 2, self.k_up // 2, self.k_up // 2],
                          mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        input_tensor = input_tensor.unfold(2, self.k_up, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        input_tensor = input_tensor.unfold(3, self.k_up, step=1)  # (N, C, H, W, Kup, Kup)
        input_tensor = input_tensor.reshape(N, C, H, W, -1)  # (N, C, H, W, Kup^2)
        input_tensor = input_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        output_tensor = torch.matmul(input_tensor, kernel_tensor)
        output_tensor = output_tensor.reshape(N, H, W, -1)
        output_tensor = output_tensor.permute(0, 3, 1, 2)
        output_tensor = F.pixel_shuffle(output_tensor, self.up_factor)
        output_tensor = self.out(output_tensor)

        return output_tensor

    def _init_weight(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1)

                m.bias.data.zero_()


def build_carafe(in_planes, up_factor):
    return CARAFE(in_planes, up_factor=up_factor)


if __name__ == "__main__":
    input_tensor = torch.rand(2, 256, 17, 21)
    carafe = CARAFE(256)
    # output_tensor = carafe(input_tensor)
    # print(output_tensor.size())
    print(carafe)
