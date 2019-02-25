import torch
import torch.nn as nn
import torch.nn.functional as F

from contrib.criterions.vgg_loss   import VGGLoss


class NNLoss(nn.Module):
    def __init__(self, neib_size=(3, 3), max_vgg_layers=3):
        super().__init__()
        self.vgg = VGGLoss().cuda()
        self.neib_size = neib_size
        self.v_pad = self.neib_size[0] // 2
        self.h_pad = self.neib_size[1] // 2
        self.max_vgg_layers = max_vgg_layers

    def get_vgg_features(self, x):
        x = self.vgg.normalize_inputs(x)
        layer_features = []
        for i, layer in enumerate(self.vgg.vgg19):
            if i == self.max_vgg_layers:
                break

            x = layer(x)

            if layer.__class__.__name__ == 'ReLU':
                layer_features.append(x)

        return layer_features

    def forward(self, x, y):
        loss = 0
        for x_vgg, y_vgg in zip(self.get_vgg_features(x), self.get_vgg_features(y)):

            x_padded = F.pad(x_vgg, pad=(self.h_pad, self.h_pad, self.v_pad, self.v_pad))
            reference_tensors = []

            for i_begin in range(0, self.neib_size[0]):
                i_end = i_begin - self.neib_size[0] + 1
                i_end = None if i_end == 0 else i_end
                for j_begin in range(0, self.neib_size[1]):
                    j_end = j_begin - self.neib_size[0] + 1
                    j_end = None if j_end == 0 else j_end

                    sub_tensor = x_padded[:, :, i_begin:i_end, j_begin:j_end]
                    reference_tensors.append(sub_tensor)
            x_references = torch.stack(reference_tensors, dim=-1)
            abs_err = torch.abs(x_references - y_vgg[..., None]).sum(dim=1)
            loss += torch.min(abs_err, dim=-1)[0].mean()

        return loss
