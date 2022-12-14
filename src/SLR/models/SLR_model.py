import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm import create_model

from .modules import Decoder, BiLSTMLayer, TemporalConv


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone,
        conv_type,
        use_bn=False,
        hidden_size=1024,
        temporal_layer=2,
        gloss_dict=None,
        weight_norm=True,
        share_classifier=True,
    ):
        super(SLRModel, self).__init__()

        self.num_classes = num_classes

        if backbone in ["resnet18", "resnet50",'resnet101']:
            self.backbone = getattr(models, backbone)(pretrained=True)
            out_dim = self.backbone.fc.in_features
            self.backbone.fc = Identity()
        elif backbone in ["swin_tiny_patch4_window7_224","convnext_tiny_in22k",'mobilevit_xxs']:
            self.backbone = create_model(
                backbone, pretrained=True, num_classes=0, in_chans=3
            )
            out_dim = self.backbone.num_features

        self.conv1d = TemporalConv(
            input_size=out_dim,
            hidden_size=hidden_size,
            conv_type=conv_type,
            use_bn=use_bn,
            num_classes=num_classes,
        )

        self.decoder = Decoder(gloss_dict, num_classes, "beam")
        self.temporal_model = BiLSTMLayer(
            rnn_type="LSTM",
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=temporal_layer,
            bidirectional=True,
        )
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        self.register_full_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat(
                [
                    tensor,
                    tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_(),
                ]
            )

        x = torch.cat(
            [
                inputs[len_x[0] * idx : len_x[0] * idx + lgt]
                for idx, lgt in enumerate(len_x)
            ]
        )
        x = self.backbone(x)
        x = torch.cat(
            [
                pad(x[sum(len_x[:idx]) : sum(len_x[: idx + 1])], len_x[0])
                for idx, lgt in enumerate(len_x)
            ]
        )
        return x

    def forward(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            framewise = self.masked_bn(inputs, len_x)
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
        else:
            # frame-wise features
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs["visual_feat"]
        lgt = conv1d_outputs["feat_len"]
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs["predictions"])
        pred = (
            None
            if self.training
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        )
        conv_pred = (
            None
            if self.training
            else self.decoder.decode(
                conv1d_outputs["conv_logits"], lgt, batch_first=False, probs=False
            )
        )

        return {
            # "framewise_features": framewise,
            # "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs["conv_logits"],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }
