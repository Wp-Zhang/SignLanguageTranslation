import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .modules import Decoder, SeqKD, BiLSTMLayer, TemporalConv
from .evaluation import evaluate
from pytorch_lightning import LightningModule
import numpy as np


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
        c2d_type,
        conv_type,
        use_bn=False,
        hidden_size=1024,
        gloss_dict=None,
        weight_norm=True,
        share_classifier=True,
    ):
        super(SLRModel, self).__init__()

        self.num_classes = num_classes
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d.fc = Identity()
        self.conv1d = TemporalConv(
            input_size=512,
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
            num_layers=2,
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
        self.register_backward_hook(self.backward_hook)

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
        x = self.conv2d(x)
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
            "framewise_features": framewise,
            "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs["conv_logits"],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }


class SLR_Lightning(LightningModule):
    def __init__(self, model_cfg, training_cfg, gloss_dict):
        super().__init__()

        self.training_cfg = training_cfg

        # * Define model
        self.model = SLRModel(gloss_dict=gloss_dict, **model_cfg)

        # * Define loss func
        self.loss = {}
        self.loss["CTCLoss"] = torch.nn.CTCLoss(reduction="none", zero_infinity=False)
        self.loss["distillation"] = SeqKD(T=8)

    def calc_loss(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.training_cfg.loss_weights.items():
            if k == "ConvCTC":
                loss += (
                    weight
                    * self.loss["CTCLoss"](
                        ret_dict["conv_logits"].log_softmax(-1),
                        label.cpu().int(),
                        ret_dict["feat_len"].cpu().int(),
                        label_lgt.cpu().int(),
                    ).mean()
                )
            elif k == "SeqCTC":
                loss += (
                    weight
                    * self.loss["CTCLoss"](
                        ret_dict["sequence_logits"].log_softmax(-1),
                        label.cpu().int(),
                        ret_dict["feat_len"].cpu().int(),
                        label_lgt.cpu().int(),
                    ).mean()
                )
            elif k == "Dist":
                loss += weight * self.loss["distillation"](
                    ret_dict["conv_logits"],
                    ret_dict["sequence_logits"].detach(),
                    use_blank=False,
                )
        return loss

    def training_step(self, batch, batch_idx):
        vid, vid_lgt, label, label_lgt = batch
        ret_dict = self.model(vid, vid_lgt, label, label_lgt)
        loss = self.calc_loss(ret_dict, label, label_lgt)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        vid, vid_lgt, label, label_lgt = batch
        ret_dict = self.model(vid, vid_lgt, label, label_lgt)

        # try:
        #     conv_ret = evaluate(
        #         prefix=self.eval_cfg.work_dir,
        #         mode=stage,
        #         output_file="output-hypothesis-{}-conv.ctm".format(stage),
        #         evaluate_dir=self.eval_cfg.evaluation_dir,
        #         evaluate_prefix=self.eval_cfg.evaluation_prefix,
        #         label_dir=self.eval_cfg.label_dir,
        #         output_dir="result/",
        #         python_evaluate=self.eval_cfg.python_eval,
        #     )
        #     lstm_ret = evaluate(
        #         prefix=self.eval_cfgwork_dir,
        #         mode=stage,
        #         output_file="output-hypothesis-{}.ctm".format(stage),
        #         evaluate_dir=self.eval_cfg.evaluation_dir,
        #         evaluate_prefix=self.eval_cfg.evaluation_prefix,
        #         label_dir=self.eval_cfg.label_dir,
        #         output_dir="result/",
        #         python_evaluate=self.eval_cfg.python_eval,
        #         triplet=True,
        #     )
        # except:
        #     self.log("Unexpected error:", sys.exc_info()[0])
        # finally:
        #     pass

        loss = self.calc_loss(ret_dict, label, label_lgt)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "dev")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        if self.training_cfg.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.training_cfg.lr,
                momentum=0.9,
                nesterov=self.training_cfg.nesterov,
                weight_decay=self.training_cfg.weight_decay,
            )
        elif self.training_cfg.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.training_cfg.lr,
                weight_decay=self.training_cfg.weight_decay,
            )
        else:
            raise ValueError()

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 0.1, milestones=self.training_cfg.step, gamma=0.2
        )
        return [optimizer], [lr_scheduler]
