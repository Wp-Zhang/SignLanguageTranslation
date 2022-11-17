import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .modules import Decoder, SeqKD, BiLSTMLayer, TemporalConv
from .evaluation import evaluate
from pytorch_lightning import LightningModule
import os
import torch.distributed as dist


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
            # "framewise_features": framewise,
            # "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs["conv_logits"],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }


class SLR_Lightning(LightningModule):
    def __init__(
        self,
        # * Model args
        num_classes,
        c2d_type,
        conv_type,
        use_bn,
        hidden_size,
        weight_norm,
        share_classifier,
        # * Training args
        optimizer,
        base_lr,
        step,
        learning_ratio,
        weight_decay,
        start_epoch,
        nesterov,
        loss_weights,
        gloss_dict,
        # * Evaluation args
        eval_script_dir,
        eval_output_dir,
        eval_label_dir,
    ):
        super().__init__()

        # * Define model
        self.model = SLRModel(
            num_classes,
            c2d_type,
            conv_type,
            use_bn,
            hidden_size,
            gloss_dict,
            weight_norm,
            share_classifier,
        )

        # * Define loss func
        self.loss = {}
        self.loss["CTCLoss"] = torch.nn.CTCLoss(reduction="none", zero_infinity=False)
        self.loss["distillation"] = SeqKD(T=8)

        # * Training cfg
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.step = step
        self.learning_ratio = learning_ratio
        self.weight_decay = weight_decay
        self.start_epoch = start_epoch
        self.nesterov = nesterov
        self.loss_weights = loss_weights

        # * Evaluation cfg
        self.eval_script_dir = eval_script_dir
        self.eval_output_dir = eval_output_dir
        self.eval_label_dir = eval_label_dir

    def calc_loss(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
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
        vid, vid_lgt, label, label_lgt, info = batch
        ret_dict = self.model(vid, vid_lgt, label, label_lgt)
        loss = self.calc_loss(ret_dict, label, label_lgt)
        self.log("train_loss", loss, batch_size=vid.size(0), on_step=True)
        return loss

    def eval_step(self, batch, stage):
        vid, vid_lgt, label, label_lgt, info = batch
        ret_dict = self.model(vid, vid_lgt, label, label_lgt)

        loss = self.calc_loss(ret_dict, label, label_lgt)

        self.log(
            f"{stage}_loss", loss, prog_bar=True, batch_size=vid.size(0), sync_dist=True
        )

        info = [filename.split("|")[0] for filename in info]
        ret_dict["info"] = info
        return ret_dict

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, "dev")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, "test")

    def eval_end(self, device_out, stage):
        if self.trainer.num_devices > 1:
            dist.barrier()
            useful_info = [
                {k: output[k] for k in ["info", "recognized_sents", "conv_sents"]}
                for output in device_out
            ]
            full_out = [None for _ in self.trainer.device_ids]
            dist.all_gather_object(full_out, useful_info)
        else:
            full_out = [device_out]

        if self.global_rank == 0:

            # if self.trainer.num_devices > 1:
            #     print(len(outputs))
            #     outputs = self.all_gather(outputs)
            #     print(len(outputs))

            total_info = []
            total_sent = []
            total_conv_sent = []
            for outputs in full_out:
                for out in outputs:
                    total_info.extend(out["info"])
                    total_sent.extend(out["recognized_sents"])
                    total_conv_sent.extend(out["conv_sents"])

            try:

                def write2file(path, info, output):
                    filereader = open(path, "w")
                    for sample_idx, sample in enumerate(output):
                        for word_idx, word in enumerate(sample):
                            filereader.writelines(
                                "{} 1 {:.2f} {:.2f} {}\n".format(
                                    info[sample_idx],
                                    word_idx * 1.0 / 100,
                                    (word_idx + 1) * 1.0 / 100,
                                    word[0],
                                )
                            )

                if not os.path.exists(self.eval_output_dir):
                    os.makedirs(self.eval_output_dir)

                write2file(
                    os.path.join(self.eval_output_dir, f"out-hypothesis-{stage}.ctm"),
                    total_info,
                    total_sent,
                )
                write2file(
                    os.path.join(
                        self.eval_output_dir, f"out-hypothesis-{stage}-conv.ctm"
                    ),
                    total_info,
                    total_conv_sent,
                )
                conv_ret = evaluate(
                    prefix=self.eval_output_dir,
                    mode=stage,
                    output_file="out-hypothesis-{}-conv.ctm".format(stage),
                    evaluate_dir=self.eval_script_dir,
                    evaluate_prefix="groundtruth",
                    label_dir=self.eval_label_dir,
                    output_dir="result/",
                    python_evaluate=True,
                )
                # lstm_ret = evaluate(
                #     prefix=self.eval_output_dir,
                #     mode=stage,
                #     output_file="out-hypothesis-{}.ctm".format(stage),
                #     evaluate_dir=self.eval_script_dir,
                #     evaluate_prefix="groundtruth",
                #     label_dir=self.eval_label_dir,
                #     output_dir="result/",
                #     python_evaluate=True,
                #     triplet=True,
                # )
                self.log(
                    f"{stage}_WER",
                    conv_ret,
                    # sync_dist=True,
                    on_epoch=True,
                )
                return {"WER": conv_ret}
            except:
                pass
                # self.log("Unexpected error:", sys.exc_info()[0], sync_dist=True)

    def validation_epoch_end(self, outputs):
        return self.eval_end(outputs, "dev")

    def test_epoch_end(self, outputs):
        return self.eval_end(outputs, "test")

    def configure_optimizers(self):
        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.base_lr,
                momentum=0.9,
                nesterov=self.nesterov,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.base_lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError()

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.step, gamma=0.2
        )
        return [optimizer], [lr_scheduler]
