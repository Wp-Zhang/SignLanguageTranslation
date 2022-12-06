import os
from pathlib import Path
import torch
import torch.distributed as dist
from pytorch_lightning import LightningModule

from .modules import SeqKD
from .evaluation import evaluate
from .SLR_model import SLRModel


class SLR_Lightning(LightningModule):
    def __init__(
        self,
        # * Model args
        num_classes,
        backbone,
        conv_type,
        use_bn,
        hidden_size,
        temporal_layer_num,
        weight_norm,
        share_classifier,
        # * Training args
        optimizer,
        base_lr,
        step,
        weight_decay,
        nesterov,
        loss_weights,
        gloss_dict,
        # * Evaluation args
        eval_script_dir,
        eval_output_dir,
        eval_label_dir,
    ):
        super().__init__()

        self.save_hyperparameters()

        # * Define model
        self.model = SLRModel(
            num_classes,
            backbone,
            conv_type,
            use_bn,
            hidden_size,
            temporal_layer_num,
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
        self.weight_decay = weight_decay
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
        vid, vid_lgt, label, label_lgt, _ = batch
        ret_dict = self.model(vid, vid_lgt, label, label_lgt)
        loss = self.calc_loss(ret_dict, label, label_lgt)
        self.log("train_loss", loss, batch_size=vid.size(0), on_step=True)
        scheduler = self.lr_schedulers()
        self.log(
            "lr",
            scheduler.get_last_lr()[0],
            batch_size=vid.size(0),
            on_step=True,
        )
        return loss

    def eval_step(self, batch, stage):
        vid, vid_lgt, label, label_lgt, name = batch
        ret_dict = self.model(vid, vid_lgt, label, label_lgt)

        loss = self.calc_loss(ret_dict, label, label_lgt)

        self.log(
            f"{stage}_loss", loss, prog_bar=True, batch_size=vid.size(0), sync_dist=True
        )

        ret_dict["name"] = name
        return ret_dict

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, "dev")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, "test")

    def eval_end(self, device_out, stage):
        if self.trainer.num_devices > 1:
            dist.barrier()
            useful_info = [
                {k: output[k] for k in ["name", "recognized_sents", "conv_sents"]}
                for output in device_out
            ]
            full_out = [None for _ in self.trainer.device_ids]
            dist.all_gather_object(full_out, useful_info)
        else:
            full_out = [device_out]

        if self.global_rank == 0:
            full_name = []
            full_sent = []
            full_conv_sent = []
            for outputs in full_out:
                for out in outputs:
                    full_name.extend(out["name"])
                    full_sent.extend(out["recognized_sents"])
                    full_conv_sent.extend(out["conv_sents"])

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
                    full_name,
                    full_sent,
                )
                write2file(
                    os.path.join(
                        self.eval_output_dir, f"out-hypothesis-{stage}-conv.ctm"
                    ),
                    full_name,
                    full_conv_sent,
                )
                conv_ret = evaluate(
                    prefix=str(Path(self.eval_output_dir)) + "/",
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
                self.log(f"{stage}_WER", conv_ret, prog_bar=True)
                return {f"{stage}_WER": conv_ret}
            except:
                pass

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
