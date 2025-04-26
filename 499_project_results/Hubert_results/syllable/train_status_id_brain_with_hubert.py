import os
import sys
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
import torch
torch.cuda.empty_cache()
class StatusIdBrainHubert(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + emotion classifier."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        outputs = self.modules.hubert(wavs, lens)
        outputs = self.hparams.avg_pool(outputs, lens)
        outputs = outputs.view(outputs.shape[0], -1)
        if stage == sb.Stage.TRAIN:
            outputs = self.modules.dropout(outputs)
        outputs = self.modules.output_mlp(outputs)
        outputs = self.hparams.log_softmax(outputs)
        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label."""
        statusid, _ = batch.status_encoded
        statusid = statusid.squeeze(1)
        class_weights = torch.tensor([1.0, 2.0], device=self.device)
        loss = self.hparams.compute_cost(predictions, statusid, weight=class_weights)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, statusid)
        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch."""
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
            }

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            old_lr_hubert, new_lr_hubert = self.hparams.lr_annealing_hubert(stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(self.hubert_optimizer, new_lr_hubert)

            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr, "hubert_lr": old_lr_hubert},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["error_rate"]
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def init_optimizers(self):
        self.hubert_optimizer = self.hparams.hubert_opt_class(
            self.modules.hubert.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("hubert_opt", self.hubert_optimizer)
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

        self.optimizers_dict = {
            "model_optimizer": self.optimizer,
            "hubert_optimizer": self.hubert_optimizer,
        }

def dataio_prep(hparams):
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        max_len = 60 * 16000  # 10 seconds at 16kHz
        if len(sig) > max_len:
          sig = sig[:max_len]
        return sig


    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    @sb.utils.data_pipeline.takes("status")
    @sb.utils.data_pipeline.provides("status", "status_encoded")
    def label_pipeline(status):
        yield status
        status_encoded = label_encoder.encode_label_torch(status)
        yield status_encoded

    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "status_encoded"],
        )

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="status",
    )

    return datasets

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from status_cap_prepare import prepare_json_data  # noqa E402

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        sb.utils.distributed.run_on_main(
            prepare_json_data,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_folder": hparams["output_folder"],
                "skip_prep": hparams["skip_prep"]
            },
        )

    datasets = dataio_prep(hparams)

    hparams["hubert"] = hparams["hubert"].to(device=run_opts["device"])
    if not hparams["freeze_hubert"] and hparams["freeze_hubert_conv"]:
        hparams["hubert"].model.feature_extractor._freeze_parameters()

    status_id_brain_hubert = StatusIdBrainHubert(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    status_id_brain_hubert.fit(
        epoch_counter=status_id_brain_hubert.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    test_stats = status_id_brain_hubert.evaluate(
        test_set=datasets["test"],
        min_key="error_rate",
        test_loader_kwargs=hparams["dataloader_options"],
    )
