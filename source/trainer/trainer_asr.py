import random
from pathlib import Path
from random import shuffle
from typing import Union
import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from source.base import BaseTrainer
from source.base.base_text_encoder import BaseTextEncoder
from source.logger.utils import plot_spectrogram_to_buf
from source.metric.asr.utils import calc_cer, calc_wer
from source.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            dataset,
            text_encoder,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, lr_scheduler, config, device)
        self.skip_oom = skip_oom
        self.text_encoder = text_encoder
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.dataset = dataset
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.log_step = config["trainer"].get("log_step", 50)
        self.n_batches_accumulation = config["trainer"].get("n_batches_accumulation", 1)
        self.beam_size = config["trainer"].get("beam_size", 100)

        if "steps_per_epoch" in config["lr_scheduler"]["args"] and self.len_epoch is not None:
            assert self.n_batches_accumulation * config["lr_scheduler"]["args"]["steps_per_epoch"] == self.len_epoch
        if "T_max" in config["lr_scheduler"]["args"] and self.len_epoch is not None:
            assert self.len_epoch * (self.epochs + 1) == config["lr_scheduler"]["args"]["T_max"]

        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.metrics if not m.only_val], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "text_encoded"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.writer.set_step((epoch - 1) * self.len_epoch)
        self.writer.add_scalar("epoch", epoch)
        batch_ind_in_batch_accumulation = 0
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            # increment the batch index count
            if not batch:
                continue  # Skip errors in dataloader
            batch_ind_in_batch_accumulation += 1
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                    batch_ind_in_batch_accumulation=batch_ind_in_batch_accumulation
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if batch_ind_in_batch_accumulation == self.n_batches_accumulation:
                batch_ind_in_batch_accumulation = 0
            if batch_ind_in_batch_accumulation == self.n_batches_accumulation or batch_idx % self.log_step == 0:
                self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_predictions(**batch)
                self._log_audio(batch)
                self._log_spectrogram(batch)
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker, batch_ind_in_batch_accumulation: Union[int, None] = None):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train and batch_ind_in_batch_accumulation == 1:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["logits"] = outputs

        batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
        batch["log_probs_length"] = self.model.transform_input_lengths(batch["spectrogram_length"])
        batch["loss"] = self.criterion(**batch)
        if is_train:
            # gradient accumulation: divide loss by number of batches
            (batch["loss"] / self.n_batches_accumulation).backward()
            if batch_ind_in_batch_accumulation >= self.n_batches_accumulation:
                self._clip_grad_norm()
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        metrics.update("loss", batch["loss"].item())
        for met in self.metrics:
            if not is_train or not met.only_val:
                metrics.update(met.name, met(**batch))
        if not torch.isfinite(batch["loss"]).item():
            raise RuntimeError(f'Non-finite loss: {batch["loss"].item()}')
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                if not batch:
                    continue  # Skip errors in dataloader
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch)
            self._log_audio(batch)
            self._log_spectrogram(batch)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _get_random_log_ind(self, batch):
        return random.choice(range(len(batch["audio_path"])))

    def _log_audio(self, batch):
        ind = self._get_random_log_ind(batch)

        # log original audio
        audio_path = Path(batch["audio_path"][ind])
        original_audio = self.dataset.load_audio(audio_path)
        self.writer.add_audio("audio_original", original_audio, sample_rate=self.config["preprocessing"]["sr"])

        # log transformed audio
        audio = batch["audio"][ind].cpu()
        wave_augs = batch["wave_augs"][ind]
        self.writer.add_audio("audio" + (f"_({(' '.join(wave_augs))})" if wave_augs else ""), audio, sample_rate=self.config["preprocessing"]["sr"])

    def _log_spectrogram(self, batch):
        ind = self._get_random_log_ind(batch)
        spectrogram = batch["spectrogram"][ind].cpu()
        spectrogram_augs = batch["spec_augs"][ind]
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram" + (f"_({(' '.join(spectrogram_augs))})" if spectrogram_augs else ""), ToTensor()(image))

    def _log_predictions(
            self,
            text,
            log_probs,
            log_probs_length,
            audio_path,
            examples_to_log=10,
            beam_search_examples_to_log=10,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return
        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode_enhanced(inds) for inds in argmax_inds]
        tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path))

        # Shuffle results
        ind = list(range(len(tuples)))
        shuffle(ind)
        tuples = [tuples[i] for i in ind]
        rows = {}
        for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
            target = BaseTextEncoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100
            rows[Path(audio_path).name] = {
                "target": target,
                "prediction": pred,
                "raw prediction": raw_pred,
                "wer": wer,
                "cer": cer,
            }
        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

        # Beam search for the first sentence
        beam_search_hypotheses = self.text_encoder.ctc_beam_search(log_probs[ind[0]].detach().cpu(), log_probs_length[ind[0]].item(), self.beam_size)[:beam_search_examples_to_log]
        rows = {}
        pred, target, raw_pred, audio_path = tuples[0]
        target = BaseTextEncoder.normalize_text(target)
        for i, hypothesis in enumerate(beam_search_hypotheses):
            wer = calc_wer(target, hypothesis.text) * 100
            cer = calc_cer(target, hypothesis.text) * 100
            rows[i] = {
                "rank": i,
                "beam_search_pred": hypothesis.text,
                "probability": hypothesis.prob,
                "wer": wer,
                "cer": cer,
                "target": target,
                "prediction": pred,
                "raw_prediction": raw_pred,
            }
        self.writer.add_table(f"beam_search ({self.beam_size})", pd.DataFrame.from_dict(rows, orient="index"))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))