import argparse
import collections
import itertools
import warnings

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from ..trainer import Trainer
from ..utils import get_logger, prepare_device
from ..utils.object_loading import get_dataloaders

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


@hydra.main(config_path="tts/configs", config_name="main_config")
def main(clf: DictConfig):
    dataloaders = get_dataloaders(clf["data"])

    model = instantiate(clf["arch"])

    logger = get_logger("train")
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(clf["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = instantiate(clf["loss"]).to(device)
    metrics = [
        #config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
        #for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(clf["optimizer"], trainable_params)
    scheduler = instantiate(clf["scheduler"], optimizer)

    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer=optimizer,
        config=clf,
        device=device,
        log_step=clf["trainer"].get("log_step", 100),
        dataloader=dataloaders,
        scheduler=scheduler,
        len_epoch=clf["trainer"].get("len_epoch", None),
    )

    trainer.train()


if __name__ == "__main__":
    main()