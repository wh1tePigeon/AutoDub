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
from ..utils import get_logger, prepare_device, CONFIGS_PATH
from ..utils.object_loading import get_dataloaders

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

CONFIG_BSRNN_PATH = CONFIGS_PATH / 'bsrnn'

@hydra.main(config_path=CONFIG_BSRNN_PATH, config_name="main")
def main(config: DictConfig):
    dataloaders = get_dataloaders(config["data"])

    model = instantiate(config["arch"])

    logger = get_logger("train")
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = instantiate(config["loss"]).to(device)
    metrics = [
        instantiate(m) for m in config["metrics"]
        #config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
        #for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config["optimizer"], trainable_params)
    scheduler = instantiate(config["scheduler"], optimizer)

    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer=optimizer,
        config=config,
        device=device,
        log_step=config["trainer"].get("log_step", 100),
        dataloader=dataloaders,
        scheduler=scheduler,
        len_epoch=config["trainer"].get("len_epoch", None),
    )

    trainer.train()


if __name__ == "__main__":
    main()