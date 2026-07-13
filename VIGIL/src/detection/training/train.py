# description: training code.

import argparse
import datetime
import glob
import os
import random
import shutil
import time
from copy import deepcopy
from datetime import timedelta
from os.path import join
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import yaml
from dataset import *
from detectors import DETECTOR
from logger import RankFilter, create_logger
from metrics.utils import parse_metric_for_print
from optimizor.LinearLR import LinearDecayLR
from optimizor.SAM import SAM
from PIL import Image as pil_image
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from trainer.trainer import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Process some paths.")
parser.add_argument(
    "--detector_path",
    type=str,
    default="<BASE_PATH>/DeepfakeBenchv2/training/config/detector/sbi.yaml",
    help="path to detector YAML file",
)
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument(
    "--no-save_ckpt", dest="save_ckpt", action="store_false", default=True
)
parser.add_argument(
    "--no-save_feat", dest="save_feat", action="store_false", default=True
)
parser.add_argument("--ddp", action="store_true", default=False)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument(
    "--task_target",
    type=str,
    default="",
    help="specify the target of current training task",
)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)


def init_seed(config):
    if config["manualSeed"] is None:
        config["manualSeed"] = random.randint(1, 10000)
    random.seed(config["manualSeed"])
    if config["cuda"]:
        torch.manual_seed(config["manualSeed"])
        torch.cuda.manual_seed_all(config["manualSeed"])


def prepare_data(config, train_set, test_set):
    # Only use the blending dataset class in training

    if config["ddp"]:
        sampler = DistributedSampler(train_set)
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config["train_batchSize"],
            num_workers=int(config["workers"]),
            sampler=sampler,
        )
        test_data_loader
    else:
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config["train_batchSize"],
            shuffle=True,
            num_workers=int(config["workers"]),
            pin_memory=True,  # newly added
            prefetch_factor=2,
        )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config["test_batchSize"],
        shuffle=False,
        num_workers=int(config["workers"]),
        drop_last=False,
    )
    return train_data_loader, test_data_loader


def choose_optimizer(model, config):
    # if config['model_name'] == 'biomconv':
    #     param_groups = [
    #     {'params': model.backbone.parameters(), 'lr': 5e-5, 'weight_decay': 5e-5,
    #         'name': 'backbone'},
    #     {'params': model.head_total.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5,
    #         'name': 'head'},
    #     {'params': model.head_gdm.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5, 'name': 'head'},
    #     {'params': model.head_green.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5, 'name': 'head'},
    #     {'params': model.head_height.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5, 'name': 'head'},
    #     {'params': model.head_hdvi.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5 ,'name': 'head'},

    #     ]
    # else:
    param_groups = model.parameters()
    opt_name = config["optimizer"]["type"]
    if opt_name == "sgd":
        optimizer = optim.SGD(
            params=param_groups,
            # params=model.parameters(),
            lr=config["optimizer"][opt_name]["lr"],
            momentum=config["optimizer"][opt_name]["momentum"],
            weight_decay=config["optimizer"][opt_name]["weight_decay"],
        )
        return optimizer
    elif opt_name == "adam":
        optimizer = optim.Adam(
            params=param_groups,
            # params=model.parameters(),
            lr=config["optimizer"][opt_name]["lr"],
            weight_decay=config["optimizer"][opt_name]["weight_decay"],
            betas=(
                config["optimizer"][opt_name]["beta1"],
                config["optimizer"][opt_name]["beta2"],
            ),
            eps=config["optimizer"][opt_name]["eps"],
            amsgrad=config["optimizer"][opt_name]["amsgrad"],
        )
        return optimizer
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            params=param_groups,
            # params=model.parameters(),
            lr=config["optimizer"][opt_name]["lr"],
            weight_decay=config["optimizer"][opt_name]["weight_decay"],
            betas=(
                config["optimizer"][opt_name]["beta1"],
                config["optimizer"][opt_name]["beta2"],
            ),
            eps=1e-8,
        )
    elif opt_name == "sam":
        optimizer = SAM(
            # model.parameters(),
            optim.SGD,
            lr=config["optimizer"][opt_name]["lr"],
            momentum=config["optimizer"][opt_name]["momentum"],
        )
    else:
        raise NotImplementedError(
            "Optimizer {} is not implemented".format(config["optimizer"])
        )
    return optimizer


def choose_scheduler(config, optimizer, dataset=None, total_nums=0):
    scheduler_name = config["lr_scheduler"]["type"]
    scheduler_opt = config["lr_scheduler"][scheduler_name]
    if scheduler_name is None:
        return None
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_opt["lr_step"],
            gamma=scheduler_opt["lr_gamma"],
        )
        return scheduler
    elif scheduler_name == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_opt["lr_max"],
            epochs=config["nEpochs"] + 1,
            steps_per_epoch=len(dataset)
            // config["train_batchSize"],  # must be provided
            pct_start=scheduler_opt["lr_pct"],  # warmup accounts for 10%
            # anneal_strategy='cos'
        )
        return scheduler
    elif scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_opt["lr_T_max"],
            eta_min=scheduler_opt["lr_eta_min"],
        )
        return scheduler
    elif scheduler_name == "cosine_warmup":
        # Recommended for ViT: Cosine + Warmup (per iteration)
        warmup_steps = 150
        total_steps = total_nums
        # Warmup scheduler
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=scheduler_opt.get("start_factor", 0.01),
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # Cosine scheduler
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=scheduler_opt.get("eta_min", 1e-6),
        )

        # Combine
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        return scheduler
    elif scheduler_name == "linear":
        scheduler = LinearDecayLR(
            optimizer,
            config["nEpochs"],
            int(config["nEpochs"] / 4),
        )
        return scheduler
    else:
        raise NotImplementedError(
            "Scheduler {} is not implemented".format(config["lr_scheduler"])
        )


def choose_metric(config):
    metric_scoring = config["metric_scoring"]
    # if metric_scoring not in ['avg_mae','mse','r2']:
    #     raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
    else:
        lam = 1

    device = x.get_device()
    batch_size = x.size()[0]

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index


def main():
    # parse options and load config
    with open(args.detector_path, "r") as f:
        config = yaml.safe_load(f)

    config["local_rank"] = args.local_rank

    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config["train_dataset"] = args.train_dataset
    if args.test_dataset:
        config["test_dataset"] = args.test_dataset
    config["save_ckpt"] = args.save_ckpt
    config["save_feat"] = args.save_feat

    if args.task_target:
        config["task_target"] = args.task_target
    # create logger
    timenow = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    task_str = (
        f"_{config['task_target']}"
        if config.get("task_target", None) is not None
        else ""
    )
    logger_path = os.path.join(
        config["log_dir"], config["model_name"] + task_str + "_" + timenow
    )
    os.makedirs(logger_path, exist_ok=True)
    shutil.copy(
        args.detector_path,
        os.path.join(logger_path, os.path.basename(args.detector_path)),
    )
    logger = create_logger(os.path.join(logger_path, "training.log"))
    logger.info("Save log to {}".format(logger_path))
    config["ddp"] = args.ddp
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config["cudnn"]:
        cudnn.benchmark = True
    if config["ddp"]:
        # dist.init_process_group(backend='gloo')
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        logger.addFilter(RankFilter(0))
    # pertrain_model(config, logger)
    # extend_train(config, logger)
    ## Hard samples in this part need to be strengthened, so repeat once
    hard_samples = [
        "8840992135d54909a810e204a2f61ae4",
        "04b038df33cd46319c5dddc0c00bf45b",
        "feec59b9fc6e42c8a855bc76040a30cd",
        "cecb2dff8c484d49a3652a8e36286127",
        "e0a8e3d8ef19470682e67c50bce6a7a0",
        "72cfa8a9af8047cebd00de9bef2a98a6",
        "42c1f974645d4147816013993c713763",
        "e7e05f0c70dc4396a444c689cd140438",
        ## Strengthen the AIGC part as well
        "9a136c257b0146089c1658c46936b073",
        "d0379c433b8e4847a7f6d2f5f76be7e2",
        "ff422ce8d21648018a3c9818ef704b08",
        "70396919c4e8424e81ecd53dc74dca46",
        "35b4601702514908b4ef00478de5f95e",
        "43a16109c215462a9932f55ca9db737c",
        "73c86c3203714ce29060f8a9fd563b9d",
    ]
    ## Add
    image_paths = glob.glob("<BASE_PATH>/data/2026forgery/data/train/Black/Image/*")
    data_list = []
    for image_path in image_paths:
        data = {}
        data["image_path"] = Path(image_path)
        data["mask_path"] = Path(
            str(image_path).replace("Image", "Mask").replace(".jpg", ".png")
        )
        data["label"] = 1
        data["forgery"] = 1
        if data["image_path"].name[:-4] in hard_samples:
            for _ in range(9):
                data_list.append(data.copy())
        data_list.append(data)
    image_paths = glob.glob("<BASE_PATH>/data/2026forgery/data/train/White/Image/*")
    for image_path in image_paths:
        data = {}
        data["image_path"] = Path(image_path)
        data["mask_path"] = ""
        data["label"] = 0
        data["forgery"] = (
            1  # marks competition data; competition data for receipts may need cropping
        )
        if data["image_path"].name[:-4] in hard_samples:
            for _ in range(9):
                data_list.append(data.copy())
        data_list.append(data)

    data_list = np.array(data_list, dtype=object)
    ### extend:
    ## Extended data
    extend_list_pos = []
    # pl_data = pl.read_csv('<BASE_PATH>/data/2026forgery/data/train.csv')
    pl_data = pl.read_csv("<BASE_PATH>/data/2026forgery/data/vertices_lesseq4_pos.csv")
    for row in pl_data.iter_rows(named=True):
        data = {}
        if row["Region"] and int(row["Label"]) == 1:
            data["mask_rel"] = row["Region"]
        else:
            continue
        data["image_path"] = Path("<BASE_PATH>/data/2026forgery/data/extend_train") / Path(
            row["Path"]
        )
        data["extend"] = 1
        data["label"] = row["Label"]
        extend_list_pos.append(data)
    extend_list_pos = np.array(extend_list_pos, dtype=object)

    pl_data = pl.read_csv("<BASE_PATH>/data/2026forgery/data/train.csv")
    extend_list_neg = []
    for row in pl_data.iter_rows(named=True):
        data = {}
        if int(row["Label"]) == 0:
            data["mask_rel"] = ""
        else:
            continue
        data["image_path"] = Path("<BASE_PATH>/data/2026forgery/data/extend_train") / Path(
            row["Path"]
        )
        data["extend"] = 1
        data["label"] = row["Label"]
        extend_list_neg.append(data)
    extend_list_neg = np.array(extend_list_neg, dtype=object)
    print(f"ori extend data pos: {len(extend_list_pos)} neg: {len(extend_list_neg)}")

    ### cutteed_dataset_fakes
    # extend_list1 = []
    # tsorie_images = glob.glob("<BASE_PATH>/data/2026forgery/data/cutted_datasets_alls/T-SROIE_train/images/*") + \
    #     glob.glob("<BASE_PATH>/data/2026forgery/data/cutted_datasets_alls/T-SROIE_test/images/*")
    # for img_p in tsorie_images:
    #     data = {
    #         'image_path': Path(img_p),
    #         'mask_path': img_p.replace("/images/", "/masks/").replace(".jpg", ".png"),
    #         'label': 1,  # must load once here to know whether it is 1
    #         'extend': 0
    #     }
    #     if cv2.imread(data['mask_path'],cv2.IMREAD_GRAYSCALE).sum()==0:
    #         # print("T-SROIE detect negative samples")
    #         data['label'] = 0
    #     extend_list1.append(data)
    # tamperd_images = glob.glob("<BASE_PATH>/data/2026forgery/data/cutted_datasets_alls/Tampered-IC13_train/images/*") + \
    #     glob.glob("<BASE_PATH>/data/2026forgery/data/cutted_datasets_alls/Tampered-IC13_test/images/*")
    # for img_p in tamperd_images:
    #     data = {
    #         'image_path': Path(img_p),
    #         'mask_path': img_p.replace("/images/", "/masks/").replace(".jpg", ".png"),
    #         'label': 1,  # must load once here to know whether it is 1
    #         'extend': 0
    #     }
    #     if cv2.imread(data['mask_path'],cv2.IMREAD_GRAYSCALE).sum()==0:
    #         # print("Tampered- detect negative samples")
    #         data['label'] = 0
    #     extend_list1.append(data)
    # realtext_images = glob.glob("<BASE_PATH>/data/2026forgery/data/cutted_datasets_alls/RealTextManipulation_train/images/*") + \
    #     glob.glob("<BASE_PATH>/data/2026forgery/data/cutted_datasets_alls/RealTextManipulation_test/images/*")
    # for img_p in realtext_images:
    #     data = {
    #         'image_path': Path(img_p),
    #         'mask_path': img_p.replace("/images/", "/masks/").replace(".jpg", ".png"),
    #         'label': 1,  # must load once here to know whether it is 1
    #         'extend': 0
    #     }
    #     if cv2.imread(data['mask_path'],cv2.IMREAD_GRAYSCALE).sum()==0:
    #         # print("RealTextManipulation detect negative samples")
    #         data['label'] = 0
    #     extend_list1.append(data)
    # ostf_images = glob.glob("<BASE_PATH>/data/2026forgery/data/cutted_datasets_alls/OSTF_train/images/*") + \
    #     glob.glob("<BASE_PATH>/data/2026forgery/data/cutted_datasets_alls/OSTF_test/images/*")
    # for img_p in ostf_images:
    #     data = {
    #         'image_path': Path(img_p),
    #         'mask_path': img_p.replace("/images/", "/masks/").replace(".jpg", ".png"),
    #         'label': 1,  # must load once here to know whether it is 1
    #         'extend': 0
    #     }
    #     if cv2.imread(data['mask_path'],cv2.IMREAD_GRAYSCALE).sum()==0:
    #         # print("OSTF detect negative samples")
    #         data['label'] = 0
    #     extend_list1.append(data)
    # extend_list1 = np.array(extend_list1,dtype=object)
    ## Save it here so it can be loaded directly next time without re-detection
    # np.save("<BASE_PATH>/data/2026forgery/data/cutted_datasets_alls_array.npy",extend_list1)
    extend_list1 = np.load(
        "<BASE_PATH>/data/2026forgery/data/cutted_datasets_alls_array.npy",
        allow_pickle=True,
    )
    extend_list1 = extend_list1
    ###IMD_20_1024 contains AIGC samples
    extend_list2 = []
    imd_images = glob.glob("<BASE_PATH>/data/2026forgery/data/IMD_20_1024/Tp/*")[::10]
    for img_p in imd_images:
        extend_list2.append(
            {
                "image_path": Path(img_p),
                "mask_path": img_p.replace("Tp", "Gt")[:-4] + "_mask.png",
                "label": 1,
                "extend": 0,
            }
        )
    extend_list2 = np.array(extend_list2, dtype=object)
    logger.info(f"exfend_list2:{len(extend_list2)}")

    kaggle_df = pl.read_csv("<BASE_PATH>/data/2026forgery/data/sroie/test_1011.csv")
    kaggle_list = []
    for row in kaggle_df.iter_rows(named=True):
        data = {}
        if int(row["label"]) == 1:
            data["mask_rel"] = row["rel"]
        else:
            data["mask_rel"] = ""
        data["image_path"] = Path("<BASE_PATH>/data/2026forgery/data/sroie/test") / Path(
            row["image_name"]
        )
        data["extend"] = 1
        data["forgery"] = 1
        data["label"] = row["label"]
        kaggle_list.append(data)
    kaggle_list = np.array(kaggle_list, dtype=object)

    if not config["5_fold"]:
        # for fold in range(5):
        nkfold = StratifiedKFold(5)
        sid = 0
        fold = f"onetrain_{sid}"
        for tr_idx, val_idx in nkfold.split(data_list, [d["label"] for d in data_list]):
            val_groups = data_list[val_idx]
            tr_groups = data_list[tr_idx]
            break
        # tr_groups = tr_groups.repeat(2)
        logger.info(f"training set size: {len(tr_groups)}")
        logger.info(f"validation set size: {len(val_groups)}")
        logger.info(
            f"positive-sample ratio in the validation set: {np.mean([d['label'] for d in val_groups])}"
        )

        train_set = IdbmDataset(
            data_list=tr_groups.repeat(2), config=config, mode="train"
        )
        test_set = IdbmDataset(data_list=val_groups, config=config, mode="test")
        # prepare the training data loader
        train_data_loader, test_data_loaders = prepare_data(config, train_set, test_set)
        ## Save the val list for each fold
        torch.save(
            {"tr_data": tr_groups, "val_data": val_groups},
            os.path.join(logger_path, f"onetrain_data.pt"),
        )
        # prepare the model (detector)
        model_class = DETECTOR[config["model_name"]]
        model = model_class(config)
        # ## Force-load here
        # model.load_state_dict(torch.load(config['pretrained']))
        # prepare the optimizer
        optimizer = choose_optimizer(model, config)

        # prepare the scheduler
        scheduler = choose_scheduler(
            config,
            optimizer,
            train_set,
            total_nums=len(train_set) // config["train_batchSize"] * 5
            + (len(extend_list1) + len(extend_list2)) // config["train_batchSize"] * 7
            + (len(train_set) * 10) // config["train_batchSize"] * 8,
        )

        # prepare the metric
        metric_scoring = choose_metric(config)

        # prepare the trainer
        trainer = Trainer(
            config,
            model,
            optimizer,
            scheduler,
            logger,
            metric_scoring,
            time_now=timenow,
            fold=fold,
        )
        # first test
        if (
            test_data_loaders is not None
            and (not config["ddp"])
            and config["test_first"]
        ):
            trainer.test_epoch(0, 0, test_data_loaders, 0)
        # start training
        for epoch in range(config["start_epoch"], config["nEpochs"] + 1):
            trainer.model.epoch = epoch
            if 11 > epoch > 3:
                ## Randomly sample new data from it each epoch
                # Randomly sample indices
                idx_pos = np.random.choice(
                    len(extend_list_pos), size=len(train_set) * 5, replace=False
                )  # random repeated sampling, with replacement
                idx_neg = np.random.choice(
                    len(extend_list_neg), size=len(train_set) * 3, replace=False
                )

                extend_group1 = extend_list_pos[idx_pos]
                extend_group2 = extend_list_neg[idx_neg]
                logger.info(
                    f"extend set size: pos {len(extend_group1)} neg  {len(extend_group2)}"
                )
                extend_group = np.concatenate(
                    [
                        tr_groups.repeat(10),
                        kaggle_list,
                        extend_group1,
                        extend_group2,
                        extend_list1,
                        extend_list2,
                    ],
                    axis=0,
                )
                logger.info(
                    f"extend set size: {len(extend_group)} , extend_list1: {len(extend_list1)}, extend_list2: {len(extend_list2)}"
                )
                train_set_extend = IdbmDataset(
                    data_list=extend_group, config=config, mode="train"
                )
                train_data_loader_extend = torch.utils.data.DataLoader(
                    dataset=train_set_extend,
                    batch_size=config["train_batchSize"],
                    shuffle=True,
                    num_workers=int(config["workers"]),
                    pin_memory=True,  # newly added
                    prefetch_factor=2,
                )

                best_metric = trainer.train_epoch(
                    epoch=epoch,
                    train_data_loader=train_data_loader_extend,
                    test_data_loaders=test_data_loaders,
                )
            elif epoch >= 11:
                train_set_extend = IdbmDataset(
                    data_list=np.concatenate(
                        [data_list.repeat(4), kaggle_list], axis=0
                    ),
                    config=config,
                    mode="train",
                )
                train_data_loader_extend = torch.utils.data.DataLoader(
                    dataset=train_set_extend,
                    batch_size=config["train_batchSize"],
                    shuffle=True,
                    num_workers=int(config["workers"]),
                    pin_memory=True,  # newly added
                    prefetch_factor=2,
                )

                best_metric = trainer.train_epoch(
                    epoch=epoch,
                    train_data_loader=train_data_loader_extend,
                    test_data_loaders=test_data_loaders,
                )
            else:
                best_metric = trainer.train_epoch(
                    epoch=epoch,
                    train_data_loader=train_data_loader,
                    test_data_loaders=test_data_loaders,
                )
            if best_metric is not None:
                logger.info(
                    f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!"
                )
        # update
    else:
        kf = KFold(n_splits=5, random_state=24, shuffle=True)
        cv_best_val = []
        for fold, (tr_ind, val_ind) in enumerate(kf.split(data_list)):
            tr_groups = data_list[tr_ind]
            val_groups = data_list[val_ind]
            tr_groups = tr_groups.repeat(4)
            logger.info(f"training set size: {len(tr_groups)}")
            logger.info(f"validation set size: {len(val_groups)}")
            train_set = BiomDataset(data_list=tr_groups, config=config, mode="train")
            test_set = BiomDataset(data_list=val_groups, config=config, mode="test")
            # prepare the training data loader
            train_data_loader, test_data_loaders = prepare_data(
                config, train_set, test_set
            )
            ## Save the val list for each fold
            torch.save(
                {"tr_data": tr_groups, "val_data": val_groups},
                os.path.join(logger_path, f"tr_val_fold{fold}_data.pt"),
            )
            # prepare the model (detector)
            model_class = DETECTOR[config["model_name"]]
            model = model_class(config)
            # ## Force-load here
            # model.load_state_dict(torch.load(config['pretrained']))
            # prepare the optimizer
            optimizer = choose_optimizer(model, config)

            # prepare the scheduler
            scheduler = choose_scheduler(config, optimizer, train_set)

            # prepare the metric
            metric_scoring = choose_metric(config)

            # prepare the trainer
            trainer = Trainer(
                config,
                model,
                optimizer,
                scheduler,
                logger,
                metric_scoring,
                time_now=timenow,
                fold=fold,
            )
            # first test
            if (
                test_data_loaders is not None
                and (not config["ddp"])
                and config["test_first"]
            ):
                trainer.test_epoch(0, 0, test_data_loaders, 0)
            # start training
            for epoch in range(config["start_epoch"], config["nEpochs"] + 1):
                trainer.model.epoch = epoch
                best_metric = trainer.train_epoch(
                    epoch=epoch,
                    train_data_loader=train_data_loader,
                    test_data_loaders=test_data_loaders,
                )
                if best_metric is not None:
                    logger.info(
                        f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!"
                    )

            logger.info(
                "Stop Training on best Testing metric {}".format(
                    parse_metric_for_print(best_metric)
                )
            )
            # update
            cv_best_val.append(best_metric)

            # close the tensorboard writers
            for writer in trainer.writers.values():
                writer.close()
        mean_metric = np.mean([x[metric_scoring] for x in cv_best_val])
        for i, val in enumerate(cv_best_val):
            logger.info(f"Fold {i} : {metric_scoring}:{val[metric_scoring]}")
        logger.info(f"Mean {metric_scoring}:{mean_metric}")


if __name__ == "__main__":
    main()
