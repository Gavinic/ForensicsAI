# description: trainer
import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import copy
import datetime
import logging
import pickle
import random
import time
from collections import defaultdict
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset.albu import DeNormalize
from metrics.base_metrics_class import Recorder
from metrics.utils import pixel_label_f1
from PIL import Image
from sklearn import metrics
from torch import distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring="auc",
        time_now=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        swa_model=None,
        fold=0,
    ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError(
                "config, model, optimizier, logger, and tensorboard writer must be implemented"
            )

        self.config = config
        self.model = model
        self.fold = fold
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler()
        self.swa_model = swa_model
        self.denormalize = DeNormalize(mean=config["mean"], std=config["std"])
        self.writers = (
            {}
        )  # dict to maintain different tensorboard writers for each dataset and metric
        self.logger = logger
        self.metric_scoring = metric_scoring
        # maintain the best metric of all epochs
        self.best_metrics_all_time = defaultdict(lambda: float("-inf"))
        self.speed_up()  # move model to GPU
        if self.config["ema"]:
            self.ema_inst = EMA(self.model, 0.999)
            self.ema_inst.register()
        # get current time
        self.timenow = time_now
        # create directory path
        if "task_target" not in config:
            self.log_dir = os.path.join(
                self.config["log_dir"], self.config["model_name"] + "_" + self.timenow
            )
        else:
            task_str = (
                f"_{config['task_target']}" if config["task_target"] is not None else ""
            )
            self.log_dir = os.path.join(
                self.config["log_dir"],
                self.config["model_name"] + task_str + "_" + self.timenow,
            )
        os.makedirs(self.log_dir, exist_ok=True)

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            # update directory path
            writer_path = os.path.join(
                self.log_dir, phase, dataset_key, metric_key, "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            # update writers dictionary
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]

    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config["ddp"] == True:
            num_gpus = torch.cuda.device_count()
            print(f"avai gpus: {num_gpus}")
            # local_rank=[i for i in range(0,num_gpus)]
            self.model = DDP(
                self.model,
                device_ids=[self.config["local_rank"]],
                find_unused_parameters=True,
                output_device=self.config["local_rank"],
            )
            # self.optimizer =  nn.DataParallel(self.optimizer, device_ids=[int(os.environ['LOCAL_RANK'])])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def visualize(self, data_dict, prediction):
        ## 1. Prepare the data
        images = data_dict["image"]
        gt_masks = (
            data_dict["gt_mask"].cpu().numpy()
        )  # shape assumed to be (B, 1, H, W) or (B, H, W)
        pred_masks = (
            prediction["pred_mask"].detach().float().sigmoid().cpu().numpy()
        )  # same shape as above
        gt_labels = data_dict["label"].cpu().numpy()
        pred_probs = (
            prediction["pred_label"].detach().float().sigmoid().cpu().numpy()
        )  # assumed to be probabilities here

        # Randomly sample indices
        ids = random.sample(
            range(len(images)), min(len(images), self.config["vis_num"])
        )

        save_dir = os.path.join(self.log_dir, "vis")
        os.makedirs(save_dir, exist_ok=True)

        # Manage file count: if it exceeds the maximum, delete old images
        exist_files = sorted(os.listdir(save_dir))
        if len(exist_files) >= self.config["vis_max"]:
            for per_file in exist_files[: self.config["vis_num"]]:
                try:
                    os.remove(os.path.join(save_dir, per_file))
                except:
                    pass

        for idx in ids:
            # --- 2. Image preprocessing ---
            img_tensor = images[idx].permute(1, 2, 0).cpu()
            image = self.denormalize(
                img_tensor
            )  # assumed to return a 0-255 np.uint8 array
            image = np.ascontiguousarray(image)
            # Ensure BGR format so cv2 can draw in color
            if image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # --- 3. Extract and draw the mask edges ---
            # Process the GT mask (green)
            gt_m = (gt_masks[idx].squeeze() > 0.5).astype(np.uint8)
            contours_gt, _ = cv2.findContours(
                gt_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image, contours_gt, -1, (0, 255, 0), 2)  # green edges

            # Process the predicted mask (red)
            pred_m = (pred_masks[idx].squeeze() > 0.5).astype(np.uint8)
            contours_pred, _ = cv2.findContours(
                pred_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image, contours_pred, -1, (0, 0, 255), 2)  # red edges

            # --- 4. Draw text info ---
            prob_val = pred_probs[idx]
            gt_val = gt_labels[idx]

            # Text background (to make it clearer, a semi-transparent black rectangle can be drawn; here we just draw the text)
            text_gt = f"GT Label: {int(gt_val)}"
            text_pred = f"Pred Prob: {prob_val:.3f}"

            cv2.putText(
                image, text_gt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                image,
                text_pred,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # --- 5. Save ---
            save_name = os.path.basename(data_dict["image_path"][idx])
            # Convert back to RGB for saving
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            Image.fromarray(image_rgb).save(os.path.join(save_dir, save_name))

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location="cpu")
            suffix = model_path.split(".")[-1]
            if suffix == "p":
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info("Model found in {}".format(model_path))
        else:
            raise NotImplementedError("=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"fold{self.fold}_ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config["ddp"] == True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if "svdd" in self.config["model_name"]:
                torch.save(
                    {
                        "R": self.model.R,
                        "c": self.model.c,
                        "state_dict": self.model.state_dict(),
                    },
                    save_path,
                )
            else:
                model_bf16 = copy.deepcopy(self.model).to(dtype=torch.bfloat16)
                torch.save(model_bf16.state_dict(), save_path)
        self.logger.info(
            f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}"
        )

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")

    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"data_dict_{phase}.pickle")
        with open(file_path, "wb") as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, "metric_dict_best.pickle")
        with open(file_path, "wb") as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_step(self, data_dict):
        if self.config["optimizer"]["type"] == "sam":
            for i in range(2):
                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions)
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses["overall"].backward()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:
            with autocast("cuda", dtype=torch.bfloat16):
                predictions = self.model(data_dict)
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)
            # self.optimizer.zero_grad()
            # self.scaler.scale(losses['overall']).backward()
            # self.scaler.unscale_(self.optimizer)
            # self.scaler.step(self.optimizer)
            # self.scaler.update()
            if self.config["ema"]:
                self.ema_inst.update()

            self.optimizer.zero_grad()
            losses["overall"].backward()
            self.optimizer.step()

            return losses, predictions

    def logging_recoder(self, recoders, metric_str="", rep_lr=False):
        for k, v in recoders.items():
            v_avg = v.average()
            if v_avg == None:
                metric_str += f"{k}: not calculated  "
                continue
            metric_str += f"{k}: {v_avg}  "
            # writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
        if rep_lr:
            metric_str += f"lr: {self.optimizer.param_groups[0]['lr']:.6f}"
        self.logger.info(metric_str)

    def train_epoch(
        self,
        epoch,
        train_data_loader,
        test_data_loaders=None,
    ):

        self.logger.info(f"===> Fold {self.fold}  Epoch[{epoch}] start!")

        test_step = len(train_data_loader)  # test 10 times per epoch
        step_cnt = epoch * len(train_data_loader)

        # save the training data_dict
        # self.save_data_dict('train', data_dict, ','.join(self.config['train_dataset']))
        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        for iteration, data_dict in enumerate(train_data_loader):
            self.setTrain()
            # more elegant and more scalable way of moving data to GPU
            for key in data_dict.keys():
                if data_dict[key] != None and key in ["image", "gt_mask", "label"]:
                    data_dict[key] = data_dict[key].cuda()

            losses, predictions = self.train_step(data_dict)
            if self.config["lr_scheduler"]["type"] != "step":
                self.scheduler.step()
                # update learning rate

            if (
                "SWA" in self.config
                and self.config["SWA"]
                and epoch > self.config["swa_start"]
            ):
                self.swa_model.update_parameters(self.model)

            ## store loss
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            # run tensorboard to visualize the training process
            if (
                iteration % self.config["rec_iter"] == 0
                and self.config["local_rank"] == 0
            ):
                ## Randomly sample a few images from a batch for visualization
                if self.config["debug"]:
                    self.visualize(data_dict, predictions)
                # info for loss
                loss_str = f"Iter: {step_cnt}  training-loss:  "
                self.logging_recoder(train_recorder_loss, loss_str, rep_lr=True)

                # info for metric
                # metric_str = f"Iter: {step_cnt}  training-metric: "
                # self.logging_recoder(train_recorder_metric, metric_str, rep_lr=False)

            if self.config["ema"]:
                self.ema_inst.apply_shadow()
            # run test
            test_best_metric = None
            if (step_cnt + 1) % test_step == 0:
                if test_data_loaders is not None and (not self.config["ddp"]):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                elif test_data_loaders is not None and (
                    self.config["ddp"] and dist.get_rank() == 0
                ):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
            if self.config["ema"]:
                self.ema_inst.restore()

                # total_end_time = time.time()
            # total_elapsed_time = total_end_time - total_start_time
            # print("Total time spent: {:.2f} s".format(total_elapsed_time))
            step_cnt += 1
        # clear recorder.
        # Note we only consider the current 300 samples for computing batch-level loss/metric
        for name, recorder in train_recorder_loss.items():  # clear loss recorder
            recorder.clear()
        # for name, recorder in train_recorder_metric.items():  # clear metric recorder
        #     recorder.clear()

        if self.config["lr_scheduler"]["type"] == "step":
            self.scheduler.step()
        ## Save at the end of the epoch
        # if epoch % self.config['save_epoch'] == 0 and self.config['local_rank']==0:
        #     self.save_ckpt(f'train',f"epoch_{epoch}_{fold}_ckpt",ckpt_info=None)
        return test_best_metric

    def get_respect_acc(self, prob, label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = pred == label
        zero_num = len(label) - np.count_nonzero(label)
        acc_fake = np.count_nonzero(judge[zero_num:]) / len(judge[zero_num:])
        acc_real = np.count_nonzero(judge[:zero_num]) / len(judge[:zero_num])
        return acc_real, acc_fake

    def test_one_dataset(self, data_loader):
        # define test recorder
        all_test_metric = defaultdict(Recorder)
        test_recorder_loss = defaultdict(Recorder)
        for i, data_dict in enumerate(data_loader):
            # move data to GPU elegantly
            for key in data_dict.keys():
                if data_dict[key] != None and key != "image_path":
                    data_dict[key] = data_dict[key].cuda()
            # model forward without considering gradient computation
            # with autocast(dtype=torch.bfloat16):
            predictions = self.inference(data_dict)
            ### Get the overall metric
            metrics_dict = pixel_label_f1(pred_dict=predictions, true_dict=data_dict)
            for name, value in metrics_dict.items():
                all_test_metric[name].update(value, num=len(data_dict["image_path"]))

            # feature_lists += list(predictions['feat'].cpu().detach().numpy())
            if type(self.model) is not AveragedModel:
                # compute all losses for each batch data
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)

                # store data by recorder
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        return test_recorder_loss, all_test_metric

    def save_best(self, metric_one_dataset):
        best_metric = self.best_metrics_all_time.get(
            self.metric_scoring, float("-inf")
        )  # the larger r2 the better, but never exceeds 1
        tmp_metric = metric_one_dataset.get(self.metric_scoring).average()
        # Check if the current score is an improvement
        improved = tmp_metric > best_metric  # the larger the better
        if improved:
            # Update the best metric
            self.best_metrics_all_time[self.metric_scoring] = tmp_metric
            # Save checkpoint, feature, and metrics if specified in config
            if self.config["save_ckpt"]:
                self.save_ckpt("test")

    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        # set model to eval mode
        self.setEval()

        # compute loss for each dataset
        losses_one_dataset_recorder, metric_one_dataset = self.test_one_dataset(
            test_data_loaders
        )

        if losses_one_dataset_recorder is not None:
            # info for each dataset
            loss_str = f"Itet: {step}  testing-loss  "
            self.logging_recoder(losses_one_dataset_recorder, loss_str, rep_lr=False)

        metric_str = f"Itet: {step} testing-metric  "
        self.logging_recoder(metric_one_dataset, metric_str, rep_lr=False)

        self.save_best(metric_one_dataset)

        self.logger.info("===> Test Done!")
        return (
            self.best_metrics_all_time
        )  # return all types of mean metrics for determining the best ckpt

    @torch.no_grad()
    def inference(self, data_dict):
        with autocast("cuda", dtype=torch.bfloat16):
            predictions = self.model(data_dict, inference=True)
        return predictions


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
