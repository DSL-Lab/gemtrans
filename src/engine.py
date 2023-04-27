import numpy as np
from src.builders import (
    data_builder,
    model_builder,
    criterion_builder,
    scheduler_builder,
    optimizer_builder,
    transform_builder,
    evaluator_builder,
    meter_builder,
)
from src.utils.misc import sanity_checks, set_random_seed
from src.utils.vis import transformer_vis
import torch.nn as nn
import torch
import os
import time
from tqdm import tqdm
import pandas as pd
from src.utils.misc import set_random_seed
from attributedict.collections import AttributeDict
import json
import random


try:
    import wandb
except ImportError:
    print("Install wandb if cloud model monitoring is required.")


class Engine:
    def __init__(
        self, config, save_dir, logger, train=True, sweep=False, pretrain=True
    ):

        self.logger = logger

        # Where to save trained model and other artefacts
        self.save_dir = save_dir

        if not sweep:
            # Log the config
            self.logger.info(json.dumps(config))

        config = AttributeDict(config)

        # Set the seed
        set_random_seed(config.train.seed)

        # Config sanity checks
        sanity_checks(config=config, train=train)

        # Init wandb config beforehand if sweeping
        if sweep:
            self._init_wandb(config=config)
            config = self._modify_config_for_sweep(config)

        # Process config
        self._process_configs(config=config, train=train, sweep=sweep)

        # Initialize wandb
        if self.config.train.use_wandb and not sweep:
            self._init_wandb(config=config)

        # Keep the best metric value
        if self.config.train.evaluator.maximize:
            self.best_eval_metric = -100000
        else:
            self.best_eval_metric = 100000

        # Other required variables
        self.train_num_steps = 0
        self.training_epoch_steps = None
        self.val_num_steps = None
        self.val_epoch_steps = None
        self.pretrain = pretrain

    @staticmethod
    def _modify_config_for_sweep(config):
        sweep_config = wandb.config

        # Train configs
        for (key, val) in sweep_config["train"]["optimizer"].items():
            config.train.optimizer[key] = val

        for (key, val) in sweep_config["train"]["criterion"].items():
            config.train.criterion[key] = val

        # Model configs
        for (key, val) in sweep_config["model"].items():
            config.model[key] = val

        # Data configs
        # for (key, val) in sweep_config["data"].items():
        #     config.data[key] = val
        frames_config = sweep_config["data"]["max_frames_sampled_frames"]
        config.data["max_frames"] = frames_config[0]
        config.data["n_sampled_frames"] = frames_config[1]

        return config

    @staticmethod
    def _init_wandb(config):
        wandb.init(
            project="heart_transformer",
            name=config.train.wandb_run_name,
            config=config,
            entity=config.train.wandb_entity,
            group=config.train.wandb_group_name,
            mode=config.train.wandb_mode,
        )

        # define our custom x axis metric
        wandb.define_metric("batch_train/step")
        wandb.define_metric("batch_valid/step")
        wandb.define_metric("epoch")

        # set all other metrics to use the corresponding step
        wandb.define_metric("batch_train/*", step_metric="batch_train/step")
        wandb.define_metric("batch_valid/*", step_metric="batch_valid/step")
        wandb.define_metric("epoch/*", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        wandb.define_metric("training_throughput", step_metric="epoch")
        wandb.define_metric("time_per_step", step_metric="epoch")

    def _process_configs(self, config, train, sweep):

        # Useful flags used by the engine
        self.train = train
        self.sweep = sweep
        self.config = config

        # Make sure wandb_log_steps is divisble by batch size
        self.config.train.batch_size = (
            self.config.train.batch_size if self.config.train else 1
        )
        self.config.train.wandb_log_steps = self.config.train.wandb_log_steps + (
            self.config.train.batch_size
            - self.config.train.wandb_log_steps % self.config.train.batch_size
        )

        # Update model's config based on what information the model needs
        self.config.model.update(
            {
                "vid_seq_len": 2
                if self.config.data.name in ["camus", "biplane", "as"]
                else 1,
                "mode": self.config.train.mode,
                "use_seg_labels": self.config.train.criterion.attn_lambda > 0,
                "use_ed_es_locs": self.config.train.criterion.frame_lambda > 0,
                "return_full_attn": not self.train,
                "use_classification_head": self.config.train.criterion.classification_lambda
                > 0,
                "n_sampled_frames": self.config.data.n_sampled_frames,
            }
        )

        # Process criterion configs
        self.config.train.criterion.update(
            {
                "frame_size": self.config.data.frame_size,  # Needed for Seg Loss
                "patches": self.config.model.patches,  # Needed for Seg Loss
                "n_sampled_frames": self.config.data.n_sampled_frames,  # Needed for Seg Loss
            }
        )

        # Process data configs
        self.config.data.update(
            {
                "batch_size": self.config.train.batch_size,
                "n_sampled_frames": self.config.data.n_sampled_frames,
                "patches": self.config.model.patches,
                "mode": self.config.train.mode,
                "vid_seq_len": 2 if self.config.data.name in ["biplane", "as"] else 1,
                "spatial_aggr_method": self.config.model.spatial_aggr_method,
                "temporal_aggr_method": self.config.model.temporal_aggr_method,
                "vid_aggr_method": self.config.model.vid_aggr_method,
                "use_seg_labels": self.config.train.criterion.attn_lambda > 0,
            }
        )

        # Process optimizer and scheduler configs
        self.config.train.optimizer.update(
            {"mode": self.config.train.mode, "use_ppnet": self.config.model.use_ppnet}
        )

        self.config.train.scheduler.update(
            {
                "mode": self.config.train.mode,
                "epochs": self.config.train.epochs,
                "batch_size": self.config.train.batch_size,
            }
        )

    def _build(self):

        # Build the required torchvision transforms
        self.transform, self.aug_transform = transform_builder.build(
            self.config.data, self.train
        )

        # Build the dataloaders
        self.dataloaders, self.data_dirs = data_builder.build(
            self.config.data,
            self.train,
            self.transform,
            self.aug_transform,
            logger=self.logger,
        )

        # Build the model
        self.checkpoint_path = self.config.model.pop("checkpoint_path")
        self.model = model_builder.build(self.config.model)

        # Build the criteria
        self.criterion = criterion_builder.build(
            self.config.train.criterion, self.config.train.mode
        )
        self.loss_meters = meter_builder.build()

        # Build the evaluator
        self.evaluator = evaluator_builder.build(self.config.train.evaluator)

        # Build the optimizer and the scheduler
        if self.train:
            self.optimizer = optimizer_builder.build(
                self.model, self.config.train.optimizer
            )
            self.scheduler = scheduler_builder.build(
                self.optimizer, self.config.train.scheduler
            )

        # Load the checkpoint
        self.misc = self.load_checkpoint()

        # Move model to correct device
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.model = nn.DataParallel(self.model)

            if self.train:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

            self.model.to(self.device)

    def train_model(self):

        self._build()

        # Number of iterations per epoch
        self.training_epoch_steps = len(self.dataloaders["train"])
        self.val_epoch_steps = len(self.dataloaders["val"])

        # Starting epoch
        start_epoch = 0 if not self.misc else self.misc["epoch"]

        for epoch in range(start_epoch, start_epoch + self.config.train.epochs):

            # Reset the evaluators
            self.reset_evaluator()
            self.reset_meters()

            # Start timer
            start_time = time.time()

            # Start training for one epoch
            self._train_one_epoch(epoch)

            total_time = time.time() - start_time

            # Compute training throughput and time per step
            self.log_computation_stats(epoch=epoch, total_time=total_time)

            if self.config.train.mode == "pretrain":
                self.save_checkpoint(
                    epoch=epoch,
                    eval_metric=self.evaluator[self.config.train.evaluator.eval_metric],
                )
            else:
                self.val_num_steps = self.train_num_steps
                self.reset_evaluator()
                self.reset_meters()

                # Perform validation
                self._evaluate_one_epoch(epoch, True)

    def log_computation_stats(self, epoch, total_time):
        training_throughput = (
            self.config.train.batch_size * len(self.dataloaders["train"]) / total_time
        )
        time_per_step = self.config.train.batch_size / training_throughput

        self.logger.info(
            "Epoch finished with a training throughput of {} and time per step of {}".format(
                training_throughput, time_per_step
            )
        )

        if self.config.train.use_wandb:
            wandb.log(
                {
                    "training_throughput": training_throughput,
                    "time_per_step": time_per_step,
                    "epoch": epoch,
                }
            )

    def evaluate(self):
        self._build()

        self.val_epoch_steps = len(self.dataloaders["val"])
        self.val_num_steps = self.train_num_steps

        self._evaluate_one_epoch(0, False)

    def _prepare_input_data(self, data_iter):
        data_dict = next(data_iter)

        for key in data_dict.keys():
            if data_dict[key] is not None:
                data_dict[key] = data_dict[key].to(self.device)

        return data_dict

    def _train_one_epoch(self, epoch):

        # Change to train mode
        self.model.train()

        # Initialize TQDM iterator
        iterator = tqdm(range(len(self.dataloaders["train"])), dynamic_ncols=True)
        data_iter = iter(self.dataloaders["train"])

        for i in iterator:

            data_dict = self._prepare_input_data(data_iter)

            self.optimizer.zero_grad()

            (
                loss,
                output,
                patch_pos,
                frame_pos,
                vid_pos,
                patch_attn,
                frame_attn,
                vid_attn,
                sampled_vid,
                logits_proto,
            ) = self._forward_path(data_dict)

            # Backward prop
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                # Update evaluators
                self.update_evaluator(
                    output.detach().cpu().numpy(),
                    data_dict["label"].detach().cpu().numpy(),
                    logits_proto
                    if logits_proto is None
                    else logits_proto.detach().cpu().numpy(),
                )

                # Update TQDM bar
                self._set_tqdm_description(
                    iterator=iterator,
                    log_mode="Training",
                    epoch=epoch,
                    loss=self.loss_meters["main_loss"].avg
                    + self.loss_meters["prototype_loss"].avg,
                )

                # Visualizations
                if not self.sweep and self.config.train.visualize:
                    transformer_vis(
                        epoch=epoch,
                        iteration=i,
                        use_wandb=self.config.train.use_wandb,
                        save_dir=self.save_dir,
                        vids=data_dict["vid"],
                        patch_attn=patch_attn,
                        frame_attn=frame_attn,
                        vid_attn=vid_attn,
                        patch_pos=patch_pos,
                        frame_pos=frame_pos,
                        vid_pos=vid_pos,
                        wandb_mode="batch_train",
                        config=self.config.data,
                        num_steps=self.train_num_steps,
                        epoch_steps=self.training_epoch_steps,
                        batch_size=self.config.train.batch_size,
                        wandb_log_steps=self.config.train.wandb_log_steps,
                    )

                self.train_num_steps += self.config.train.batch_size

        self._train_epoch_summary(epoch=epoch, lr=self.optimizer.param_groups[0]["lr"])

    def _evaluate_one_epoch(self, epoch, train=True):

        with torch.no_grad():

            logits = list()

            # Change to eval mode
            self.model.eval()

            # Initialize TQDM iterator
            iterator = tqdm(range(len(self.dataloaders["val"])), dynamic_ncols=True)
            data_iter = iter(self.dataloaders["val"])

            for i in iterator:
                data_dict = self._prepare_input_data(data_iter)

                # TODO: find more elegant solution
                if self.config.train.mode != "pretrain":
                    data_dict["vid"] = data_dict["vid"].squeeze(0)
                    data_dict["mask"] = data_dict["mask"].squeeze(0)

                output_dict = self.model(data_dict)

                # During test time, multiple clips are extracted and averaged
                output = torch.mean(output_dict["x"], dim=0)

                if not train:
                    logits.append(output.detach().cpu().numpy())

                output = (
                    output.unsqueeze(0)
                    if self.config.train.mode in ["as", "pretrain"]
                    else output
                )

                # Compute loss
                loss = self.criterion[
                    "regression" if self.config.train.mode == "ef" else "classification"
                ](output, data_dict["label"])

                with torch.no_grad():
                    self.loss_meters["main_loss"].update(loss.detach().cpu().item())

                prototype_loss = 0
                if self.config.model.use_ppnet:
                    prototype_loss = self.criterion["classification"](
                        logits_proto, labels
                    )

                with torch.no_grad():
                    self.loss_meters["prototype_loss"].update(
                        prototype_loss.detach().cpu().item()
                        if prototype_loss > 0
                        else prototype_loss
                    )

                self._set_tqdm_description(
                    iterator=iterator,
                    log_mode="Validation" if self.train else "Test",
                    epoch=epoch,
                    loss=self.loss_meters["main_loss"].avg
                    + self.loss_meters["prototype_loss"].avg,
                )

                self.update_evaluator(
                    output.detach().cpu().numpy(),
                    data_dict["label"].detach().cpu().numpy(),
                    proto_logits=output_dict["logits"]
                    if output_dict["logits"] is None
                    else output_dict["logits"].detach().cpu().numpy(),
                )

                if not self.sweep and self.config.train.visualize:
                    transformer_vis(
                        epoch=epoch,
                        iteration=i,
                        use_wandb=self.config.train.use_wandb,
                        save_dir=self.save_dir,
                        vids=output_dict["sampled_vid"].squeeze()
                        if not self.train
                        else data_dict["vid"],
                        patch_attn=output_dict["patch_attn"],
                        frame_attn=output_dict["frame_attn"],
                        vid_attn=output_dict["vid_attn"],
                        patch_pos=output_dict["patch_pos_embed"],
                        frame_pos=output_dict["frame_pos_embed"],
                        vid_pos=output_dict["vid_pos_embed"],
                        wandb_mode="batch_valid",
                        config=self.config.data,
                        num_steps=self.val_num_steps,
                        epoch_steps=self.val_epoch_steps,
                        batch_size=1,
                        wandb_log_steps=self.config.train.wandb_log_steps,
                        ed_frame=output_dict["ed_frames"].detach().cpu().numpy()[0]
                        // (
                            self.config.data.max_frames
                            // self.config.data.n_sampled_frames
                        )
                        if not train
                        else None,
                        ed_valid=output_dict["ed_valid"].detach().cpu().numpy()[0]
                        if not train
                        else None,
                        es_frame=output_dict["es_frames"].detach().cpu().numpy()[0]
                        // (
                            self.config.data.max_frames
                            // self.config.data.n_sampled_frames
                        )
                        if not train
                        else None,
                        es_valid=output_dict["es_valid"].detach().cpu().numpy()[0]
                        if not train
                        else None,
                        test=not self.train,
                        num_frames=self.config.data.n_sampled_frames,
                        sample_idx=i,
                        label=data_dict["label"][0].item(),
                    )

                self.val_num_steps += 1

            eval_metric = self.evaluator[
                self.config.train.evaluator.eval_metric
            ].compute()

            if train:
                self.scheduler.step(eval_metric)

            if self.train:
                if self.config.train.evaluator.maximize:
                    self.best_eval_metric = (
                        eval_metric
                        if eval_metric > self.best_eval_metric
                        else self.best_eval_metric
                    )
                else:
                    self.best_eval_metric = (
                        eval_metric
                        if eval_metric < self.best_eval_metric
                        else self.best_eval_metric
                    )

                if not self.sweep:
                    self.save_checkpoint(epoch=epoch, eval_metric=eval_metric)

                self.log_wandb_summary()
            else:
                # Save predictions csv
                prediction_df = pd.DataFrame(
                    {
                        "preds": self.evaluator[
                            self.config.train.evaluator.eval_metric
                        ].y_pred,
                        "labels": self.evaluator[
                            self.config.train.evaluator.eval_metric
                        ].y_true,
                        "logits": logits,
                    }
                )
                prediction_df.to_csv(os.path.join(self.save_dir, "preds.csv"))

            self._val_epoch_summary(epoch=epoch)

    def _train_epoch_summary(
        self,
        epoch,
        lr,
    ):

        log_str = "Training Epoch {} - Total Loss = {} ".format(
            epoch, self.loss_meters["total_loss"].avg
        )

        if self.config.train.use_wandb:
            wandb.log(
                {
                    "epoch/train_total_loss": self.loss_meters["total_loss"].avg,
                    "epoch": epoch,
                }
            )

        log_str += "- Main Loss = {} ".format(self.loss_meters["main_loss"].avg)

        if self.config.train.use_wandb:
            wandb.log(
                {
                    "epoch/train_main_loss": self.loss_meters["main_loss"].avg,
                    "epoch": epoch,
                }
            )

        log_str += "- Spatial Attn Loss = {} ".format(
            self.loss_meters["spatial_location_loss"].avg,
        )

        if self.config.train.use_wandb:
            wandb.log(
                {
                    "epoch/train_spatial_attn_loss": self.loss_meters[
                        "spatial_location_loss"
                    ].avg,
                    "epoch": epoch,
                }
            )

        log_str += "- Temporal Attn Loss = {} ".format(
            self.loss_meters["temporal_location_loss"].avg,
        )

        if self.config.train.use_wandb:
            wandb.log(
                {
                    "epoch/train_temporal_attn_loss": self.loss_meters[
                        "temporal_location_loss"
                    ].avg,
                    "epoch": epoch,
                }
            )

        log_str += "- Aux Classification Loss = {} ".format(
            self.loss_meters["aux_classification_loss"].avg,
        )

        if self.config.train.use_wandb:
            wandb.log(
                {
                    "epoch/train_aux_class_loss": self.loss_meters[
                        "aux_classification_loss"
                    ].avg,
                    "epoch": epoch,
                }
            )

        log_str += "- Proto Loss = {} ".format(
            self.loss_meters["prototype_loss"].avg,
        )

        if self.config.train.use_wandb:
            wandb.log(
                {
                    "epoch/train_prototype_loss": self.loss_meters[
                        "prototype_loss"
                    ].avg,
                    "epoch": epoch,
                }
            )

        log_str += "- LR = {} ".format(
            lr,
        )

        if self.config.train.use_wandb:
            wandb.log({"lr": lr, "epoch": epoch})

        for eval_type in self.evaluator.keys():
            log_str += (
                "- "
                + eval_type.upper()
                + " = {}".format(self.evaluator[eval_type].compute())
            )

            if self.config.train.use_wandb:
                wandb.log(
                    {
                        "epoch/train_{}_score".format(eval_type): self.evaluator[
                            eval_type
                        ].compute(),
                        "epoch": epoch,
                    }
                )

        self.logger.info(log_str)

    def _val_epoch_summary(self, epoch):

        log_str = "{} Epoch {} - Main Loss = {} ".format(
            "Validation" if self.train else "Test",
            epoch,
            self.loss_meters["main_loss"].avg,
        )

        if self.config.train.use_wandb:
            wandb.log(
                {
                    "epoch/val_main_loss": self.loss_meters["main_loss"].avg,
                    "epoch": epoch,
                }
            )

        for eval_type in self.evaluator.keys():
            log_str += (
                "- "
                + eval_type.upper()
                + " = {} ".format(self.evaluator[eval_type].compute())
            )

            if self.config.train.use_wandb:
                wandb.log(
                    {
                        "epoch/val_{}_score".format(eval_type): self.evaluator[
                            eval_type
                        ].compute(),
                        "epoch": epoch,
                    }
                )

            log_str += "- Best Eval Metric = {}".format(
                self.evaluator[eval_type].compute()
            )

            if self.config.train.use_wandb:
                wandb.log(
                    {
                        "epoch/best_eval_metric".format(
                            eval_type
                        ): self.best_eval_metric,
                        "epoch": epoch,
                    }
                )

        self.logger.info(log_str)

    def _forward_path(self, data_dict):
        output_dict = self.model(data_dict)

        # Compute loss
        loss = self.criterion[
            "regression" if self.config.train.mode == "ef" else "classification"
        ](output_dict["x"].squeeze(1), data_dict["label"])

        with torch.no_grad():
            self.loss_meters["main_loss"].update(loss.detach().cpu().item())

        # Patch attention supervision
        attn_loss = 0
        if self.config.train.criterion.attn_lambda > 0:
            attn_loss = self.criterion["spatial_location"](
                output_dict["last_layer_patch_attn"], data_dict["lv_mask"]
            )
            loss += attn_loss * self.config.train.criterion.attn_lambda

        with torch.no_grad():
            self.loss_meters["spatial_location_loss"].update(
                attn_loss.detach().cpu().item() if attn_loss > 0 else attn_loss
            )

        # Frame location supervision
        frame_loc_loss = 0
        if self.config.train.criterion.frame_lambda > 0:
            frame_loc_loss = self.criterion["temporal_location"](
                output_dict["last_layer_frame_attn"],
                output_dict["ed_frames"],
                output_dict["ed_valid"],
                output_dict["es_frames"],
                output_dict["es_valid"],
            )
            loss += frame_loc_loss * self.config.train.criterion.frame_lambda

        with torch.no_grad():
            self.loss_meters["temporal_location_loss"].update(
                frame_loc_loss.detach().cpu().item()
                if frame_loc_loss > 0
                else frame_loc_loss
            )

        aux_class_loss = 0
        if self.config.train.criterion.classification_lambda > 0:
            aux_class_loss = self.criterion["aux_classification"](
                output_dict["x_class"], data_dict["class_label"]
            )
            loss += aux_class_loss * self.config.train.criterion.classification_lambda

        with torch.no_grad():
            self.loss_meters["aux_classification_loss"].update(
                aux_class_loss.detach().cpu().item()
                if aux_class_loss > 0
                else aux_class_loss
            )

        prototype_loss = 0
        if self.config.model.use_ppnet:
            prototype_loss = self.criterion["classification"](
                output_dict["logits"], data_dict["label"]
            )
            loss += prototype_loss
            if epoch > 5:
                loss += output_dict["ppc_loss"].mean()

        with torch.no_grad():
            self.loss_meters["prototype_loss"].update(
                prototype_loss.detach().cpu().item()
                if prototype_loss > 0
                else prototype_loss
            )

            self.loss_meters["total_loss"].update(loss.detach().cpu().item())

        return (
            loss,
            output_dict["x"],
            output_dict["patch_pos_embed"],
            output_dict["frame_pos_embed"],
            output_dict["vid_pos_embed"],
            output_dict["patch_attn"],
            output_dict["frame_attn"],
            output_dict["vid_attn"],
            output_dict["sampled_vid"],
            output_dict["logits"],
        )

    @staticmethod
    def _set_tqdm_description(iterator, log_mode, epoch, loss):

        iterator.set_description(
            "[Epoch {}] | {} | Loss: {:.4f}".format(epoch, log_mode, loss),
            refresh=True,
        )

    def log_wandb_summary(self):
        if self.config.train.use_wandb:
            wandb.run.summary["best_eval_metric"] = self.best_eval_metric

    def is_rank_0(self):
        return get_ddp_save_flag() or not self.ddp

    def reset_evaluator(self):
        for key in self.evaluator.keys():
            self.evaluator[key].reset()

    def reset_meters(self):
        for key in self.loss_meters.keys():
            self.loss_meters[key].reset()

    def update_evaluator(self, pred, label, proto_logits=None):
        for key in self.evaluator.keys():
            if key == "proto_acc":
                self.evaluator[key].update(y_pred=proto_logits, y_true=label)
            else:
                self.evaluator[key].update(y_pred=pred, y_true=label)

    def save_checkpoint(self, epoch, eval_metric):

        checkpoint = {
            "epoch": epoch,
            "best_eval_metric": self.best_eval_metric,
            "eval_metric": eval_metric,
        }

        for eval_type in self.evaluator.keys():
            checkpoint["metric_{}".format(eval_type)] = self.evaluator[
                eval_type
            ].compute()

        # Add model state dicts
        try:
            checkpoint["model"] = self.model.module.state_dict()
        except AttributeError:
            checkpoint.model = self.model.state_dict()

        # Add optimizer state dicts
        checkpoint["optimizer"] = self.optimizer.state_dict()

        # Add scheduler state dicts
        checkpoint["scheduler"] = self.scheduler.state_dict()

        # Save last_checkpoint
        checkpoint_path = os.path.join(self.save_dir, "checkpoint_last.pth")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info("last_checkpoint is saved for epoch {}.".format(epoch))

        # Save the best_checkpoint if performance improved
        if (eval_metric != self.best_eval_metric) or (
            self.config.train.mode == "pretrain"
        ):
            return

        # Save best_checkpoint
        checkpoint_path = os.path.join(self.save_dir, "checkpoint_best.pth")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(
            "best_checkpoint is saved for epoch {} with eval metric {}.".format(
                epoch, eval_metric
            )
        )

    def load_checkpoint(self):

        checkpoint = None

        if self.checkpoint_path:
            self.logger.info("Loading checkpoint from {}".format(self.checkpoint_path))
            checkpoint = torch.load(self.checkpoint_path)

            # Load model weights
            self.model.load_state_dict(checkpoint.pop("model"), strict=False)

            if self.train:
                # Load optimizer state
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))

                # Load scheduler state
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        return checkpoint
