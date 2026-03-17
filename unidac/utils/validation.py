import json
import os
from typing import Any, Dict, Optional

import numpy as np
import tqdm
import torch
import torch.utils.data.distributed
from torch import nn
from torch.utils.data import DataLoader

from unidac.utils.metrics import RunningMetric
from unidac.utils.misc import is_main_process
from unidac.utils.visualization import visualize_results
from unidac.utils.erp_geometry import erp_patch_to_cam_fast

import cv2
from PIL import Image
import pdb

min_steps_to_save = 10000

def log_losses(losses_all):
    for loss_name, loss_val in losses_all.items():
        print(f"Test/{loss_name}: ", loss_val)


# def update_best(metrics_all, metrics_best="abs_rel"):
#     curr_loss = []
#     for metrics_name, metrics_value in metrics_all.items():
#         if metrics_best in metrics_name:
#             curr_loss.append(metrics_value)

#     curr_loss = np.mean(curr_loss)
#     if curr_loss < validate.best_loss:
#         validate.best_loss = curr_loss
#         validate.best_metrics = metrics_all

#     for metrics_name, metrics_value in metrics_all.items():
#         try:
#             print(
#                 f"{metrics_name} {round(validate.best_metrics[metrics_name], 4)} ({round(metrics_value, 4)})"
#             )
#         except:
#             print(f"Error in best. {metrics_name} ({round(metrics_value, 4)})")


def save_model(
    metrics_all, state_dict, run_save_dir, step, config, best_loss, metrics_best="abs_rel", is_last_step=False
):
    curr_loss = []
    curr_dataset = config["data"]["train_dataset"]
    for metrics_name, metrics_value in metrics_all.items():
        if metrics_best in metrics_name:
            curr_loss.append(metrics_value)
    curr_loss = np.mean(curr_loss)

    # if curr_loss == validate.best_loss:
    if (curr_loss == best_loss and step > min_steps_to_save) or is_last_step or step % 20000 == 0:
        if is_last_step:
            postfix = f"it{step}-last"
        elif step % 20000 == 0:
            postfix = f"it{step}"
        else:    
            postfix = f"best_{metrics_best}"
        
        try:
            torch.save(
                state_dict, os.path.join(run_save_dir, f"{curr_dataset}-{postfix}.pt")
            )
            with open(
                os.path.join(run_save_dir, f"{curr_dataset}-config.json"), "w+"
            ) as fp:
                json.dump(config, fp)
        except OSError as e:
            print(f"Error while saving model: {e}")
        except:
            print("Generic error while saving")


def validate(
    model: nn.Module,
    test_loader: DataLoader,
    metrics_tracker: RunningMetric,
    context: torch.autocast,
    config: Dict[str, Any],
    run_id: Optional[str] = None,
    data_dir: Optional[str] = None,
    step: int = 0,
    vis: bool = False, 
    save_pcd: bool = False, # save pcd file, space consuming
    out_dir: Optional[str] = None,
    erp_mode: bool = False,
    max_num_samples: int = None,
    is_last_step: bool = False,
    vis_step: int = 1,
):
    ds_losses = {}
    device = model.device

    for i, batch in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        if max_num_samples is not None and i >= max_num_samples:
            break
        # print(f'Processing {i} / {len(test_loader)} batches')
        with context:
            if erp_mode:
                gt, mask, lat_range, long_range = batch["gt"].to(device), batch["mask"].to(device), batch["lat_range"].to(device), batch["long_range"].to(device)
                attn_mask = batch.get('attn_mask', None)
                lat_grid = batch.get('lat_grid', None)
                attn_mask = attn_mask.to(device) if attn_mask is not None else None
                lat_grid = lat_grid.to(device) if lat_grid is not None else None
                preds, losses, _ = model(batch["image"].to(device), lat_range, long_range, gt, mask, attn_mask=attn_mask, lat_grid=lat_grid)
                # if 'attn_mask' in batch.keys():
                #     attn_mask = batch['attn_mask'].to(device)
                #     preds, losses, _ = model(batch["image"].to(device), lat_range, long_range, gt, mask, attn_mask)
                # elif 'lat_grid' in batch.keys():
                #     lat_grid = batch['attn_grid'].to(device)
                #     preds, losses, _ = model(batch["image"].to(device), lat_range, long_range, gt, mask, lat_grid)
                # else:
                #     preds, losses, _ = model(batch["image"].to(device), lat_range, long_range, gt, mask)
            else:
                gt, mask = batch["gt"].to(device), batch["mask"].to(device)
                preds, losses, _ = model(batch["image"].to(device), gt, mask)

        losses = {k: v for l in losses.values() for k, v in l.items()}
        for loss_name, loss_val in losses.items():
            ds_losses[loss_name] = (
                loss_val.detach().cpu().item() + i * ds_losses.get(loss_name, 0.0)
            ) / (i + 1)

        if 'info' in batch.keys() and 'pred_scale_factor' in batch['info'].keys():
            scale_factor = batch['info']['pred_scale_factor'].to(device)
            preds *= scale_factor[:, None, None, None]
        

        metrics_tracker.accumulate_metrics(
            gt.permute(0, 2, 3, 1), preds.permute(0, 2, 3, 1), mask.permute(0, 2, 3, 1)
        )
        
        if vis and i % vis_step == 0:
            visualize_results(batch, preds, out_dir, config, data_dir, save_pcd=save_pcd, index=i)
                
    losses_all = ds_losses
    metrics_all = metrics_tracker.get_metrics()
    metrics_tracker.reset_metrics()

    if is_main_process():
        log_losses(losses_all=losses_all)
        # update_best(metrics_all=metrics_all, metrics_best="abs_rel")
        
        # update best
        metrics_best="abs_rel"
        curr_loss = []
        for metrics_name, metrics_value in metrics_all.items():
            if metrics_best in metrics_name:
                curr_loss.append(metrics_value)

        curr_loss = np.mean(curr_loss)
        if curr_loss < validate.best_loss:
            if step > min_steps_to_save:
                validate.best_loss = curr_loss
            validate.best_metrics = metrics_all

        for metrics_name, metrics_value in metrics_all.items():
            try:
                print(
                    f"{metrics_name} {round(validate.best_metrics[metrics_name], 4)} ({round(metrics_value, 4)})"
                )
            except:
                print(f"Error in best. {metrics_name} ({round(metrics_value, 4)})")
        
        # save model
        if out_dir is not None and run_id is not None:
            run_save_dir = os.path.join(out_dir, run_id)
            os.makedirs(run_save_dir, exist_ok=True)

            with open(os.path.join(run_save_dir, f"metrics_{step}.json"), "w") as f:
                json.dump({**losses_all, **metrics_all}, f)
            save_model(
                metrics_all=metrics_all,
                state_dict=model.state_dict(),
                config=config,
                metrics_best="abs_rel",
                run_save_dir=run_save_dir,
                step=step,
                best_loss=validate.best_loss,
                is_last_step=is_last_step,
            )
    return metrics_all