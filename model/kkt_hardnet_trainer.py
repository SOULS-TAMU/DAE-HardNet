import os
import torch
import re
import time
import numpy as np
import pandas as pd
import json

class KKT_HardNet_Trainer:
    def __init__(self, config_dir, model, train_loader, val_loader, test_loaders, optimizer, criterion,
                parameters_list, variables_list,
                pinn_reg_factor=1, hardnet_reg_factor=1, num_epochs=500, eta=1e-3, model_loss_tolerance=1e-4, save_checkpoint_iter=50, 
                taylor_offset=1e-6, checkpoint_path=None, device=None, gaussian_mean=0, gaussian_var=0, gaussian_scale=0, is_linear=False):
        self.config_dir = config_dir
        self.model_config_path = f"{self.config_dir}/model_config.json"
        self.model = model
        self.num_ys = len([v for v in variables_list if str(v).startswith("y")])
        self.sym_names = model.newton.sym_names
        self.res_fn = model.newton.res_fn
        self.orig_eq_viol_fn = model.newton._evaluate_orig_eq_res
        self.orig_ineq_viol_fn = model.newton._evaluate_orig_ineq_res
        self.eq_viol_fn = model.newton._evaluate_eq_res
        self.ineq_viol_fn = model.newton._evaluate_ineq_res
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loaders = test_loaders
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.eta = eta
        self.pinn_reg_factor = pinn_reg_factor
        self.hardnet_reg_factor = hardnet_reg_factor
        self.taylor_offset = taylor_offset
        self.gaussian_noise_mean = gaussian_mean
        self.gaussian_noise_var = gaussian_var
        self.gaussian_noise_scale = gaussian_scale
        self.model_loss_tolerance = model_loss_tolerance
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.use_newton = False
        self.losses_save_path = f"{self.config_dir}/kkt_hardnet_losses.npz"
        self.model_save_path = f"{self.config_dir}/kkt_hardnet_model.pth"
        self.predictions_save_path = f"{self.config_dir}/kkt_hardnet_predictions.csv"
        self.mse_mape_save_path = f"{self.config_dir}/kkt_hardnet_metrics.txt"

        self.train_data_losses = []
        self.train_data_losses_orig = []
        self.train_grad_losses = []
        self.test_data_losses = []
        self.test_data_losses_orig = []
        self.test_grad_losses = []
        self.train_pinn_losses = []
        self.train_pinn_losses_autograd = []
        self.test_pinn_losses = []
        self.test_pinn_losses_autograd = []
        self.train_abs_violation = []
        self.train_abs_violation_autograd = []
        self.test_abs_violation = []
        self.test_abs_violation_autograd = []
        self.epoch_times = []
        
        self.train_data_losses_before_projection = []
        self.train_pinn_losses_before_projection = []
        self.train_abs_pinn_losses_before_projection = []
        self.test_data_losses_before_projection = []
        self.test_pinn_losses_before_projection =[]
        self.test_abs_pinn_losses_before_projection = []

        # Saving the time taken each step
        self.backbone_times = []
        self.hat_gradient_times = []
        self.projection_times = []
        self.tilde_gradient_times = []
        self.backprop_times = []
        self.optimizer_step_times = []
        
        self.parameters_list = parameters_list
        self.variables_list = variables_list
        
        # ===============================
        # Differential Term Detection
        # ===============================
        self.has_differential_terms = False
        self.best_checkpoint_path = None
        self.required_derivatives = []
        self.max_diff_order = None
        diff_orders = []
        self.is_linear = is_linear

        for name in self.sym_names:
            if "_data" in name:   # optional: skip *_data symbols
                continue

            # Matches dy1dx1 (first-order spatial)
            if re.fullmatch(r"d(\d+)y\d+dx\d+", name):
                self.has_differential_terms = True
                diff_orders.append(1)
                meta_info = re.findall(r"\d+", name)
                y_idx, x_idx = meta_info[1:]
                self.required_derivatives.append({
                    'target': f'y{y_idx}',
                    'order': meta_info[0],
                    'wrt': [f'x{x_idx}'],
                    'symbol': name
                })

            # NEW: Matches dy1dt (first-order temporal)
            elif re.fullmatch(r"d(\d+)y(\d+)dt", name):
                self.has_differential_terms = True
                diff_orders.append(1)
                meta_info = re.findall(r"\d+", name)
                y_idx = meta_info[1]
                self.required_derivatives.append({
                    'target': f'y{y_idx}',
                    'order': meta_info[0],
                    'wrt': ['t'],
                    'symbol': name
                })

            # NEW: Matches explicit order with x or t: d2y1dx1dt, d3y2dtdx1dx2, d1y1dt, d2y1dtdt, etc.
            elif match := re.fullmatch(r"d(\d+)y(\d+)((?:d(?:x\d+|t))+)", name):
                order = int(match.group(1))
                y_idx = int(match.group(2))
                trailer = match.group(3)
                # returns ['x1','t','x2', ...]
                wrt_tokens = re.findall(r"d(x\d+|t)", trailer)
                if len(wrt_tokens) != order:
                    continue  # or raise ValueError if you want strictness
                self.has_differential_terms = True
                self.required_derivatives.append({
                    'target': f'y{y_idx}',
                    'order': order,
                    'wrt': wrt_tokens,   # e.g., ['x1','t']
                    'symbol': name
                })
                diff_orders.append(order)

        self.num_gradient_terms = len(self.required_derivatives)
        if self.has_differential_terms:
            self.max_diff_order = max(diff_orders)

        # ===============================
        # Input/Output Variable Detection
        # ===============================
        self.input_symbols = [s for s in self.sym_names if s.startswith("x") or s == "t"]
        self.output_symbols = [s for s in self.sym_names if re.fullmatch(r"y\d+", s)]

        # Checkpoint detection and model weights settings:
        # If checkpoint path provided, load weights before training
        # Checkpoint Directory
        self.save_checkpoint_iter = save_checkpoint_iter
        self.best_checkpoint_loss = float('inf')

        if checkpoint_path is not None and len(checkpoint_path) == 0:
            checkpoint_path = None
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # If the checkpoint contains model_state_dict, use it
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # This means the checkpoint is just the raw state dict
                self.model.load_state_dict(checkpoint)

            # If loss is there in checkpoint
            if "loss" in checkpoint:
                print("==> Updating the best loss value to: {}".format(checkpoint["loss"]))
                self.best_checkpoint_loss = checkpoint["loss"]
                if checkpoint["loss"] < self.eta:
                    print("✅ Current Loss is less than cutoff. Using Newton from the start.")
                    self.use_newton = True
                    

            print(f"✅ Loaded model weights from {checkpoint_path}")
        else:
            print(f"⚠️ No checkpoint file found. Starting training from scratch")

        

        # Make checkpoints directory
        self.checkpoint_dir = os.path.join(self.config_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _save_checkpoint(self, epoch, data_loss, data_loss_orig, consistency_loss,  grad_loss, pinn_loss, abs_violation, pinn_loss_autograd, abs_violation_autograd,
                                test_data_loss, test_data_loss_orig, test_consistency_loss, test_grad_loss, test_pinn_loss, test_abs_violation, test_pinn_loss_autograd, test_abs_violation_autograd,
                                data_loss_before_projection, pinn_loss_before_projection, abs_pinn_loss_before_projection, test_data_loss_before_projection, test_pinn_loss_before_projection, test_abs_pinn_loss_before_projection):
        loss = test_data_loss
        """Save a training checkpoint if loss improves."""
        if loss < self.best_checkpoint_loss:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"kkt_hardnet_checkpoint_hidden_dim_{self.model.hidden_dims}_depth_{self.model.model_depth}.pth"
            )

            torch.save({
                'epoch': epoch + 1,
                'hidden_dim': self.model.hidden_dims,
                'model_depth': self.model.model_depth,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'train_data_loss': data_loss,
                'train_data_loss_orig': data_loss_orig,
                'train_grad_loss': grad_loss,
                'train_pinn_loss': pinn_loss,
                'train_abs_violation': abs_violation,
                'train_pinn_loss_autograd': pinn_loss_autograd,
                'train_abs_violation_autograd': abs_violation_autograd,
                'test_data_loss': test_data_loss,
                'test_data_loss_orig': test_data_loss_orig,
                'test_grad_loss': test_grad_loss,
                'test_pinn_loss': test_pinn_loss,
                'test_abs_violation': test_abs_violation,
                'test_pinn_loss_autograd': test_pinn_loss_autograd,
                'test_abs_violation_autograd': test_abs_violation_autograd,
                'data_loss_before_projection': data_loss_before_projection,
                'pinn_loss_before_projection': pinn_loss_before_projection,
                'abs_pinn_loss_before_projection': abs_pinn_loss_before_projection,
                'test_data_loss_before_projection': test_data_loss_before_projection,
                'test_pinn_loss_before_projection': test_pinn_loss_before_projection,
                'test_abs_pinn_loss_before_projection': test_abs_pinn_loss_before_projection
            }, checkpoint_path)

            self.best_checkpoint_loss = loss
            # print(f"💾 Checkpoint saved at epoch {epoch+1} with loss {loss:.8f}")
            self.best_checkpoint_path = checkpoint_path
        else:
            # print(f"⚠️ No checkpoint saved at epoch {epoch+1} (loss {loss:.8f} >= best {self.best_checkpoint_loss:.8f})")
            pass
    
    # ADD inside class KKT_HardNet_Trainer (e.g., right after __init__)
    def _compute_per_param_deltas(self, x_batch):
        """Return concatenated NN outputs at x+delta e_k for each input dimension k.
        Shape: (B, y_dim * n_params)"""
        y_delta_blocks = []
        for k in range(x_batch.shape[1]):
            x_delta = x_batch.clone()
            x_delta[:, k] = x_delta[:, k] + self.taylor_offset
            y_delta = self.model.nn(x_delta)
            y_delta_blocks.append(y_delta[:, :self.num_ys])
        return torch.cat(y_delta_blocks, dim=1)


    def train_model(self):
        self.model.train()
        total_loss = 0
        total_data_loss = 0
        total_data_loss_orig = 0
        total_grad_loss = 0
        total_consistency_loss = 0
        total_pinn_loss = 0
        total_pinn_loss_autograd = 0
        total_abs_pinn_loss = 0
        total_abs_pinn_loss_autograd = 0
        
        total_data_loss_hat = 0
        total_pinn_loss_hat = 0
        total_abs_pinn_loss_hat = 0
        total_diff = 0

        backbone_time = 0
        hat_gradient_time = 0
        projection_time = 0
        tilde_gradient_time = 0
        backprop_time = 0
        optimizer_step_time = 0


        for x_batch, y_batch, y_data_batch in self.train_loader:
            x_batch, y_batch, y_data_batch = (
                x_batch.to(self.device),
                y_batch.to(self.device),
                y_data_batch.to(self.device)
            )

            # print("Seed of torch:", torch.initial_seed())


            y_batch_orig = y_batch.clone()

            if self.gaussian_noise_scale > 0:
                noise = torch.randn_like(y_batch) * self.gaussian_noise_var + self.gaussian_noise_mean
                y_batch = y_batch + self.gaussian_noise_scale * noise


            # Add small offset to inputs for Taylor expansion stability
            x_batch = x_batch # - self.taylor_offset * torch.ones_like(x_batch, device=self.device)

            x_batch.requires_grad_(True)  # Enable autograd for derivative computation
            self.optimizer.zero_grad()

            # Step 1: Predict base NN outputs (y1, y2, ...)
            start_time = time.time()
            y_hat_base = self.model.nn(x_batch)  # shape: (B, num_outputs)
            end_time = time.time()
            backbone_time += end_time - start_time

            grad_outputs = []
            start_time = time.time()
            # print(self.required_derivatives)
            if self.required_derivatives:
                # map output names to columns
                output_map = {f'y{i+1}': y_hat_base[:, i] for i in range(y_hat_base.shape[1])}

                for item in self.required_derivatives:
                    target_name = item['target']        # e.g., 'y1'
                    order = int(item['order'])         # integer order
                    wrt_list = item.get('wrt', [])     # e.g., ['x1'] or ['x1','x2']
                    y_target = output_map[target_name] # shape [B]

                    # convert wrt variable names to indices: 'x1' -> 0, 'x2' -> 1, ...
                    has_t = 't' in wrt_list
                    wrt_indices = [0 if v == 't' else (int(v[1:]) - 1 + (1 if has_t else 0)) for v in wrt_list]
                    
                    # build differentiation sequence (length == order)
                    # - if user provided exactly 'order' vars, use them
                    # - if single entry provided, repeat it 'order' times
                    # - if fewer than order provided, repeat the last one to fill
                    if len(wrt_indices) == order:
                        seq = wrt_indices
                    elif len(wrt_indices) == 1:
                        seq = [wrt_indices[0]] * order
                    elif 1 < len(wrt_indices) < order:
                        seq = wrt_indices + [wrt_indices[-1]] * (order - len(wrt_indices))
                    else:  # if none provided, default to differentiate w.r.t. x1 repeatedly
                        seq = [0] * order

                    # iterative differentiation following the sequence
                    grad = y_target         # shape [B]
                    # we need a scalar-valued grad_outputs of same shape for autograd.grad
                    for j, var_idx in enumerate(seq):
                        # compute gradients of 'grad' w.r.t. x_batch (returns [B, input_dim])
                        grads_wrt_x = torch.autograd.grad(
                            grad,
                            x_batch,
                            grad_outputs=torch.ones_like(grad),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True
                        )[0]  # shape [B, input_dim]

                        # select the column corresponding to var_idx
                        grad = grads_wrt_x[:, var_idx]  # shape [B]

                    # now 'grad' is the desired derivative array shape [B]
                    grad_outputs.append(grad)   # collect for later concatenation/use

            end_time = time.time()
            hat_gradient_time += end_time - start_time

            # Ensure all grad_outputs are 2D column tensors [B, 1]
            grad_outputs = [g.unsqueeze(1) if g.ndim == 1 else g for g in grad_outputs]

            # Step 3: Concatenate all outputs in the order of sym_names
            # y_head = y_hat_base[:, :y_batch.shape[-1]]  # original y vars
            # y_tail = y_hat_base[:, y_batch.shape[-1]:]  # remaining vars after y
            y_head = y_hat_base[:, :self.num_ys]  # original y vars
            y_tail = y_hat_base[:, self.num_ys:]  # remaining vars after y

            if grad_outputs:
                y_hat = torch.cat([y_head] + grad_outputs + [y_tail], dim=1)
            else:
                y_hat = y_hat_base

            
            if not self.use_newton:
                # x_batch_delta = x_batch + self.taylor_offset * torch.ones_like(x_batch, device=self.device)
                # # at delta distance
                # y_hat_delta = self.model.nn(x_batch_delta)
                # # x_input = torch.cat([x_batch, y_hat[:, :y_batch.shape[-1]]] + grad_outputs, dim=1)
                # x_input = torch.cat([x_batch, y_hat[:, :y_batch.shape[-1]]], dim=1)
                # x_input = torch.cat([x_input, y_hat_delta[:, :y_batch.shape[-1]]], dim=1)
                # x_input = torch.cat([x_input, y_hat_delta[:, :y_batch.shape[-1]]], dim=1)
                
                y_dim = y_batch.shape[-1]
                y_hat_deltas = self._compute_per_param_deltas(x_batch)  # (B, y_dim * n_params)
                x_input = torch.cat([x_batch, y_hat[:, :self.num_ys]], dim=1)
                x_input = torch.cat([x_input, y_hat_deltas], dim=1)

                data_loss = self.criterion(y_hat[:, :y_batch.shape[-1]], y_batch)
                data_loss_yhat = self.criterion(y_hat[:, :y_batch.shape[-1]], y_batch)
                data_loss_orig = self.criterion(y_hat[:, :y_batch.shape[-1]], y_batch_orig)
                consistency_loss = torch.tensor(0.0, device=self.device)
                grad_loss = torch.tensor(0.0, device=self.device)
                

                # # print(y_hat.shape, x_input.shape)
                # print("DAE Hardnet Non-Newton Train y_hat shape: ", y_hat.shape)
                # print("DAE Hardnet Non Newton Train x_input shape: ", x_input.shape)

                ## PINN and absolute violation wrt hat predictions
                eq_res_hat = self.orig_eq_viol_fn(y_hat, x_input)
                ineq_res_hat = self.orig_ineq_viol_fn(y_hat, x_input)

                ineq_res_hat = torch.clamp_min(ineq_res_hat,0)
                combined_hat = torch.cat([eq_res_hat, ineq_res_hat], dim=1)
                abs_pinn_loss_hat = torch.mean(combined_hat.abs().sum(dim=1))
                pinn_loss_hat = torch.linalg.norm(combined_hat, dim=1).mean()
                # pinn_loss_autograd_hat = pinn_loss_hat.clone()
                # abs_pinn_loss_autograd_hat = abs_pinn_loss_hat.clone()

                eq_res = self.orig_eq_viol_fn(y_hat, x_input)
                ineq_res = self.orig_ineq_viol_fn(y_hat, x_input)

                ineq_res = torch.clamp_min(ineq_res,0)
                combined = torch.cat([eq_res, ineq_res], dim=1)
                abs_pinn_loss = torch.mean(combined.abs().sum(dim=1))
                pinn_loss = torch.linalg.norm(combined, dim=1).mean()
                pinn_loss_autograd = pinn_loss.clone()
                abs_pinn_loss_autograd = abs_pinn_loss.clone()
                # print("Computing PINN loss on base NN output... Done")
                # loss = self.pinn_reg_factor * pinn_loss
                loss = data_loss + self.pinn_reg_factor * pinn_loss
                
            else:
                # x_batch_delta = x_batch + self.taylor_offset * torch.ones_like(x_batch, device=self.device)
                # # at delta distance
                # y_hat_delta = self.model.nn(x_batch_delta)
                # # x_input = torch.cat([x_batch, y_hat[:, :y_batch.shape[-1]]] + grad_outputs, dim=1)
                # x_input = torch.cat([x_batch, y_hat[:, :y_batch.shape[-1]]], dim=1)
                # x_input = torch.cat([x_input, y_hat_delta[:, :y_batch.shape[-1]]], dim=1)
                # x_input = torch.cat([x_input, y_hat_delta[:, :y_batch.shape[-1]]], dim=1)
                
                y_dim = y_batch.shape[-1]
                y_hat_deltas = self._compute_per_param_deltas(x_batch)  # (B, y_dim * n_params) 
                x_input = torch.cat([x_batch, y_hat[:, :self.num_ys]], dim=1)
                x_input = torch.cat([x_input, y_hat_deltas], dim=1)             
                # print("DAE Hardnet Newton Train y_hat shape: ", y_hat.shape)
                # print("DAE Hardnet Newton Train x_input shape: ", x_input.shape)   
                
                start_time = time.time()
                y_tilde = self.model.newton(y_hat, x_input)
                end_time = time.time()
                projection_time += end_time - start_time
                y_taylor_recomputed = self.model.newton.taylor_fn(y_tilde, x_input)
                y_tilde[:, :self.num_ys] = y_taylor_recomputed
                
                # Step B: Recompute gradients for y_tilde
                start_time = time.time()
                grad_outputs_tilde = []
                if self.required_derivatives:
                    output_map_tilde = {f'y{i+1}': y_tilde[:, i] for i in range(y_tilde.shape[1])}

                    for item in self.required_derivatives:
                        target_name = item['target']
                        order = int(item['order'])
                        wrt_list = item.get('wrt', [])
                        y_target = output_map_tilde[target_name]

                        # convert wrt variable names to indices: 'x1' -> 0, 'x2' -> 1, ...
                        has_t = 't' in wrt_list
                        wrt_indices = [0 if v == 't' else (int(v[1:]) - 1 + (1 if has_t else 0)) for v in wrt_list]
                    
                        if len(wrt_indices) == order:
                            seq = wrt_indices
                        elif len(wrt_indices) == 1:
                            seq = [wrt_indices[0]] * order
                        elif 1 < len(wrt_indices) < order:
                            seq = wrt_indices + [wrt_indices[-1]] * (order - len(wrt_indices))
                        else:
                            seq = [0] * order

                        grad = y_target
                        for j, var_idx in enumerate(seq):
                            grads_wrt_x = torch.autograd.grad(
                                grad,
                                x_batch,
                                grad_outputs=torch.ones_like(grad),
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True
                            )[0]
                            grad = grads_wrt_x[:, var_idx]

                        grad_outputs_tilde.append(grad)

                end_time = time.time()
                tilde_gradient_time += end_time - start_time

                # print("Recomputing Derivatives for Newton corrected output... Done")
                grad_outputs_tilde = [g.unsqueeze(1) if g.ndim == 1 else g for g in grad_outputs_tilde]

                # Step C: Reconstruct augmented y_tilde with derivative terms
                # y_head_tilde = y_tilde[:, :y_batch.shape[-1]]
                # y_tail_tilde = y_tilde[:, y_batch.shape[-1] + len(grad_outputs_tilde):]
                y_head_tilde = y_tilde[:, :self.num_ys]
                y_tail_tilde = y_tilde[:, self.num_ys + len(grad_outputs_tilde):]

                if grad_outputs_tilde:
                    y_tilde_aug = torch.cat([y_head_tilde] + grad_outputs_tilde + [y_tail_tilde], dim=1)
                else:
                    y_tilde_aug = y_tilde

                if grad_outputs:
                    y_hat_aug = torch.cat([y_head_tilde] + grad_outputs + [y_tail_tilde], dim=1)
                else:
                    y_hat_aug = y_tilde

                # print("Reconstructing Newton corrected output with recomputed derivatives... Done")
                # Step D: Now use y_tilde_aug for residual computation
                data_loss = self.criterion(y_tilde[:, :y_batch.shape[-1]], y_batch)
                data_loss_yhat = self.criterion(y_hat[:, :y_batch.shape[-1]], y_batch)
                data_loss_orig = self.criterion(y_tilde[:, :y_batch.shape[-1]], y_batch_orig)

                # data_loss_tilde_hat = self.criterion(y_tilde[:, :y_batch.shape[-1] + len(grad_outputs)], y_hat[:, :y_batch.shape[-1] + len(grad_outputs)])

                grad_loss = self.criterion(y_tilde_aug, y_tilde)
                # grad_loss = self.criterion(y_hat_aug, y_tilde)
                
                consistency_loss = self.criterion(y_tilde, y_hat)

                ## PINN and absolute violation wrt hat predictions
                eq_res_hat = self.orig_eq_viol_fn(y_hat, x_input)
                ineq_res_hat = self.orig_ineq_viol_fn(y_hat, x_input)

                ineq_res_hat = torch.clamp_min(ineq_res_hat,0)
                combined_hat = torch.cat([eq_res_hat, ineq_res_hat], dim=1)
                abs_pinn_loss_hat = torch.mean(combined_hat.abs().sum(dim=1))
                pinn_loss_hat = torch.linalg.norm(combined_hat, dim=1).mean()
                # pinn_loss_autograd_hat = pinn_loss_hat.clone()
                # abs_pinn_loss_autograd_hat = abs_pinn_loss_hat.clone()

                x_input = torch.cat([x_batch, y_tilde[:, :self.num_ys]], dim=1)
                x_input = torch.cat([x_input, y_hat_deltas], dim=1)   
                
                eq_res = self.orig_eq_viol_fn(y_tilde, x_input)
                ineq_res = self.orig_ineq_viol_fn(y_tilde, x_input)
                
                ineq_res = torch.clamp_min(ineq_res, 0)
                combined = torch.cat([eq_res, ineq_res], dim=1)
                abs_pinn_loss = torch.mean(combined.abs().sum(dim=1))
                pinn_loss = torch.linalg.norm(combined, dim=1).mean()

                # Computing autograd pinn loss
                eq_res_autograd = self.orig_eq_viol_fn(y_tilde_aug, x_input)
                ineq_res_autograd = self.orig_ineq_viol_fn(y_tilde_aug, x_input)
                
                ineq_res_autograd = torch.clamp_min(ineq_res_autograd, 0)
                combined_autograd = torch.cat([eq_res_autograd, ineq_res_autograd], dim=1)
                abs_pinn_loss_autograd = torch.mean(combined_autograd.abs().sum(dim=1))
                pinn_loss_autograd = torch.linalg.norm(combined_autograd, dim=1).mean()

                # print("Computing PINN loss on Newton corrected output... Done")
                # loss = data_loss + grad_loss# + self.pinn_reg_factor * pinn_loss_autograd # + data_loss_tilde_hat # + data_loss_yhat  # + self.pinn_reg_factor * pinn_loss
                loss = data_loss + self.hardnet_reg_factor * grad_loss

            # Step 5: Backpropagation and optimizer step
            start_time = time.time()
            loss.backward()
            end_time = time.time()
            backprop_time += end_time - start_time
            start_time = time.time()
            self.optimizer.step()
            end_time = time.time()
            optimizer_step_time += end_time - start_time

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_data_loss += data_loss.item() * batch_size
            total_data_loss_orig += data_loss_orig.item() * batch_size
            total_grad_loss += grad_loss.item() * batch_size
            total_consistency_loss += consistency_loss.item() * batch_size
            total_pinn_loss += pinn_loss.item() * batch_size
            total_abs_pinn_loss += abs_pinn_loss.item() * batch_size
            total_pinn_loss_autograd += pinn_loss_autograd.item() * batch_size
            total_abs_pinn_loss_autograd += abs_pinn_loss_autograd.item() * batch_size
            
            total_data_loss_hat += data_loss_yhat.item() * batch_size
            total_pinn_loss_hat += pinn_loss_hat.item() * batch_size
            total_abs_pinn_loss_hat += abs_pinn_loss_hat.item() * batch_size
            
        n_samples = len(self.train_loader.dataset)
        return (
            total_loss / n_samples,
            total_data_loss / n_samples,
            total_data_loss_orig / n_samples,
            total_consistency_loss / n_samples,
            total_grad_loss / n_samples,
            total_pinn_loss / n_samples,
            total_abs_pinn_loss /n_samples,
            total_pinn_loss_autograd / n_samples,
            total_abs_pinn_loss_autograd / n_samples,
            total_data_loss_hat / n_samples,
            total_pinn_loss_hat / n_samples,
            total_abs_pinn_loss_hat / n_samples,
            backbone_time,
            hat_gradient_time,
            projection_time,
            tilde_gradient_time,
            backprop_time,
            optimizer_step_time
        )

    def test_model(self):
        self.model.eval()

        # Initialize global totals
        total_loss = 0
        total_data_loss = 0
        total_data_loss_orig = 0
        total_grad_loss = 0
        total_consistency_loss = 0
        total_pinn_loss = 0
        total_abs_pinn_loss = 0
        total_pinn_loss_autograd = 0
        total_abs_pinn_loss_autograd = 0
        total_samples = 0
        
        total_data_loss_hat = 0
        total_pinn_loss_hat = 0
        total_abs_pinn_loss_hat = 0

        # Loop over each test loader in the list
        for i, test_loader in enumerate(self.test_loaders):
            kkt_metadata = test_loader.kkt_info

            for x_test_batch, y_test_batch, y_data_test_batch in test_loader:
                x_test_batch = x_test_batch.to(self.device).requires_grad_(True)
                y_test_batch = y_test_batch.to(self.device)
                y_data_test_batch = y_data_test_batch.to(self.device)
                # Add small offset to inputs for Taylor expansion stability
                x_test_batch = x_test_batch #- self.taylor_offset * torch.ones_like(x_test_batch, device=self.device)

                # Step 1: Base NN predictions
                y_hat_base = self.model.nn(x_test_batch)

                grad_outputs = []
                if self.required_derivatives:
                    output_map = {f'y{i+1}': y_hat_base[:, i] for i in range(y_hat_base.shape[1])}

                    for item in self.required_derivatives:
                        target_name = item['target']
                        order = int(item['order'])
                        wrt_list = item.get('wrt', [])
                        y_target = output_map[target_name]

                        # convert wrt variable names to indices: 'x1' -> 0, 'x2' -> 1, ...
                        has_t = 't' in wrt_list
                        wrt_indices = [0 if v == 't' else (int(v[1:]) - 1 + (1 if has_t else 0)) for v in wrt_list]
                    

                        if len(wrt_indices) == order:
                            seq = wrt_indices
                        elif len(wrt_indices) == 1:
                            seq = [wrt_indices[0]] * order
                        elif 1 < len(wrt_indices) < order:
                            seq = wrt_indices + [wrt_indices[-1]] * (order - len(wrt_indices))
                        else:
                            seq = [0] * order

                        grad = y_target
                        for var_idx in seq:
                            grads_wrt_x = torch.autograd.grad(
                                grad,
                                x_test_batch,
                                grad_outputs=torch.ones_like(grad),
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True
                            )[0]
                            grad = grads_wrt_x[:, var_idx]

                        grad_outputs.append(grad)

                grad_outputs = [g.unsqueeze(1) if g.ndim == 1 else g for g in grad_outputs]

                # y_head = y_hat_base[:, :y_test_batch.shape[-1]]
                # y_tail = y_hat_base[:, y_test_batch.shape[-1]:]
                y_head = y_hat_base[:, :self.num_ys]
                y_tail = y_hat_base[:, self.num_ys:]

                if grad_outputs:
                    y_hat = torch.cat([y_head] + grad_outputs + [y_tail], dim=1)
                else:
                    y_hat = y_hat_base

                if self.use_newton:
                    # x_test_batch_delta = x_test_batch + self.taylor_offset * torch.ones_like(x_test_batch, device=self.device)
                    # # at delta distance
                    # y_hat_delta = self.model.nn(x_test_batch_delta)
                    # # x_input = torch.cat([x_test_batch, y_hat[:, :y_test_batch.shape[-1]]] + grad_outputs, dim=1)
                    # x_input = torch.cat([x_test_batch, y_hat[:, :y_test_batch.shape[-1]]], dim=1)
                    # x_input = torch.cat([x_input, y_hat_delta[:, :y_test_batch.shape[-1]]], dim=1)
                    # x_input = torch.cat([x_input, y_hat_delta[:, :y_test_batch.shape[-1]]], dim=1)
                    
                    y_dim = y_test_batch.shape[-1]
                    y_hat_deltas = self._compute_per_param_deltas(x_test_batch)
                    x_input = torch.cat([x_test_batch, y_hat[:, :self.num_ys]], dim=1)
                    x_input = torch.cat([x_input, y_hat_deltas], dim=1)

                    # print("DAE Hardnet Newton Train y_hat shape: ", y_hat.shape)
                    # print("DAE Hardnet Newton Train x_input shape: ", x_input.shape)

                    # Step A: Run Newton correction
                    y_tilde = self.model.newton(y_hat, x_input)
                    y_taylor_recomputed = self.model.newton.taylor_fn(y_tilde, x_input)
                    y_tilde[:, :self.num_ys] = y_taylor_recomputed

                    # Step B: Recompute gradients for y_tilde
                    grad_outputs_tilde = []
                    if self.required_derivatives:
                        output_map_tilde = {f'y{i+1}': y_tilde[:, i] for i in range(y_tilde.shape[1])}

                        for item in self.required_derivatives:
                            target_name = item['target']
                            order = int(item['order'])
                            wrt_list = item.get('wrt', [])
                            y_target = output_map_tilde[target_name]

                            # convert wrt variable names to indices: 'x1' -> 0, 'x2' -> 1, ...
                            has_t = 't' in wrt_list
                            wrt_indices = [0 if v == 't' else (int(v[1:]) - 1 + (1 if has_t else 0)) for v in wrt_list]
                            
                            if len(wrt_indices) == order:
                                seq = wrt_indices
                            elif len(wrt_indices) == 1:
                                seq = [wrt_indices[0]] * order
                            elif 1 < len(wrt_indices) < order:
                                seq = wrt_indices + [wrt_indices[-1]] * (order - len(wrt_indices))
                            else:
                                seq = [0] * order

                            grad = y_target
                            for j, var_idx in enumerate(seq):
                                grads_wrt_x = torch.autograd.grad(
                                    grad,
                                    x_test_batch,
                                    grad_outputs=torch.ones_like(grad),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True
                                )[0]
                                grad = grads_wrt_x[:, var_idx]

                            grad_outputs_tilde.append(grad)

                    grad_outputs_tilde = [g.unsqueeze(1) if g.ndim == 1 else g for g in grad_outputs_tilde]

                    # Step C: Reconstruct augmented y_tilde with derivative terms
                    # y_head_tilde = y_tilde[:, :y_test_batch.shape[-1]]
                    # y_tail_tilde = y_tilde[:, y_test_batch.shape[-1] + len(grad_outputs_tilde):]
                    y_head_tilde = y_tilde[:, :self.num_ys]
                    y_tail_tilde = y_tilde[:, self.num_ys + len(grad_outputs_tilde):]

                    if grad_outputs_tilde:
                        y_tilde_aug = torch.cat([y_head_tilde] + grad_outputs_tilde + [y_tail_tilde], dim=1)
                    else:
                        y_tilde_aug = y_tilde

                    if grad_outputs:
                        y_hat_aug = torch.cat([y_head_tilde] + grad_outputs + [y_tail_tilde], dim=1)
                    else:
                        y_hat_aug = y_tilde

                    data_loss = self.criterion(y_tilde[:, :y_test_batch.shape[-1]], y_test_batch)
                    data_loss_yhat = self.criterion(y_hat[:, :y_test_batch.shape[-1]], y_test_batch)
                    # data_loss_tilde_hat = self.criterion(y_tilde[:, :y_test_batch.shape[-1] + len(grad_outputs)], y_hat[:, :y_test_batch.shape[-1] + len(grad_outputs)])
                    data_loss_orig = self.criterion(y_tilde[:, :y_test_batch.shape[-1]], y_test_batch)
                    consistency_loss = self.criterion(y_tilde, y_hat)
                    grad_loss = self.criterion(y_tilde_aug, y_tilde)
                    # grad_loss = self.criterion(y_hat_aug, y_tilde)
                    
                    ## PINN and absolute violation wrt hat predictions
                    eq_res_hat = self.orig_eq_viol_fn(y_hat, x_input)
                    ineq_res_hat = self.orig_ineq_viol_fn(y_hat, x_input)

                    ineq_res_hat = torch.clamp_min(ineq_res_hat,0)
                    combined_hat = torch.cat([eq_res_hat, ineq_res_hat], dim=1)
                    abs_pinn_loss_hat = torch.mean(combined_hat.abs().sum(dim=1))
                    pinn_loss_hat = torch.linalg.norm(combined_hat, dim=1).mean()
                    # pinn_loss_autograd_hat = pinn_loss_hat.clone()
                    # abs_pinn_loss_autograd_hat = abs_pinn_loss_hat.clone()

                    x_input = torch.cat([x_test_batch, y_tilde[:, :self.num_ys]], dim=1)
                    x_input = torch.cat([x_input, y_hat_deltas], dim=1)   

                    eq_res = self.orig_eq_viol_fn(y_tilde, x_input)
                    ineq_res = self.orig_ineq_viol_fn(y_tilde, x_input)
                    

                    ineq_res = torch.clamp_min(ineq_res, 0)
                    combined = torch.cat([eq_res, ineq_res], dim=1)
                    abs_pinn_loss = torch.mean(combined.abs().sum(dim=1))
                    pinn_loss = torch.linalg.norm(combined, dim=1).mean()

                    # Computing autograd pinn loss
                    eq_res_autograd = self.orig_eq_viol_fn(y_tilde_aug, x_input)
                    ineq_res_autograd = self.orig_ineq_viol_fn(y_tilde_aug, x_input)
                    
                    ineq_res_autograd = torch.clamp_min(ineq_res_autograd, 0)
                    combined_autograd = torch.cat([eq_res_autograd, ineq_res_autograd], dim=1)
                    abs_pinn_loss_autograd = torch.mean(combined_autograd.abs().sum(dim=1))
                    pinn_loss_autograd = torch.linalg.norm(combined_autograd, dim=1).mean()

                    # loss = data_loss + grad_loss # + self.pinn_reg_factor * pinn_loss_autograd #+ data_loss_tilde_hat # + data_loss_yhat  # + self.pinn_reg_factor * pinn_loss
                    loss = data_loss + self.hardnet_reg_factor * grad_loss
                else:
                    # x_batch_delta = x_test_batch + self.taylor_offset * torch.ones_like(x_test_batch, device=self.device)

                    # # at delta distance
                    # y_hat_delta = self.model.nn(x_batch_delta)
                    # # x_input = torch.cat([x_test_batch, y_hat[:, :y_test_batch.shape[-1]]] + grad_outputs, dim=1)
                    # x_input = torch.cat([x_test_batch, y_hat[:, :y_test_batch.shape[-1]]], dim=1)
                    # x_input = torch.cat([x_input, y_hat_delta[:, :y_test_batch.shape[-1]]], dim=1)
                    # x_input = torch.cat([x_input, y_hat_delta[:, :y_test_batch.shape[-1]]], dim=1)
                    
                    y_dim = y_test_batch.shape[-1]
                    y_hat_deltas = self._compute_per_param_deltas(x_test_batch)
                    x_input = torch.cat([x_test_batch, y_hat[:, :self.num_ys]], dim=1)
                    x_input = torch.cat([x_input, y_hat_deltas], dim=1)

                    # print("DAE Hardnet Non-Newton Train y_hat shape: ", y_hat.shape)
                    # print("DAE Hardnet Non-Newton Train x_input shape: ", x_input.shape)
                    
                    ## PINN and absolute violation wrt hat predictions
                    eq_res_hat = self.orig_eq_viol_fn(y_hat, x_input)
                    ineq_res_hat = self.orig_ineq_viol_fn(y_hat, x_input)

                    ineq_res_hat = torch.clamp_min(ineq_res_hat,0)
                    combined_hat = torch.cat([eq_res_hat, ineq_res_hat], dim=1)
                    abs_pinn_loss_hat = torch.mean(combined_hat.abs().sum(dim=1))
                    pinn_loss_hat = torch.linalg.norm(combined_hat, dim=1).mean()
                    # pinn_loss_autograd_hat = pinn_loss_hat.clone()
                    # abs_pinn_loss_autograd_hat = abs_pinn_loss_hat.clone()
                    
                    eq_res = self.orig_eq_viol_fn(y_hat, x_input)
                    ineq_res = torch.clamp_min(self.orig_ineq_viol_fn(y_hat, x_input), 0)
                    combined = torch.cat([eq_res, ineq_res], dim=1)

                    abs_pinn_loss = torch.mean(combined.abs().sum(dim=1))
                    pinn_loss = torch.linalg.norm(combined, dim=1).mean()
                    pinn_loss_autograd = pinn_loss.clone()  
                    abs_pinn_loss_autograd = abs_pinn_loss.clone()
                    data_loss = self.criterion(y_hat[:, :y_test_batch.shape[-1]], y_test_batch)
                    data_loss_yhat = self.criterion(y_hat[:, :y_test_batch.shape[-1]], y_test_batch)
                    data_loss_orig = self.criterion(y_hat[:, :y_test_batch.shape[-1]], y_test_batch)
                    consistency_loss = torch.tensor(0.0, device=self.device)
                    grad_loss = torch.tensor(0.0, device=self.device)
                    loss = data_loss + self.pinn_reg_factor * pinn_loss
                    # loss = self.pinn_reg_factor * pinn_loss

                batch_size = x_test_batch.size(0)
                total_loss += loss.item() * batch_size
                total_data_loss += data_loss.item() * batch_size
                total_data_loss_orig += data_loss_orig.item() * batch_size
                total_consistency_loss += consistency_loss.item() * batch_size
                total_grad_loss += grad_loss.item() * batch_size
                total_pinn_loss += pinn_loss.item() * batch_size
                total_abs_pinn_loss += abs_pinn_loss.item() * batch_size
                total_pinn_loss_autograd += pinn_loss_autograd.item() * batch_size
                total_abs_pinn_loss_autograd += abs_pinn_loss_autograd.item() * batch_size
                total_samples += batch_size
                
                total_data_loss_hat += data_loss_yhat.item() * batch_size
                total_pinn_loss_hat += pinn_loss_hat.item() * batch_size
                total_abs_pinn_loss_hat += abs_pinn_loss_hat.item() * batch_size

        # Compute global averages across all test loaders
        return (
            total_loss / total_samples,
            total_data_loss / total_samples,
            total_data_loss_orig / total_samples,
            total_consistency_loss / total_samples,
            total_grad_loss / total_samples,
            total_pinn_loss / total_samples,
            total_abs_pinn_loss / total_samples,
            total_pinn_loss_autograd / total_samples,
            total_abs_pinn_loss_autograd / total_samples,
            total_data_loss_hat / total_samples,
            total_pinn_loss_hat / total_samples,
            total_abs_pinn_loss_hat / total_samples
        )



    def display_results(self, epoch, avg_loss, avg_test_loss,
                        data_loss, data_loss_orig, consistency_loss, grad_loss, pinn_loss, abs_violation, pinn_loss_autograd, abs_violation_autograd,
                        test_data_loss, test_data_loss_orig, test_consistency_loss, test_grad_loss, test_pinn_loss, test_abs_violation, test_pinn_loss_autograd, test_abs_violation_autograd,
                        data_loss_before_projection, pinn_loss_before_projection, abs_pinn_loss_before_projection,
                        test_data_loss_before_projection, test_pinn_loss_before_projection, test_abs_pinn_loss_before_projection):
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"[Epoch {epoch + 1}]")
            print(f"  🔧 Train Loss = {avg_loss:.6f} | Data = {data_loss:.6f}", end='')
            # print(f", Data(Orig) = {data_loss_orig:.6f}", end='')
            # print(f", Consistency = {consistency_loss:.8f}", end='')
            print(f", Grad = {grad_loss:.6f}", end='')
            
            print(f", PINN = {pinn_loss:.6f}",end='')
            print(f", AV = {abs_violation:.6f}",end='')

            print(f", AutoPINN = {pinn_loss_autograd:.6f}",end='')
            print(f", AutoAV = {abs_violation_autograd:.6f}")


            print(f"  📊 Test Loss = {avg_test_loss:.6f} | Data = {test_data_loss:.6f}", end='')
            # print(f", Data(Orig) = {test_data_loss_orig:.6f}", end='')
            # print(f", Consistency = {test_consistency_loss:.8f}", end='')
            print(f", Grad = {test_grad_loss:.6f}", end='')
            
            print(f", PINN = {test_pinn_loss:.6f}", end='')
            print(f", AV = {test_abs_violation:.6f}", end='')

            print(f", AutoPINN = {test_pinn_loss_autograd:.6f}", end='')
            print(f", AutoAV = {test_abs_violation_autograd:.6f}")
            
            print("  📐 Losses Before Projection: ")
            print(f" Train:     Data = {data_loss_before_projection:.6f}, PINN = {pinn_loss_before_projection:.6f}, AV = {abs_pinn_loss_before_projection:.6f}")
            print(f" Test:      Data = {test_data_loss_before_projection:.6f}, PINN = {test_pinn_loss_before_projection:.6f}, AV = {test_abs_pinn_loss_before_projection:.6f}")

    def _is_converged(self, avg_loss, avg_test_loss):
        return avg_loss < self.model_loss_tolerance and avg_test_loss < self.model_loss_tolerance

    def _save_losses(self):
        np.savez(self.losses_save_path,
                train_data_loss=np.array(self.train_data_losses),
                train_data_loss_orig=np.array(self.train_data_losses_orig),
                train_pinn_loss=np.array(self.train_pinn_losses),
                train_abs_violation = np.array(self.train_abs_violation),
                train_pinn_loss_autograd=np.array(self.train_pinn_losses_autograd),
                train_abs_violation_autograd = np.array(self.train_abs_violation_autograd),
                train_grad_loss=np.array(self.train_grad_losses),
                test_data_loss=np.array(self.test_data_losses),
                test_data_loss_orig=np.array(self.test_data_losses_orig),
                test_pinn_loss=np.array(self.test_pinn_losses),
                test_abs_violation = np.array(self.test_abs_violation),
                test_grad_loss=np.array(self.test_grad_losses),
                test_pinn_loss_autograd=np.array(self.test_pinn_losses_autograd),
                test_abs_violation_autograd = np.array(self.test_abs_violation_autograd),
                train_data_loss_before_projection=np.array(self.train_data_losses_before_projection),
                train_pinn_loss_before_projection=np.array(self.train_pinn_losses_before_projection),
                train_abs_pinn_loss_before_projection=np.array(self.train_abs_pinn_losses_before_projection),
                test_data_loss_before_projection=np.array(self.test_data_losses_before_projection),
                test_pinn_loss_before_projection=np.array(self.test_pinn_losses_before_projection),
                test_abs_pinn_loss_before_projection=np.array(self.test_abs_pinn_losses_before_projection),
                epoch_time=np.array(self.epoch_times),
                backbone_time=np.array(self.backbone_times),
                hat_gradient_time=np.array(self.hat_gradient_times),
                projection_time=np.array(self.projection_times),
                tilde_gradient_time=np.array(self.tilde_gradient_times),
                backprop_time=np.array(self.backprop_times),
                optimizer_step_time=np.array(self.optimizer_step_times)
                )

    def _save_model(self):
        torch.save(self.model.nn.state_dict(), self.model_save_path)

    def export_predictions(self, save_path):
        # Loading the best checkpoint if available
        if self.best_checkpoint_path is not None:
            print(f"Loading best model checkpoint from {self.best_checkpoint_path} with loss {self.best_checkpoint_loss:.6f}...")
            checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)

            # If the checkpoint contains model_state_dict, use it
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # This means the checkpoint is just the raw state dict
                self.model.load_state_dict(checkpoint)

        self.model.eval()
        all_dfs = []
    
        # -------------------------
        # 1) Train loader predictions
        # -------------------------
        x_all, y_all = [], []
        for x_batch, y_batch, *_ in self.train_loader:
            x_batch = x_batch.to(self.device).requires_grad_(True)
            y_batch = y_batch.to(self.device)

            # y_batch_orig = y_batch.clone()
            # if self.gaussian_noise_scale > 0:
            #     noise = torch.randn_like(y_batch) * self.gaussian_noise_var + self.gaussian_noise_mean
            #     y_batch = y_batch + self.gaussian_noise_scale * noise
    
            # Forward + derivative composition (same as training)
            y_hat = self.model.nn(x_batch)
    
            grad_outputs = []
            if self.required_derivatives:
                output_map = {f'y{i+1}': y_hat[:, i] for i in range(y_hat.shape[1])}
                for item in self.required_derivatives:
                    target_name = item['target']
                    order = int(item['order'])
                    wrt_list = item.get('wrt', [])
                    y_target = output_map[target_name]
    
                    # convert wrt variable names to indices: 'x1' -> 0, 'x2' -> 1, ...
                    has_t = 't' in wrt_list
                    wrt_indices = [0 if v == 't' else (int(v[1:]) - 1 + (1 if has_t else 0)) for v in wrt_list]
                        
                    if len(wrt_indices) == order:
                        seq = wrt_indices
                    elif len(wrt_indices) == 1:
                        seq = [wrt_indices[0]] * order
                    elif 1 < len(wrt_indices) < order:
                        seq = wrt_indices + [wrt_indices[-1]] * (order - len(wrt_indices))
                    else:
                        seq = [0] * order
    
                    grad = y_target
                    for var_idx in seq:
                        grads_wrt_x = torch.autograd.grad(
                            grad,
                            x_batch,
                            grad_outputs=torch.ones_like(grad),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True
                        )[0]
                        grad = grads_wrt_x[:, var_idx]
                    grad_outputs.append(grad)
    
            grad_outputs = [g.unsqueeze(1) if g.ndim == 1 else g for g in grad_outputs]
    
            y_head = y_hat[:, :y_batch.shape[-1]]
            y_tail = y_hat[:, y_batch.shape[-1]:]
            if grad_outputs:
                y_hat = torch.cat([y_head] + grad_outputs + [y_tail], dim=1)
    
            # x_batch_delta = x_batch + self.taylor_offset * torch.ones_like(x_batch, device=self.device)
            # # at delta distance
            # y_hat_delta = self.model.nn(x_batch_delta)
            # x_input = torch.cat([x_batch, y_hat[:, :y_batch.shape[-1]]] + grad_outputs, dim=1)
            # x_input = torch.cat([x_input, y_hat_delta[:, :y_batch.shape[-1]]], dim=1)
            
            y_dim = y_batch.shape[-1]
            y_hat_deltas = self._compute_per_param_deltas(x_batch)
            x_input = torch.cat([x_batch, y_hat[:, :self.num_ys]], dim=1)
            x_input = torch.cat([x_input, y_hat_deltas], dim=1)
            
            #   x_input = torch.cat([x_batch, y_hat[:, :y_batch.shape[-1]]] + grad_outputs, dim=1)
            y_tilde = self.model.newton(y_hat, x_input)
            y_taylor_recomputed = self.model.newton.taylor_fn(y_tilde, x_input)
            y_tilde[:, :self.num_ys] = y_taylor_recomputed
    
            # Violations
            eq_res = self.orig_eq_viol_fn(y_tilde, x_input)
            #   eq_res = self.eq_viol_fn(y_tilde, x_input)
            ineq_res = torch.clamp_min(self.orig_ineq_viol_fn(y_tilde, x_input), 0)
            combined = torch.cat([eq_res, ineq_res], dim=1)
            abs_violation = combined.abs().sum(dim=1, keepdim=True).detach().cpu().numpy()
    
            y_pred = y_tilde.detach().cpu().numpy()
            x_np = x_batch.detach().cpu().numpy()
            x_all.append(x_np)
            y_all.append(np.hstack([y_pred, abs_violation]))  # <-- include violation
    
        x_all = np.vstack(x_all)
        y_all = np.vstack(y_all)
    
        # Column Names
        columns_x = self.parameters_list
        columns_y = self.variables_list
        columns_all = columns_x + columns_y + ["abs_violation"]
    
        train_df = pd.DataFrame(np.hstack([x_all, y_all]), columns=columns_all)
        all_dfs.append(train_df)
    
        # -------------------------
        # 2) Test loaders predictions (one or more loaders)
        # -------------------------
        for i, loader in enumerate(self.test_loaders):
            x_all, y_all = [], []
            kkt_metadata = getattr(loader, "kkt_info", None)
    
            for x_batch, y_batch, *_ in loader:
                x_batch = x_batch.to(self.device).requires_grad_(True)
                y_batch = y_batch.to(self.device)
    
                y_hat = self.model.nn(x_batch)
    
                grad_outputs = []
                if self.required_derivatives:
                    output_map = {f'y{i+1}': y_hat[:, i] for i in range(y_hat.shape[1])}
                    for item in self.required_derivatives:
                        target_name = item['target']
                        order = int(item['order'])
                        wrt_list = item.get('wrt', [])
                        y_target = output_map[target_name]
    
                        # convert wrt variable names to indices: 'x1' -> 0, 'x2' -> 1, ...
                        has_t = 't' in wrt_list
                        wrt_indices = [0 if v == 't' else (int(v[1:]) - 1 + (1 if has_t else 0)) for v in wrt_list]
                        
                        if len(wrt_indices) == order:
                            seq = wrt_indices
                        elif len(wrt_indices) == 1:
                            seq = [wrt_indices[0]] * order
                        elif 1 < len(wrt_indices) < order:
                            seq = wrt_indices + [wrt_indices[-1]] * (order - len(wrt_indices))
                        else:
                            seq = [0] * order
    
                        grad = y_target
                        for var_idx in seq:
                            grads_wrt_x = torch.autograd.grad(
                                grad,
                                x_batch,
                                grad_outputs=torch.ones_like(grad),
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True
                            )[0]
                            grad = grads_wrt_x[:, var_idx]
                        grad_outputs.append(grad)
    
                grad_outputs = [g.unsqueeze(1) if g.ndim == 1 else g for g in grad_outputs]
    
                y_head = y_hat[:, :y_batch.shape[-1]]
                y_tail = y_hat[:, y_batch.shape[-1]:]
                if grad_outputs:
                    y_hat = torch.cat([y_head] + grad_outputs + [y_tail], dim=1)
                
                y_dim = y_batch.shape[-1]
                y_hat_deltas = self._compute_per_param_deltas(x_batch)
                x_input = torch.cat([x_batch, y_hat[:, :self.num_ys]], dim=1)
                x_input = torch.cat([x_input, y_hat_deltas], dim=1)
            
                y_tilde = self.model.newton(y_hat, x_input, is_test=False, kkt=kkt_metadata)
                y_taylor_recomputed = self.model.newton.taylor_fn(y_tilde, x_input)
                y_tilde[:, :self.num_ys] = y_taylor_recomputed

                x_input = torch.cat([x_batch, y_tilde[:, :self.num_ys]], dim=1)
                x_input = torch.cat([x_input, y_hat_deltas], dim=1)
    
                # Violations
                #   eq_res = self.eq_viol_fn(y_tilde, x_input)
                eq_res = self.orig_eq_viol_fn(y_tilde, x_input)
                ineq_res = torch.clamp_min(self.orig_ineq_viol_fn(y_tilde, x_input), 0)
                combined = torch.cat([eq_res, ineq_res], dim=1)
                abs_violation = combined.abs().sum(dim=1, keepdim=True).detach().cpu().numpy()
    
                #   y_pred = y_tilde[:, :y_batch.shape[-1]].detach().cpu().numpy()
                y_pred = y_tilde.detach().cpu().numpy()
                x_np = x_batch.detach().cpu().numpy()
    
                x_all.append(x_np)
                y_all.append(np.hstack([y_pred, abs_violation]))  # <-- MUST include violation here too
    
            x_all = np.vstack(x_all)
            y_all = np.vstack(y_all)
    
            # Column Names
            columns_x = self.parameters_list
            columns_y = self.variables_list
            columns_all = columns_x + columns_y + ["abs_violation"]
    
            test_df = pd.DataFrame(np.hstack([x_all, y_all]), columns=columns_all)
            all_dfs.append(test_df)
    
        # -------------------------
        # 3) Combine and save
        # -------------------------
        final_df = pd.concat(all_dfs, ignore_index=True)
        #   print("Length of saved file (rows):", len(final_df))
        final_df.to_csv(self.predictions_save_path, index=False)
        print(f"💾 Predictions saved at {self.predictions_save_path}")
        return checkpoint

    def export_predictions_for_analysis(self, save_path):
        self.model.eval()
        all_dfs = []
    
        # -------------------------
        # 1) Train loader predictions
        # -------------------------
        x_all, y_all = [], []
        for x_batch, y_batch, *_ in self.train_loader:
            x_batch = x_batch.to(self.device).requires_grad_(True)
            y_batch = y_batch.to(self.device)

            # y_batch_orig = y_batch.clone()
            # if self.gaussian_noise_scale > 0:
            #     noise = torch.randn_like(y_batch) * self.gaussian_noise_var + self.gaussian_noise_mean
            #     y_batch = y_batch + self.gaussian_noise_scale * noise
    
            # Forward + derivative composition (same as training)
            y_hat = self.model.nn(x_batch)
    
            grad_outputs = []
            if self.required_derivatives:
                output_map = {f'y{i+1}': y_hat[:, i] for i in range(y_hat.shape[1])}
                for item in self.required_derivatives:
                    target_name = item['target']
                    order = int(item['order'])
                    wrt_list = item.get('wrt', [])
                    y_target = output_map[target_name]
    
                    # convert wrt variable names to indices: 'x1' -> 0, 'x2' -> 1, ...
                    has_t = 't' in wrt_list
                    wrt_indices = [0 if v == 't' else (int(v[1:]) - 1 + (1 if has_t else 0)) for v in wrt_list]
                        
                    if len(wrt_indices) == order:
                        seq = wrt_indices
                    elif len(wrt_indices) == 1:
                        seq = [wrt_indices[0]] * order
                    elif 1 < len(wrt_indices) < order:
                        seq = wrt_indices + [wrt_indices[-1]] * (order - len(wrt_indices))
                    else:
                        seq = [0] * order
    
                    grad = y_target
                    for var_idx in seq:
                        grads_wrt_x = torch.autograd.grad(
                            grad,
                            x_batch,
                            grad_outputs=torch.ones_like(grad),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True
                        )[0]
                        grad = grads_wrt_x[:, var_idx]
                    grad_outputs.append(grad)
    
            grad_outputs = [g.unsqueeze(1) if g.ndim == 1 else g for g in grad_outputs]
    
            y_head = y_hat[:, :y_batch.shape[-1]]
            y_tail = y_hat[:, y_batch.shape[-1]:]
            if grad_outputs:
                y_hat = torch.cat([y_head] + grad_outputs + [y_tail], dim=1)
    
            # x_batch_delta = x_batch + self.taylor_offset * torch.ones_like(x_batch, device=self.device)
            # # at delta distance
            # y_hat_delta = self.model.nn(x_batch_delta)
            # x_input = torch.cat([x_batch, y_hat[:, :y_batch.shape[-1]]] + grad_outputs, dim=1)
            # x_input = torch.cat([x_input, y_hat_delta[:, :y_batch.shape[-1]]], dim=1)
            
            y_dim = y_batch.shape[-1]
            y_hat_deltas = self._compute_per_param_deltas(x_batch)
            x_input = torch.cat([x_batch, y_hat[:, :self.num_ys]], dim=1)
            x_input = torch.cat([x_input, y_hat_deltas], dim=1)
            
            #   x_input = torch.cat([x_batch, y_hat[:, :y_batch.shape[-1]]] + grad_outputs, dim=1)
            y_tilde = self.model.newton(y_hat, x_input)
            y_taylor_recomputed = self.model.newton.taylor_fn(y_tilde, x_input)
            y_tilde[:, :self.num_ys] = y_taylor_recomputed
    
            # Violations
            eq_res = self.orig_eq_viol_fn(y_tilde, x_input)
            #   eq_res = self.eq_viol_fn(y_tilde, x_input)
            ineq_res = torch.clamp_min(self.orig_ineq_viol_fn(y_tilde, x_input), 0)
            combined = torch.cat([eq_res, ineq_res], dim=1)
            abs_violation = combined.abs().sum(dim=1, keepdim=True).detach().cpu().numpy()
    
            y_pred = y_tilde.detach().cpu().numpy()
            x_np = x_batch.detach().cpu().numpy()
            x_all.append(x_np)
            y_all.append(np.hstack([y_pred, abs_violation]))  # <-- include violation
    
        x_all = np.vstack(x_all)
        y_all = np.vstack(y_all)
    
        # Column Names
        columns_x = self.parameters_list
        columns_y = self.variables_list
        columns_all = columns_x + columns_y + ["abs_violation"]
    
        train_df = pd.DataFrame(np.hstack([x_all, y_all]), columns=columns_all)
        all_dfs.append(train_df)
    
        # -------------------------
        # 2) Test loaders predictions (one or more loaders)
        # -------------------------
        for i, loader in enumerate(self.test_loaders):
            x_all, y_all = [], []
            kkt_metadata = getattr(loader, "kkt_info", None)
    
            for x_batch, y_batch, *_ in loader:
                x_batch = x_batch.to(self.device).requires_grad_(True)
                y_batch = y_batch.to(self.device)
    
                y_hat = self.model.nn(x_batch)
    
                grad_outputs = []
                if self.required_derivatives:
                    output_map = {f'y{i+1}': y_hat[:, i] for i in range(y_hat.shape[1])}
                    for item in self.required_derivatives:
                        target_name = item['target']
                        order = int(item['order'])
                        wrt_list = item.get('wrt', [])
                        y_target = output_map[target_name]
    
                        # convert wrt variable names to indices: 'x1' -> 0, 'x2' -> 1, ...
                        has_t = 't' in wrt_list
                        wrt_indices = [0 if v == 't' else (int(v[1:]) - 1 + (1 if has_t else 0)) for v in wrt_list]
                        
                        if len(wrt_indices) == order:
                            seq = wrt_indices
                        elif len(wrt_indices) == 1:
                            seq = [wrt_indices[0]] * order
                        elif 1 < len(wrt_indices) < order:
                            seq = wrt_indices + [wrt_indices[-1]] * (order - len(wrt_indices))
                        else:
                            seq = [0] * order
    
                        grad = y_target
                        for var_idx in seq:
                            grads_wrt_x = torch.autograd.grad(
                                grad,
                                x_batch,
                                grad_outputs=torch.ones_like(grad),
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True
                            )[0]
                            grad = grads_wrt_x[:, var_idx]
                        grad_outputs.append(grad)
    
                grad_outputs = [g.unsqueeze(1) if g.ndim == 1 else g for g in grad_outputs]
    
                y_head = y_hat[:, :y_batch.shape[-1]]
                y_tail = y_hat[:, y_batch.shape[-1]:]
                if grad_outputs:
                    y_hat = torch.cat([y_head] + grad_outputs + [y_tail], dim=1)
                
                y_dim = y_batch.shape[-1]
                y_hat_deltas = self._compute_per_param_deltas(x_batch)
                x_input = torch.cat([x_batch, y_hat[:, :self.num_ys]], dim=1)
                x_input = torch.cat([x_input, y_hat_deltas], dim=1)
            
                y_tilde = self.model.newton(y_hat, x_input, is_test=False, kkt=kkt_metadata)
                y_taylor_recomputed = self.model.newton.taylor_fn(y_tilde, x_input)
                y_tilde[:, :self.num_ys] = y_taylor_recomputed

                x_input = torch.cat([x_batch, y_tilde[:, :self.num_ys]], dim=1)
                x_input = torch.cat([x_input, y_hat_deltas], dim=1)
    
                # Violations
                #   eq_res = self.eq_viol_fn(y_tilde, x_input)
                eq_res = self.orig_eq_viol_fn(y_tilde, x_input)
                ineq_res = torch.clamp_min(self.orig_ineq_viol_fn(y_tilde, x_input), 0)
                combined = torch.cat([eq_res, ineq_res], dim=1)
                abs_violation = combined.abs().sum(dim=1, keepdim=True).detach().cpu().numpy()
    
                #   y_pred = y_tilde[:, :y_batch.shape[-1]].detach().cpu().numpy()
                y_pred = y_tilde.detach().cpu().numpy()
                x_np = x_batch.detach().cpu().numpy()
    
                x_all.append(x_np)
                y_all.append(np.hstack([y_pred, abs_violation]))  # <-- MUST include violation here too
    
            x_all = np.vstack(x_all)
            y_all = np.vstack(y_all)
    
            # Column Names
            columns_x = self.parameters_list
            columns_y = self.variables_list
            columns_all = columns_x + columns_y + ["abs_violation"]
    
            test_df = pd.DataFrame(np.hstack([x_all, y_all]), columns=columns_all)
            all_dfs.append(test_df)
    
        # -------------------------
        # 3) Combine and save
        # -------------------------
        final_df = pd.concat(all_dfs, ignore_index=True)
        #   print("Length of saved file (rows):", len(final_df))
        final_df.to_csv(save_path, index=False)
        print(f"💾 Predictions saved at {save_path}")
        # return checkpoint


    def train(self):
        for epoch in range(self.num_epochs):
            start_time = time.time()
            (avg_loss, data_loss, data_loss_orig, consistency_loss, grad_loss, 
            pinn_loss, abs_violation, pinn_loss_autograd, abs_violation_autograd,
            data_loss_before_projection, pinn_loss_before_projection, abs_pinn_loss_before_projection, 
            backbone_time, hat_gradient_time, projection_time, tilde_gradient_time, backprop_time, optimizer_step_time) = self.train_model()
            (avg_test_loss, test_data_loss, test_data_loss_orig, test_consistency_loss, test_grad_loss, 
            test_pinn_loss, test_abs_violation, test_pinn_loss_autograd, test_abs_violation_autograd,
            test_data_loss_before_projection, test_pinn_loss_before_projection, test_abs_pinn_loss_before_projection) = self.test_model()
            end_time = time.time()
            epoch_duration = end_time - start_time
            self.epoch_times.append(epoch_duration)

            # Store losses
            self.train_data_losses.append(data_loss)
            self.train_data_losses_orig.append(data_loss_orig)
            self.train_pinn_losses.append(pinn_loss)
            self.train_abs_violation.append(abs_violation)
            self.train_grad_losses.append(grad_loss)
            self.test_data_losses.append(test_data_loss)
            self.test_data_losses_orig.append(test_data_loss_orig)
            self.test_pinn_losses.append(test_pinn_loss)
            self.test_abs_violation.append(test_abs_violation)
            self.test_grad_losses.append(test_grad_loss)
            self.train_pinn_losses_autograd.append(pinn_loss_autograd)
            self.train_abs_violation_autograd.append(abs_violation_autograd)
            self.test_pinn_losses_autograd.append(test_pinn_loss_autograd)
            self.test_abs_violation_autograd.append(test_abs_violation_autograd)
            
            self.train_data_losses_before_projection.append(data_loss_before_projection)
            self.train_pinn_losses_before_projection.append(pinn_loss_before_projection)
            self.train_abs_pinn_losses_before_projection.append(abs_pinn_loss_before_projection)
            self.test_data_losses_before_projection.append(test_data_loss_before_projection)
            self.test_pinn_losses_before_projection.append(test_pinn_loss_before_projection)
            self.test_abs_pinn_losses_before_projection.append(test_abs_pinn_loss_before_projection)

            # Saving the times
            self.backbone_times.append(backbone_time)
            self.hat_gradient_times.append(hat_gradient_time)
            self.projection_times.append(projection_time)
            self.tilde_gradient_times.append(tilde_gradient_time)
            self.backprop_times.append(backprop_time)
            self.optimizer_step_times.append(optimizer_step_time)

            self.display_results(epoch, avg_loss, avg_test_loss,
                                data_loss, data_loss_orig, consistency_loss,  grad_loss, pinn_loss, abs_violation, pinn_loss_autograd, abs_violation_autograd,
                                test_data_loss, test_data_loss_orig, test_consistency_loss, test_grad_loss, test_pinn_loss, test_abs_violation, test_pinn_loss_autograd, test_abs_violation_autograd,
                                data_loss_before_projection, pinn_loss_before_projection, abs_pinn_loss_before_projection,
                                test_data_loss_before_projection, test_pinn_loss_before_projection, test_abs_pinn_loss_before_projection)
            
            # Save checkpoint every save_checkpoint_iter if loss improves
            if (epoch + 1) % self.save_checkpoint_iter == 0:
                # print(f"--- Saving checkpoint at epoch {epoch + 1} ---")
                self._save_checkpoint(epoch, data_loss, data_loss_orig, consistency_loss,  grad_loss, pinn_loss, abs_violation, pinn_loss_autograd, abs_violation_autograd,
                                test_data_loss, test_data_loss_orig, test_consistency_loss, test_grad_loss, test_pinn_loss, test_abs_violation, test_pinn_loss_autograd, test_abs_violation_autograd,
                                data_loss_before_projection, pinn_loss_before_projection, abs_pinn_loss_before_projection, test_data_loss_before_projection, test_pinn_loss_before_projection, test_abs_pinn_loss_before_projection)
            
            if not self.use_newton and data_loss < self.eta:
                print(f"🔁 Activating Newton loop at epoch {epoch + 1}, loss = {avg_loss:.8f}")
                self.use_newton = True

            if self._is_converged(avg_loss, avg_test_loss):
                print(f"✅ Training complete at epoch {epoch + 1}")
                break

            # --- Save predictions every 50 epochs ---
            if (epoch + 1) % 50 == 0:
                if self.best_checkpoint_path is not None:
                    checkpoint_pred = self.export_predictions_for_analysis(save_path=self.predictions_save_path.split(".csv")[0] + f"_epoch_{epoch+1}.csv")
                    # with open(self.mse_mape_save_path.split(".txt")[0] + f"_epoch_{epoch+1}.txt", "w") as f:
                    #     f.write("==================== Model Configuration ====================\n")
                    #     f.write(f"Epoch = {checkpoint_pred['epoch']}\n")
                    #     f.write(f"hidden_dim = {checkpoint_pred['hidden_dim']}\n")
                    #     f.write(f"model_depth = {checkpoint_pred['model_depth']}\n")
                    #     f.write("Newton Activated: {}\n".format("Yes" if self.use_newton else "No"))
                    #     f.write("==================== Train Metrices ====================\n")
                    #     f.write(f"mse train = {checkpoint_pred['train_data_loss']}\n")
                    #     f.write(f"mse train (grad) = {checkpoint_pred['train_grad_loss']}\n")
                    #     f.write(f"mse train (pinn) = {checkpoint_pred['train_pinn_loss']}\n")
                    #     f.write(f"mse train (pinn) (autograd) = {checkpoint_pred['train_pinn_loss_autograd']}\n")
                    #     f.write(f"absolute violation train = {checkpoint_pred['train_abs_violation']}\n")
                    #     f.write(f"absolute violation train (autograd) = {checkpoint_pred['train_abs_violation_autograd']}\n")
                    #     f.write("==================== Test Metrices ====================\n")
                    #     f.write(f"mse test = {checkpoint_pred['test_data_loss']}\n")
                    #     f.write(f"mse test (grad) = {checkpoint_pred['test_grad_loss']}\n")
                    #     f.write(f"mse test (pinn) = {checkpoint_pred['test_pinn_loss']}\n")
                    #     f.write(f"mse test (pinn) (autograd) = {checkpoint_pred['test_pinn_loss_autograd']}\n")
                    #     f.write(f"absolute violation test = {checkpoint_pred['test_abs_violation']}\n")
                    #     f.write(f"absolute violation test (autograd) = {checkpoint_pred['test_abs_violation_autograd']}\n")
            
        self._save_model()
        self._save_losses()
        checkpoint = self.export_predictions(save_path=self.predictions_save_path)

        with open(self.mse_mape_save_path, "w") as f:
            f.write("==================== Model Configuration ====================\n")
            f.write(f"Epoch = {checkpoint['epoch']}\n")
            f.write(f"hidden_dim = {checkpoint['hidden_dim']}\n")
            f.write(f"model_depth = {checkpoint['model_depth']}\n")
            f.write("==================== Train Metrices ====================\n")
            f.write(f"mse train = {checkpoint['train_data_loss']}\n")
            f.write(f"mse train (grad) = {checkpoint['train_grad_loss']}\n")
            f.write(f"mse train (pinn) = {checkpoint['train_pinn_loss']}\n")
            f.write(f"mse train (pinn) (autograd) = {checkpoint['train_pinn_loss_autograd']}\n")
            f.write(f"absolute violation train = {checkpoint['train_abs_violation']}\n")
            f.write(f"absolute violation train (autograd) = {checkpoint['train_abs_violation_autograd']}\n")
            f.write("==================== Test Metrices ====================\n")
            f.write(f"mse test = {checkpoint['test_data_loss']}\n")
            f.write(f"mse test (grad) = {checkpoint['test_grad_loss']}\n")
            f.write(f"mse test (pinn) = {checkpoint['test_pinn_loss']}\n")
            f.write(f"mse test (pinn) (autograd) = {checkpoint['test_pinn_loss_autograd']}\n")
            f.write(f"absolute violation test = {checkpoint['test_abs_violation']}\n")
            f.write(f"absolute violation test (autograd) = {checkpoint['test_abs_violation_autograd']}\n")

            # --- Read and append JSON file content ---
            json_path = self.model_config_path  # <--- path to your JSON file
            f.write("\n==================== JSON Configuration =====================\n")
            try:
                with open(json_path, "r") as jf:
                    json_data = json.load(jf)
                # Pretty print JSON as key-value pairs
                json_text = json.dumps(json_data, indent=4)
                f.write(json_text + "\n")
            except Exception as e:
                f.write(f"Error reading JSON file: {e}\n")

            print(f"💾 MSE and violation saved at file {self.mse_mape_save_path}")


