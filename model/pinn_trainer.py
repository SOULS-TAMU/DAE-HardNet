import os
import torch
import re
import time
import numpy as np
import pandas as pd
import json


class PINN_Trainer:
    def __init__(self, config_dir, model, train_loader, val_loader, test_loader, optimizer, criterion,
                 parameters_list, variables_list,
                 pinn_reg_factor=1, num_epochs=500, eta=1e-3, model_loss_tolerance=1e-4, save_checkpoint_iter=50, 
                 taylor_offset=1e-3, gaussian_mean=0.0, gaussian_var=0.0, gaussian_scale=0.0, checkpoint_path=None, device=None, is_linear=False):
        self.config_dir = config_dir
        self.model_config_path = f"{self.config_dir}/model_config.json"
        self.model = model
        self.num_ys = len([v for v in variables_list if str(v).startswith("y")])
        self.sym_names = model.newton.sym_names
        self.res_fn = model.newton.res_fn
        self.eq_viol_fn = model.newton._evaluate_eq_res
        self.ineq_viol_fn = model.newton._evaluate_ineq_res
        self.orig_eq_viol_fn = model.newton._evaluate_orig_eq_res
        self.orig_ineq_viol_fn = model.newton._evaluate_orig_ineq_res
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.eta = eta
        self.taylor_offset = taylor_offset
        self.pinn_reg_factor = pinn_reg_factor
        self.gaussian_noise_mean = gaussian_mean
        self.gaussian_noise_var = gaussian_var
        self.gaussian_noise_scale = gaussian_scale
        self.model_loss_tolerance = model_loss_tolerance
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.use_newton = False
        self.best_checkpoint_path = None
        self.losses_save_path = f"{self.config_dir}/pinn_losses.npz"
        self.model_save_path = f"{self.config_dir}/pinn_model.pth"
        self.predictions_save_path = f"{self.config_dir}/pinn_predictions.csv"
        self.mse_mape_save_path = f"{self.config_dir}/pinn_metrics.txt"

        self.train_data_losses = []
        self.train_data_losses_orig = []
        self.test_data_losses = []
        self.test_data_losses_orig = []
        self.train_pinn_losses = []
        self.test_pinn_losses = []
        self.train_abs_violation = []
        self.test_abs_violation = []
        self.epoch_times = []

        # Saving the time taken each step
        self.backbone_times = []
        self.hat_gradient_times = []
        self.backprop_times = []
        self.optimizer_step_times = []
        
        self.parameters_list = parameters_list
        self.variables_list = variables_list
        
        # ===============================
        # Differential Term Detection
        # ===============================
        self.has_differential_terms = False
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
        self.save_checkpoint_iter = save_checkpoint_iter
        self.best_checkpoint_loss = float('inf')

        # If checkpoint path provided, load weights before training
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

            print(f"✅ Loaded model weights from {checkpoint_path}")
        else:
            print(f"⚠️ No checkpoint file found. Starting training from scratch")

        

        # Make checkpoints directory
        self.checkpoint_dir = os.path.join(self.config_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _save_checkpoint(self, epoch, avg_loss, avg_test_loss, 
                         data_loss, data_loss_orig, consistency_loss, pinn_loss, abs_violation,
                         test_data_loss, test_data_loss_orig, test_consistency_loss, test_pinn_loss, test_abs_violation):
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
                'train_pinn_loss': pinn_loss,
                'train_abs_violation': abs_violation,
                'test_data_loss': test_data_loss,
                'test_data_loss_orig': test_data_loss_orig,
                'test_pinn_loss': test_pinn_loss,
                'test_abs_violation': test_abs_violation,
            }, checkpoint_path)

            self.best_checkpoint_loss = loss
            # print(f"💾 Checkpoint saved at epoch {epoch+1} with loss {loss:.8f}")
            self.best_checkpoint_path = checkpoint_path
        else:
            # print(f"⚠️ No checkpoint saved at epoch {epoch+1} (loss {loss:.8f} >= best {self.best_checkpoint_loss:.8f})")
            pass


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
        total_consistency_loss = 0
        total_pinn_loss = 0
        total_abs_pinn_loss = 0

        backbone_time = 0
        hat_gradient_time = 0
        backprop_time = 0
        optimizer_step_time = 0

        for x_batch, y_batch, y_data_batch in self.train_loader:
            x_batch, y_batch, y_data_batch = (
                x_batch.to(self.device),
                y_batch.to(self.device),
                y_data_batch.to(self.device)
            )

            y_batch_orig = y_batch.clone()

            if self.gaussian_noise_scale > 0:
                noise = torch.randn_like(y_batch) * self.gaussian_noise_var + self.gaussian_noise_mean
                y_batch = y_batch + self.gaussian_noise_scale * noise
                # print("Added noise with scale:", self.gaussian_noise_scale)

            x_batch.requires_grad_(True)  # Enable autograd for derivative computation
            self.optimizer.zero_grad()

            # Step 1: Predict base NN outputs (y1, y2, ...)
            start_time = time.time()
            y_hat_base = self.model.nn(x_batch)  # shape: (B, num_outputs)
            end_time = time.time()
            backbone_time += end_time - start_time

            grad_outputs = []
            start_time = time.time()
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

            # Step 4: Build input for res_fn (x + original NN outputs)
            y_hat_deltas = self._compute_per_param_deltas(x_batch)  # (B, y_dim * n_params)
            x_input = torch.cat([x_batch, y_hat[:, :self.num_ys]], dim=1)
            x_input = torch.cat([x_input, y_hat_deltas], dim=1)

            # if not self.use_newton:
            data_loss = self.criterion(y_hat[:, :y_batch.shape[-1]], y_batch)
            data_loss_orig = self.criterion(y_hat[:, :y_batch.shape[-1]], y_batch_orig)
            consistency_loss = torch.tensor(0.0, device=self.device)

            # print("PINN Train y_hat shape: ", y_hat.shape)
            # print("PINN Test x_input shape: ", x_input.shape)

            eq_res = self.orig_eq_viol_fn(y_hat, x_input)
            ineq_res = self.orig_ineq_viol_fn(y_hat, x_input)

            ineq_res = torch.clamp_min(ineq_res,0)
            combined = torch.cat([eq_res, ineq_res], dim=1)
            abs_pinn_loss = torch.mean(combined.abs().sum(dim=1))
            pinn_loss = torch.linalg.norm(combined, dim=1).mean()
            
            loss = data_loss + self.pinn_reg_factor * pinn_loss
            # loss = self.pinn_reg_factor * pinn_loss

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
            total_consistency_loss += consistency_loss.item() * batch_size
            total_pinn_loss += pinn_loss.item() * batch_size
            total_abs_pinn_loss += abs_pinn_loss.item() * batch_size

        n_samples = len(self.train_loader.dataset)
        return (
            total_loss / n_samples,
            total_data_loss / n_samples,
            total_data_loss_orig / n_samples,
            total_consistency_loss / n_samples,
            total_pinn_loss / n_samples,
            total_abs_pinn_loss /n_samples,
            backbone_time,
            hat_gradient_time,
            backprop_time,
            optimizer_step_time
        )

    def test_model(self):
        self.model.eval()
        total_loss = 0
        total_data_loss = 0
        total_data_loss_orig = 0
        total_consistency_loss = 0
        total_pinn_loss = 0
        total_abs_pinn_loss = 0

        for x_test_batch, y_test_batch, y_data_test_batch in self.test_loader:
            x_test_batch = x_test_batch.to(self.device).requires_grad_(True)
            y_test_batch = y_test_batch.to(self.device)
            y_data_test_batch = y_data_test_batch.to(self.device)

            # Step 1: Base NN predictions
            y_hat_base = self.model.nn(x_test_batch)  # shape: (B, num_outputs)

            grad_outputs = []
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
                            x_test_batch,
                            grad_outputs=torch.ones_like(grad),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True
                        )[0]  # shape [B, input_dim]

                        # select the column corresponding to var_idx
                        grad = grads_wrt_x[:, var_idx]  # shape [B]

                    # now 'grad' is the desired derivative array shape [B]
                    grad_outputs.append(grad)   # collect for later concatenation/use

            # Ensure all grad_outputs are 2D column tensors [B, 1]
            grad_outputs = [g.unsqueeze(1) if g.ndim == 1 else g for g in grad_outputs]

            # Step 3: Concatenate all outputs in the order of sym_names
            # y_head = y_hat_base[:, :y_test_batch.shape[-1]]  # original y vars
            # y_tail = y_hat_base[:, y_test_batch.shape[-1]:]  # remaining vars after y
            y_head = y_hat_base[:, :self.num_ys]  # original y vars
            y_tail = y_hat_base[:, self.num_ys:]  # remaining vars after y

            if grad_outputs:
                y_hat = torch.cat([y_head] + grad_outputs + [y_tail], dim=1)
            else:
                y_hat = y_hat_base

            # Step 4: Prepare residual input
            y_hat_deltas = self._compute_per_param_deltas(x_test_batch)  # (B, y_dim * n_params)
            x_input = torch.cat([x_test_batch, y_hat[:, :self.num_ys]], dim=1)
            x_input = torch.cat([x_input, y_hat_deltas], dim=1)
            
            # print("PINN Test y_hat shape: ", y_hat.shape)
            # print("PINN Test x_input shape: ", x_input.shape)
            
            eq_res = self.orig_eq_viol_fn(y_hat, x_input)
            ineq_res = self.orig_ineq_viol_fn(y_hat, x_input)

            ineq_res = torch.clamp_min(ineq_res,0)
            combined = torch.cat([eq_res, ineq_res], dim=1)
            abs_pinn_loss = torch.mean(combined.abs().sum(dim=1))
            pinn_loss = torch.linalg.norm(combined, dim=1).mean()
            
            data_loss = self.criterion(y_hat[:, :y_test_batch.shape[-1]], y_test_batch)
            data_loss_orig = self.criterion(y_hat[:, :y_test_batch.shape[-1]], y_test_batch)
            consistency_loss = torch.tensor(0.0, device=self.device)
            loss = data_loss + self.pinn_reg_factor * pinn_loss
            # loss = self.pinn_reg_factor * pinn_loss

            batch_size = x_test_batch.size(0)
            total_loss += loss.item() * batch_size
            total_data_loss += data_loss.item() * batch_size
            total_data_loss_orig += data_loss_orig.item() * batch_size
            total_consistency_loss += consistency_loss.item() * batch_size
            total_pinn_loss += pinn_loss.item() * batch_size
            total_abs_pinn_loss += abs_pinn_loss.item() * batch_size

        n_samples = len(self.test_loader.dataset)
        return (
            total_loss / n_samples,
            total_data_loss / n_samples,
            total_data_loss_orig / n_samples,
            total_consistency_loss / n_samples,
            total_pinn_loss / n_samples,
            total_abs_pinn_loss /n_samples
        )

    def display_results(self, epoch, avg_loss, avg_test_loss,
                        data_loss, data_loss_orig, consistency_loss, pinn_loss, abs_violation,
                        test_data_loss, test_data_loss_orig, test_consistency_loss, test_pinn_loss, test_abs_violation):
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"[Epoch {epoch + 1}]")
            print(f"  🔧 Train Loss = {avg_loss:.6f} | Data = {data_loss:.6f}", end='')
            print(f" (Orig = {data_loss_orig:.6f})", end='')
            # if self.use_newton:
            # print(f", Consistency = {consistency_loss:.8f}", end='')
            print(f", PINN = {pinn_loss:.6f}",end='')
            print(f", AV = {abs_violation:.6f}")

            print(f"  📊 Test Loss = {avg_test_loss:.6f} | Data = {test_data_loss:.6f}", end='')
            print(f" (Orig = {test_data_loss_orig:.6f})", end='')
            # if self.use_newton:
            #     print(f", Consistency = {test_consistency_loss:.8f}", end='')
            print(f", PINN = {test_pinn_loss:.6f}", end='')
            print(f", AV = {test_abs_violation:.6f}")

    def _is_converged(self, avg_loss, avg_test_loss):
        return avg_loss < self.model_loss_tolerance and avg_test_loss < self.model_loss_tolerance

    def _save_losses(self):
        np.savez(self.losses_save_path,
                 train_data_loss=np.array(self.train_data_losses),
                 train_data_loss_orig=np.array(self.train_data_losses_orig),
                 train_pinn_loss=np.array(self.train_pinn_losses),
                 train_abs_violation = np.array(self.train_abs_violation),
                 test_data_loss=np.array(self.test_data_losses),
                 test_data_loss_orig=np.array(self.test_data_losses_orig),
                 test_pinn_loss=np.array(self.test_pinn_losses),
                 test_abs_violation = np.array(self.test_abs_violation),
                 epoch_time=np.array(self.epoch_times),
                 backbone_time=np.array(self.backbone_times),
                 hat_gradient_time=np.array(self.hat_gradient_times),
                 backprop_time=np.array(self.backprop_times),
                 optimizer_step_time=np.array(self.optimizer_step_times)
                 )

    def _save_model(self):
        torch.save(self.model.nn.state_dict(), self.model_save_path)

    def export_predictions(self, save_path):
        """
        Runs the model on training and testing sets,
        extracts y_tilde[:, :y_batch.shape[-1]], 
        and saves inputs + outputs in a single CSV.
        """
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
        x_all, y_all = [], []

        with torch.enable_grad():
            for loader in [self.train_loader, self.test_loader]:
                for x_batch, y_batch, *_ in loader:
                    x_batch = x_batch.to(self.device).requires_grad_(True)
                    y_batch = y_batch.to(self.device)

                    # Forward pass (adjust this part if your forward returns differently)
                    y_hat_base = self.model.nn(x_batch)

                    grad_outputs = []
                    if self.required_derivatives:
                        # map output names to columns
                        output_map = {f'y{i + 1}': y_hat_base[:, i] for i in range(y_hat_base.shape[1])}

                        for item in self.required_derivatives:
                            target_name = item['target']  # e.g., 'y1'
                            order = int(item['order'])  # integer order
                            wrt_list = item.get('wrt', [])  # e.g., ['x1'] or ['x1','x2']
                            y_target = output_map[target_name]  # shape [B]

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
                            grad = y_target  # shape [B]
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
                            grad_outputs.append(grad)  # collect for later concatenation/use

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

                    y_hat_deltas = self._compute_per_param_deltas(x_batch)  # (B, y_dim * n_params)
                    x_input = torch.cat([x_batch, y_hat[:, :self.num_ys]], dim=1)
                    x_input = torch.cat([x_input, y_hat_deltas], dim=1)

                    # Select only the first part of y_tilde
                    # y_pred = y_hat[:, :y_batch.shape[-1]].detach().numpy()
                    y_pred = y_hat.detach().numpy()

                    ############## Adding the Violation term ############################
                    # After computing y_tilde

                    eq_res = self.orig_eq_viol_fn(y_hat, x_input)
                    ineq_res = self.orig_ineq_viol_fn(y_hat, x_input)

                    ineq_res = torch.clamp_min(ineq_res, 0)
                    combined = torch.cat([eq_res, ineq_res], dim=1)

                    # Absolute violation
                    abs_violation = combined.abs().sum(dim=1, keepdim=True).detach().numpy()
                    #####################################################################

                    x_np = x_batch.detach().numpy()

                    x_all.append(x_np)
                    y_all.append(np.hstack([y_pred, abs_violation]))

        # Combine all data
        x_all = np.vstack(x_all)
        y_all = np.vstack(y_all)
        
        # Column Names
        columns_x = self.parameters_list
        columns_y = self.variables_list
        columns_all = columns_x + columns_y + ["abs_violation"]

        # Create DataFrame with appropriate column names
        df = pd.DataFrame(
            np.hstack([x_all, y_all]),
            columns=columns_all
        )

        # Save to CSV
        print("Length of saved file: ", len(df))
        df.to_csv(save_path, index=False)
        print(f"💾 Predictions saved at file {save_path}")
        return checkpoint
    
    def export_predictions_for_analysis(self, save_path):
        """
        Runs the model on training and testing sets,
        extracts y_tilde[:, :y_batch.shape[-1]], 
        and saves inputs + outputs in a single CSV.
        """
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
        x_all, y_all = [], []

        with torch.enable_grad():
            for loader in [self.train_loader, self.test_loader]:
                for x_batch, y_batch, *_ in loader:
                    x_batch = x_batch.to(self.device).requires_grad_(True)
                    y_batch = y_batch.to(self.device)

                    # Forward pass (adjust this part if your forward returns differently)
                    y_hat_base = self.model.nn(x_batch)

                    grad_outputs = []
                    if self.required_derivatives:
                        # map output names to columns
                        output_map = {f'y{i + 1}': y_hat_base[:, i] for i in range(y_hat_base.shape[1])}

                        for item in self.required_derivatives:
                            target_name = item['target']  # e.g., 'y1'
                            order = int(item['order'])  # integer order
                            wrt_list = item.get('wrt', [])  # e.g., ['x1'] or ['x1','x2']
                            y_target = output_map[target_name]  # shape [B]

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
                            grad = y_target  # shape [B]
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
                            grad_outputs.append(grad)  # collect for later concatenation/use

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

                    y_hat_deltas = self._compute_per_param_deltas(x_batch)  # (B, y_dim * n_params)
                    x_input = torch.cat([x_batch, y_hat[:, :self.num_ys]], dim=1)
                    x_input = torch.cat([x_input, y_hat_deltas], dim=1)

                    # Select only the first part of y_tilde
                    # y_pred = y_hat[:, :y_batch.shape[-1]].detach().numpy()
                    y_pred = y_hat.detach().numpy()

                    ############## Adding the Violation term ############################
                    # After computing y_tilde

                    eq_res = self.orig_eq_viol_fn(y_hat, x_input)
                    ineq_res = self.orig_ineq_viol_fn(y_hat, x_input)

                    ineq_res = torch.clamp_min(ineq_res, 0)
                    combined = torch.cat([eq_res, ineq_res], dim=1)

                    # Absolute violation
                    abs_violation = combined.abs().sum(dim=1, keepdim=True).detach().numpy()
                    #####################################################################

                    x_np = x_batch.detach().numpy()

                    x_all.append(x_np)
                    y_all.append(np.hstack([y_pred, abs_violation]))

        # Combine all data
        x_all = np.vstack(x_all)
        y_all = np.vstack(y_all)
        
        # Column Names
        columns_x = self.parameters_list
        columns_y = self.variables_list
        columns_all = columns_x + columns_y + ["abs_violation"]

        # Create DataFrame with appropriate column names
        df = pd.DataFrame(
            np.hstack([x_all, y_all]),
            columns=columns_all
        )

        # Save to CSV
        print("Length of saved file: ", len(df))
        df.to_csv(save_path, index=False)
        print(f"💾 Predictions saved at file {save_path}")
        # return checkpoint

    def modify_reg_factor(self, epoch):
        # Example: Increase pinn_reg_factor every 500 epochs
        if (epoch + 1) % 400 == 0:
            self.pinn_reg_factor *= 1e2
            print(f"🔧 Updated PINN regularization factor to {self.pinn_reg_factor}")

    def train(self):
        # Create lists to store the losses

        for epoch in range(self.num_epochs):
            start_time = time.time()
            (avg_loss, data_loss, data_loss_orig, consistency_loss, pinn_loss, abs_violation,
             backbone_time, hat_gradient_time, backprop_time, optimizer_step_time) = self.train_model()
            (avg_test_loss, test_data_loss, test_data_loss_orig, test_consistency_loss, test_pinn_loss, test_abs_violation) = self.test_model()
            end_time = time.time()
            epoch_duration = end_time - start_time
            self.epoch_times.append(epoch_duration)

            # Store losses
            self.train_data_losses.append(data_loss)
            self.train_data_losses_orig.append(data_loss_orig)
            self.train_pinn_losses.append(pinn_loss)
            self.train_abs_violation.append(abs_violation)
            self.test_data_losses.append(test_data_loss)
            self.test_data_losses_orig.append(test_data_loss_orig)
            self.test_pinn_losses.append(test_pinn_loss)
            self.test_abs_violation.append(test_abs_violation)

            self.backbone_times.append(backbone_time)
            self.hat_gradient_times.append(hat_gradient_time)
            self.backprop_times.append(backprop_time)
            self.optimizer_step_times.append(optimizer_step_time)

            # self.modify_reg_factor(epoch)

            self.display_results(epoch, avg_loss, avg_test_loss,
                                 data_loss, data_loss_orig, consistency_loss, pinn_loss, abs_violation,
                                 test_data_loss, test_data_loss_orig, test_consistency_loss, test_pinn_loss, test_abs_violation)
            
            # Save checkpoint every save_checkpoint_iter if loss improves
            if (epoch + 1) % self.save_checkpoint_iter == 0:
                self._save_checkpoint(epoch, avg_loss, avg_test_loss,
                                 data_loss, data_loss_orig, consistency_loss, pinn_loss, abs_violation,
                                 test_data_loss, test_data_loss_orig, test_consistency_loss, test_pinn_loss, test_abs_violation)

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
            f.write(f"mse train (orig) = {checkpoint['train_data_loss_orig']}\n")
            f.write(f"mse train (pinn) = {checkpoint['train_pinn_loss']}\n")
            f.write(f"mse train absolute violation = {checkpoint['train_abs_violation']}\n")

            f.write("\n==================== Test Metrices =====================\n")
            f.write(f"mse test = {checkpoint['test_data_loss']}\n")
            f.write(f"mse test (orig) = {checkpoint['test_data_loss_orig']}\n")
            f.write(f"mse test (pinn) = {checkpoint['test_pinn_loss']}\n")
            f.write(f"mse test absolute violation = {checkpoint['test_abs_violation']}\n")

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

