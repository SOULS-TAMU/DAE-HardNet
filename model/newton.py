import torch
import re
import sympy as sp
from torch import nn
from run.utils import categorize
import pprint
import time

# def categorize(sym):
#     name = str(sym)

#     # Pure y variables like y1, y2
#     if re.fullmatch(r"y\d+", name):
#         return (0, 0, name)
    
#     elif name.startswith("y") and not name.endswith("data"):
#         return (1, 0, name)
    
#     elif name.startswith("d") and not name.endswith("data"):
#         return (2, 0, name)

#     # Other y-prefixed variables
#     elif name.startswith("y") and not name.endswith("data"):
#         return (3, 0, name)

#     elif name.startswith("mu"):
#         return (4, 0, name)
#     elif name.startswith("s"):
#         return (5, 0, name)
#     elif name.startswith("delta"):
#         return (6, 0, name)
#     elif name.startswith("sigma"):
#         return (7, 0, name)
#     elif name.startswith("lambda"):
#         return (8, 0, name)
#     elif name.startswith("t"):
#         return (9, 0, name)
#     elif name.startswith("x"):
#         return (10, 0, name)
#     elif name.startswith("y") and name.endswith("data"):
#         return (11, 0, name)
#     elif name.startswith("d") and name.endswith("data"):
#         return (12, 0, name)
#     elif name.startswith("y") and name.endswith("delta"):
#         return (13, 0, name)
#     else:
#         return (14, 0, name)


class NewtonLayer(nn.Module):
    def __init__(self, residuals, taylor_exprs, orig_eq_violation, orig_ineq_violation, eq_violation, ineq_violation, variables, parameters,
                 taylor_offset, step_length=0.1, tol=1e-12, reg_factor=1e-6, max_iter=100):
        super(NewtonLayer, self).__init__()
        # preserve initial training context
        y_vars = [v for v in variables if str(v).startswith("y")]
        self.variables = variables
        self.parameters = parameters
        
        # print(f"Parameters: {self.parameters}, type: {type(self.parameters)}")
        # print(f"Variables: {self.variables}, type: {type(self.variables)}")
        self.res_exprs = residuals
        self.taylor_exprs = taylor_exprs

        # print(residuals)
        self.J_exprs = residuals.jacobian(variables[len(y_vars):])
        # pprint.pprint(self.J_exprs)

        self.orig_eq_viol_exprs = orig_eq_violation
        self.orig_ineq_viol_exprs = orig_ineq_violation
        self.eq_viol_exprs = eq_violation
        self.ineq_viol_exprs = ineq_violation
        self.step_size = step_length
        self.tol = tol
        self.taylor_offset = taylor_offset
        self.reg_factor = reg_factor
        self.max_iter = max_iter

        # Combine all symbols for training-time functions
        self.all_syms = sorted(variables + parameters, key=categorize)
        self.sym_names = [str(s) for s in self.all_syms]
        
        # print(f"all symbols: {self.all_syms}")
        # print(f"symbol names: {self.sym_names}")
        
        self.y_syms = [s for s in self.all_syms if re.fullmatch(r"y\d+", str(s))]
        self.n_y_syms = len(self.y_syms)

        # Sample batch (64 examples)
        # y_sample = torch.randn(64, len(self.variables))
        # x_sample = torch.randn(64, len(self.parameters))

        # Build differentiable torch-compatible functions for training context
        # print("Residuals:", self.res_exprs)
        # print("variables:", self.variables, "parameters:", self.parameters)
        
        self.res_fn = self._make_torch_fn(self.res_exprs, is_matrix=False,
                                        var_list=self.variables, param_list=self.parameters)
        # print("Jacobian:", self.J_exprs)
        self.jac_fn = self._make_torch_fn(self.J_exprs.tolist(), is_matrix=True,
                                        var_list=self.variables, param_list=self.parameters)
        # print("Taylor expressions:", self.taylor_exprs)
        # print("type of taylor_exprs:", type(self.taylor_exprs))
        self.taylor_fn = self._make_torch_fn(self.taylor_exprs, is_matrix=False,
                                        var_list=self.variables, param_list=self.parameters)
        

        # print("Residuals:", self.res_exprs)
        # res_out = self.res_fn(y_sample, x_sample)
        # print("Residual output shape:", res_out.shape)  # expect (64, n_res)
        # print("Residual sample:", res_out[0])

        # # Jacobian
        # print("Jacobian:", self.J_exprs)
        # jac_out = self.jac_fn(y_sample, x_sample)
        # print("Jacobian output shape:", jac_out.shape)  # expect (64, n_rows, n_cols)
        # print("Jacobian sample (first row of first batch):", jac_out[0,0])

        # # Taylor
        # print("Taylor expressions:", self.taylor_exprs)
        # taylor_out = self.taylor_fn(y_sample, x_sample)
        # print("Taylor output shape:", taylor_out.shape)  # expect (64, n_taylor_exprs)
        # print("Taylor sample:", taylor_out[0])

        # print(self.variables)
        # print(self.parameters)

        if len(self.eq_viol_exprs):
            # print("Eq constraints:", self.eq_viol_exprs)
            # print("Making eq_viol_fn with variables:", self.variables)
            self.eq_viol_fn = self._make_torch_fn(self.eq_viol_exprs, is_matrix=False,
                                                var_list=self.variables, param_list=self.parameters)
        else:
            self.eq_viol_fn = None

        if len(self.orig_eq_viol_exprs):
            # print("Eq constraints:", self.eq_viol_exprs)
            # print("Making eq_viol_fn with variables:", self.variables)
            self.orig_eq_viol_fn = self._make_torch_fn(self.orig_eq_viol_exprs, is_matrix=False,
                                                var_list=self.variables, param_list=self.parameters)
        else:
            self.orig_eq_viol_fn = None

        if len(self.ineq_viol_exprs):
            # print("In Eq constraints:", self.ineq_viol_exprs)
            self.ineq_viol_fn = self._make_torch_fn(self.ineq_viol_exprs, is_matrix=False,
                                                    var_list=self.variables, param_list=self.parameters)
        else:
            self.ineq_viol_fn = None

        if len(self.orig_ineq_viol_exprs):
            # print("Eq constraints:", self.eq_viol_exprs)
            # print("Making eq_viol_fn with variables:", self.variables)
            self.orig_ineq_viol_fn = self._make_torch_fn(self.orig_ineq_viol_exprs, is_matrix=False,
                                                var_list=self.variables, param_list=self.parameters)
        else:
            self.orig_ineq_viol_fn = None


        # Cache: key -> (res_fn, jac_fn, variables_list, n_y_syms, optional_kkt_system)
        self.kkt_cache = {}
        self.is_concatenated_during_test = False

    def _evaluate_eq_res(self, y_data, x_data):
        bsz, device, dtype = y_data.shape[0], y_data.device, y_data.dtype
        if self.eq_viol_fn is None:
            return torch.zeros(bsz, 0, device=device, dtype=dtype)
        else:
            return self.eq_viol_fn(y_data, x_data)

    def _evaluate_ineq_res(self, y_data, x_data):
        bsz, device, dtype = y_data.shape[0], y_data.device, y_data.dtype
        if self.ineq_viol_fn is None:
            return torch.zeros(bsz, 0, device=device, dtype=dtype)
        else:
            return self.ineq_viol_fn(y_data, x_data)
        
    def _evaluate_orig_eq_res(self, y_data, x_data):
        bsz, device, dtype = y_data.shape[0], y_data.device, y_data.dtype
        if self.eq_viol_fn is None:
            return torch.zeros(bsz, 0, device=device, dtype=dtype)
        else:
            return self.orig_eq_viol_fn(y_data, x_data)

    def _evaluate_orig_ineq_res(self, y_data, x_data):
        bsz, device, dtype = y_data.shape[0], y_data.device, y_data.dtype
        if self.ineq_viol_fn is None:
            return torch.zeros(bsz, 0, device=device, dtype=dtype)
        else:
            return self.orig_ineq_viol_fn(y_data, x_data)

    # def _make_torch_fn(self, sym_exprs, is_matrix=False, var_list=None, param_list=None):
    #     """
    #     Create a torch-callable function that accepts (y, x) where
    #     - y: (B, n_vars_for_this_fn)
    #     - x: (B, n_params)
    #     var_list and param_list define the correct symbol order used for lambdify.
    #     """
    #     # fallback to class defaults if not provided
    #     vars_ = var_list if var_list is not None else self.variables
    #     params_ = param_list if param_list is not None else self.parameters
        
    #     # print(vars_)
    #     # print(params_)
        

    #     all_syms_ = sorted(list(vars_) + list(params_), key=categorize)
    #     # print("\n=== Torch Function Creation ===")
    #     # print("Raw Taylor expressions:")
    #     for expr in sym_exprs:
    #         print(" ", expr)

    #     # print("Variable symbols (y):", vars_)
    #     # print("Parameter symbols (x):", params_)
        

    #     sym_names_ = [str(s) for s in all_syms_]
    #     # print("Total symbols (in lambdify order):", sym_names_)
    #     # print("Num y inputs:", len(vars_))
    #     # print("Num x inputs:", len(params_))

    #     # print(all_syms_)
    #     # print(sym_names_)
        
    #     torch_exprs = []

    #     # Build lambdify expressions using this function's symbol list
    #     if is_matrix:
    #         for row in sym_exprs:
    #             row_exprs = []
    #             for expr in row:
    #                 f = sp.lambdify(all_syms_, expr, modules=torch)
    #                 row_exprs.append(f)
    #             torch_exprs.append(row_exprs)
    #     else:
    #         for expr in sym_exprs:
    #             f = sp.lambdify(all_syms_, expr, modules=torch)
    #             torch_exprs.append(f)

    #     def torch_fn(y, x):
    #         # y: (B, n_vars_for_this_fn), x: (B, n_params)
    #         bsz = y.shape[0]
    #         device = y.device

    #         # Build a mapping from symbol name to column tensor using the provided vars_/params_
    #         input_dict = {}
    #         for i, sym in enumerate(vars_):
    #             # print("Var symbol:", sym, "y shape:", y.shape)
    #             input_dict[str(sym)] = y[:, i]
    #         for i, sym in enumerate(params_):
    #             # print("Param symbol:", sym, "x shape:", x.shape)
    #             input_dict[str(sym)] = x[:, i]

    #         # Assemble inputs in the lambdify order
    #         inputs = [input_dict[name] for name in sym_names_]

    #         if is_matrix:
    #             rows = []
    #             for row_expr in torch_exprs:
    #                 row_vals = []
    #                 for f in row_expr:
    #                     val = f(*inputs)
    #                     if not isinstance(val, torch.Tensor):
    #                         val = torch.full((bsz,), float(val), device=device)
    #                     elif val.ndim == 0:
    #                         val = val.expand(bsz)
    #                     row_vals.append(val)
    #                 rows.append(torch.stack(row_vals, dim=1))  # (B, row_len)
    #             return torch.stack(rows, dim=1)  # (B, n_rows, n_cols)
    #         else:
    #             vals = []
    #             for f in torch_exprs:
    #                 val = f(*inputs)
    #                 print("Evaluated val:", len(val))

    #                 if not isinstance(val, torch.Tensor):
    #                     # If val is list/ndarray of length 1, reduce it
    #                     if isinstance(val, (list, tuple)) and len(val) == 1:
    #                         val = val[0]
    #                     elif hasattr(val, "shape") and val.shape == ():
    #                         val = val.item()

    #                     # Now val should be scalar
    #                     val = torch.full((bsz,), float(val), device=device)

    #                 elif val.ndim == 0:
    #                     val = val.expand(bsz)

    #                 vals.append(val)

    #             return torch.stack(vals, dim=1)  # (B, n_exprs)


    #     return torch_fn

    def _make_torch_fn(self, sym_exprs, is_matrix=False, var_list=None, param_list=None):
        """
        Create a torch-callable function that accepts (y, x) where
        - y: (B, n_vars_for_this_fn)
        - x: (B, n_params)
        var_list and param_list define the correct symbol order used for lambdify.
        """

        # fallback to class defaults if not provided
        vars_ = var_list if var_list is not None else self.variables
        params_ = param_list if param_list is not None else self.parameters

        # Flatten sym_exprs if it's a sympy.Matrix
        if isinstance(sym_exprs, sp.Matrix):
            if is_matrix:
                sym_exprs = [[e for e in row] for row in sym_exprs.tolist()]
            else:
                sym_exprs = list(sym_exprs)  # 1D list of expressions

        all_syms_ = sorted(list(vars_) + list(params_), key=categorize)
        sym_names_ = [str(s) for s in all_syms_]

        # print(all_syms_)

        torch_exprs = []

        # Build lambdify expressions using this function's symbol list
        if is_matrix:
            for row in sym_exprs:
                row_exprs = []
                for expr in row:
                    f = sp.lambdify(all_syms_, expr, modules="torch")
                    row_exprs.append(f)
                torch_exprs.append(row_exprs)
        else:
            for expr in sym_exprs:
                f = sp.lambdify(all_syms_, expr, modules="torch")
                torch_exprs.append(f)

        def torch_fn(y, x):
            # y: (B, n_vars_for_this_fn), x: (B, n_params)
            bsz = y.shape[0]
            device = y.device

            # Build mapping from symbol -> tensor column
            input_dict = {}
            for i, sym in enumerate(vars_):
                input_dict[str(sym)] = y[:, i]
            for i, sym in enumerate(params_):
                input_dict[str(sym)] = x[:, i]

            # Assemble inputs in lambdify order
            inputs = [input_dict[name] for name in sym_names_]

            if is_matrix:
                rows = []
                for row_expr in torch_exprs:
                    row_vals = []
                    for f in row_expr:
                        val = f(*inputs)

                        # Ensure val is tensor of shape (B,)
                        if not isinstance(val, torch.Tensor):
                            val = torch.as_tensor(val, dtype=torch.float32, device=device)
                        if val.ndim == 0:
                            val = val.expand(bsz)
                        elif val.shape[0] != bsz:
                            val = val.expand(bsz)

                        row_vals.append(val)
                    rows.append(torch.stack(row_vals, dim=1))  # (B, row_len)
                return torch.stack(rows, dim=1)  # (B, n_rows, n_cols)
            else:
                vals = []
                for f in torch_exprs:
                    val = f(*inputs)

                    # Ensure val is tensor of shape (B,)
                    if not isinstance(val, torch.Tensor):
                        val = torch.as_tensor(val, dtype=torch.float32, device=device)
                    if val.ndim == 0:
                        val = val.expand(bsz)
                    elif val.shape[0] != bsz:
                        val = val.expand(bsz)

                    vals.append(val)

                return torch.stack(vals, dim=1)  # (B, n_exprs)

        return torch_fn

    
    def forward(self, y, x, is_test=False, kkt=None):
        # Select active functions and symbol context
        if is_test and kkt is not None:
            key = kkt["key"]

            if key in self.kkt_cache:
                res_fn_test, jac_fn_test, variables_test, n_y_syms_test, _ = self.kkt_cache[key]
            else:
                # print("Creating new KKT functions for key:", key)
                variables_test = sorted(kkt["variables"], key=categorize)
                # print("KKT Variables: ", variables_test)

                # Build new torch-compatible functions with explicit var/param lists
                res_fn_test = self._make_torch_fn(kkt["kkt_system"],
                                                  is_matrix=False,
                                                  var_list=variables_test,
                                                  param_list=self.parameters)

                jac_exprs = kkt["kkt_system"].jacobian(variables_test).tolist()
                jac_fn_test = self._make_torch_fn(jac_exprs,
                                                  is_matrix=True,
                                                  var_list=variables_test,
                                                  param_list=self.parameters)

                # compute how many leading y* symbols (for clamping)
                all_syms_test = sorted(list(variables_test) + list(self.parameters), key=categorize)
                y_syms_test = [s for s in all_syms_test if re.fullmatch(r"y\d+", str(s))]
                n_y_syms_test = len(y_syms_test)

                # Cache the functions and the local symbol context
                # store optional kkt system for debugging as fifth element
                self.kkt_cache[key] = (
                res_fn_test, jac_fn_test, variables_test, n_y_syms_test, kkt.get("kkt_system", None))

            # Use the test functions / context locally (do NOT mutate self.variables)
            res_fn = res_fn_test
            jac_fn = jac_fn_test
            active_variables = variables_test
            active_n_y_syms = n_y_syms_test

        else:
            # training context
            res_fn = self.res_fn
            jac_fn = self.jac_fn
            active_variables = self.variables
            active_n_y_syms = self.n_y_syms

        # ---- prepare yk ----
        yk = y.clone()
        B, n = yk.shape
        
        # print(yk)
        # time.sleep(5)

        # print("Input yk shape: ", yk.shape)
        # print("Input x shape: ", x.shape)

        # Add extra variables if needed for the active symbol context
        if is_test and kkt is not None and len(active_variables) > n:
            extra_dim = len(active_variables) - n
            extra_vals = torch.zeros(B, extra_dim, device=yk.device, dtype=yk.dtype)
            yk = torch.cat([yk, extra_vals], dim=1)
            self.is_concatenated_during_test = True
        else:
            self.is_concatenated_during_test = False

        # print("Prepared yk shape: ", yk.shape)
        # print("Prepared x shape: ", x.shape)

        # ---- Newton iteration ----
        for _ in range(self.max_iter):
            # Clamp only the active y symbols, leave extra vars free
            # yk = torch.cat(
            #     [yk[:, :active_n_y_syms].clamp_min(1e-9), yk[:, active_n_y_syms:]], dim=1
            # )

            # Evaluate residual and Jacobian using the selected functions
            # print("yk:", yk)
            # print("x:", x)
            r = res_fn(yk, x)  # (B, n_total)
            # print("residuals:", r)
            # time.sleep(15)
            J = jac_fn(yk, x)  # (B, n_total, n_total)
            
            norm = torch.linalg.norm(r, dim=1)  # (B,)
            # print(f"Iter {_}: Residual norm (batch) =", norm)
            # time.sleep(15)
            if torch.all(norm < self.tol):
                # print(f"Newton Loop Converged at iteration {_}!!")
                break

            # Ensure shapes: r (B, m), J (B, m, n)
            if r.ndim == 1:
                r = r.unsqueeze(0)
            # Solve J * delta = -r : shape handling
            try:
                # prefer batched solve when square
                delta_x = torch.linalg.solve(J, -r.unsqueeze(-1)).squeeze(-1)
                # print("Delta_x:", delta_x)
                # time.sleep(10)
                # print("Delta_x shape:", delta_x.shape)
                # Pad with zeros on the right
                zeros = torch.zeros(delta_x.shape[0], self.n_y_syms, device=delta_x.device, dtype=delta_x.dtype)
                delta_x_padded = torch.cat([zeros, delta_x], dim=1)

            except RuntimeError:
                # Regularize if singular: solve (J^T J + reg I) delta = J^T (-r)
                JT = J.transpose(-2, -1)
                JTJ = JT @ J
                I = torch.eye(JTJ.size(-1), device=yk.device)
                A = JTJ + self.reg_factor * I.unsqueeze(0)
                b = JT @ (-r.unsqueeze(-1))
                delta_x = torch.linalg.solve(A, b).squeeze(-1)

                # Pad with zeros on the right
                zeros = torch.zeros(delta_x.shape[0], self.n_y_syms, device=delta_x.device, dtype=delta_x.dtype)
                delta_x_padded = torch.cat([zeros, delta_x], dim=1)

            # print("yk", yk)
            # print("delta_x", delta_x)
            # print("delta_x_padded", delta_x_padded)
            yk = yk + self.step_size * delta_x_padded
            # print("Updated yk", yk)
            # time.sleep(10)
        
        if torch.any(norm >= self.tol):
            # print(f"Newton Loop did NOT converge within {self.max_iter} iterations.")
            pass
        
        # ---- remove concatenated extras if added ----
        if self.is_concatenated_during_test:
            yk = yk[:, :n]  # return only original variables

        # print("Output yk: ", yk)
        # time.sleep(10)

        return yk
        
