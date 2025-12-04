import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from torch import nn
import json
import pandas as pd
from run.newton_runner import NewtonRunner
from dataset.dataset import ModelDataset
from torch.utils.data import Subset,DataLoader, random_split
from dataset.test_dataset1 import BDataset
from sympy import symbols, Matrix, lambdify, solve
from KKT.ICs_BCs import ICs_BCs
from model.model import NewtonModel
from model.linear_model import NewtonModelLinear
from model.kkt_hardnet_trainer import KKT_HardNet_Trainer
from model.pinn_trainer import PINN_Trainer
from model.mlp_trainer import MLP_Trainer
import pprint
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import matplotlib as mpl

class Runner:
    def __init__(self, dir_path):
        self.config_dir = dir_path
        self.P = None
        self.system = None
        self.variables = None
        self.parameters = None
        self.kkt_combination_save_path = f"{self.config_dir}/kkt_combination.json"
        self._extract_files()
        self._define_runner()
        self._save_kkt_combinations()

    def _extract_files(self):
        # === Load configs ===
        with open(self.config_dir + "/" + "problem.json", "r") as f:
            self.P = json.load(f)

        # with open(self.config_dir + "/" + "config.json", "r") as f:
        #     self.config_dict = json.load(f)

        with open(self.config_dir + "/" + "model_config.json", "r") as f:
            self.model_config_dict = json.load(f)

    def _define_runner(self):
        # === Extract model info ===
        self.objective = self.P['objective']
        self.constraints = self.P['constraints']
        self.parameters = self.P['parameters']
        self.variables = self.P['variables']
        self.binary_variables = self.P['binary_variables']
        self.file_name = self.P["file_name"]
        
        # self.raw_ICs = self.P.get('ICs', [])
        # n_indep_bcs = self.P.get('Independent_BCs', 0)

        self.raw_ICs = self.P['ICs']
        raw_bc_blocks = {}
        
        n_indep_bcs = self.P['Independent_BCs']
        for i in range(1, n_indep_bcs + 1):
            key = f"BC{i}"
            if key in self.P and self.P[key]:
                raw_bc_blocks[key] = self.P[key]
        self.raw_BCs = raw_bc_blocks  # e.g. {"BC1": ["y1 == y1_in @ x1 = 0", ...]}
        
        # Parse ICs
        self.ICs_BCs = ICs_BCs(self.raw_ICs, self.raw_BCs)
        self.ICs_BCs.create()
        
        self.ICs = self.ICs_BCs.ICs
        self.BCs = self.ICs_BCs.BCs
        

        # === Run Newton solver ===
        runner = NewtonRunner(
            self.parameters, self.variables, self.binary_variables,
            self.objective, self.constraints, self.ICs, self.BCs,
            formulation_index=self.model_config_dict["kkt_formulation_index"], 
            taylor_order=self.model_config_dict.get("taylor_order", "auto") , 
            taylor_offset=self.model_config_dict["taylor_offset"]
        )

        self.system = runner.creator.model.residuals
        self.kkt_variables = runner.variables
        self.kkt_parameters = runner.parameters
        self.orig_eq_viol_fn = runner.creator.model.orig_eq_violation
        self.orig_ineq_viol_fn = runner.creator.model.orig_ineq_violation
        self.eq_viol = runner.creator.model.eq_violation
        self.ineq_viol = runner.creator.model.ineq_violation
        self.is_linear = runner.creator.model.is_linear
        self.kkt_combinations = runner.kkt_combinations
        self.kkt_combinations_sympy = runner.kkt_combinations_sympy
        self.y_taylor_expr = runner.creator.model.y_taylor_expr
        self.parameters_list = runner.parameters_list
        self.variables_list = runner.variables_list
        # print("Residuals: ", type(self.system))
        # print("Y taylor expr: ", type(self.y_taylor_expr))
        
        # pp = pprint.PrettyPrinter(indent=2, sort_dicts=True, compact=False)
        # pp.pprint(self.kkt_combinations)

        # print("Residuals Length: ", len(self.system))
        # for residual in self.system:
        #     print(residual)

        # print("Number of variables: ", self.kkt_variables)

    def _save_kkt_combinations(self):
        with open(self.kkt_combination_save_path, "w") as file:
            json.dump(self.kkt_combinations, file, indent=2)
        
    
    # def _create_dataset(self):
    #     self.df = pd.read_csv(self.config_dir + "/" + self.file_name)

    #     self.param_cols = [p for p in self.parameters if (not p.endswith("_data") and not p.endswith("_delta"))]
    #     print(self.param_cols)
    #     self.var_cols = [v for v in self.variables if re.fullmatch(r'y\d+', v)] + self.binary_variables

    #     self.param_df = self.df[self.param_cols].copy()
    #     self.var_df = self.df[self.var_cols].copy()
    #     self.obj_param_df = self.var_df.copy()
    #     self.obj_param_df.columns = [f"{col}_data" for col in self.obj_param_df.columns] + [f"{col}_delta" for col in self.obj_param_df.columns]

    #     self.dataset = ModelDataset(self.param_df, self.var_df, self.obj_param_df, noise_mean = self.model_config_dict["noise_mean"],
    #                                 noise_std = self.model_config_dict["noise_std"], noise_scale = self.model_config_dict["noise_scale"], apply_noise = self.model_config_dict["apply_noise"])

    def _create_dataset(self):

        # 1) Load CSV
        self.df = pd.read_csv(os.path.join(self.config_dir, self.file_name))

        # 2) Parameter columns: exclude *_data and *_delta
        self.param_cols = [p for p in self.parameters if not p.endswith(("_data", "_delta"))]

        # 3) Variable columns: y1, y2, ... plus any binary variables (dedup while preserving order)
        y_cols   = [v for v in self.variables if re.fullmatch(r"y\d+", v)]
        bin_cols = list(self.binary_variables or [])
        self.var_cols = list(dict.fromkeys(y_cols + bin_cols))

        # 4) Sanity check: ensure requested columns exist in CSV
        missing_params = [c for c in self.param_cols if c not in self.df.columns]
        missing_vars   = [c for c in self.var_cols   if c not in self.df.columns]

        # 4) Keep only columns that actually exist in the CSV
        existing_param_cols = [c for c in self.param_cols if c in self.df.columns]
        existing_var_cols   = [c for c in self.var_cols if c in self.df.columns]

        # if missing_params or missing_vars:
        #     raise UserWarning(
        #         f"Missing columns in CSV. "
        #         f"params missing: {missing_params}; vars missing: {missing_vars}"
        #     )

        # 5) Slice dataframes
        self.param_df = self.df[existing_param_cols].copy()
        self.var_df   = self.df[existing_var_cols].copy()

        # 6) Build objective-parameter dataframe: one set for *_data and one for *_delta (zeros by default)
        data_cols  = self.var_df.add_suffix("_data")
        delta_cols = pd.DataFrame(
            0.0,
            index=self.var_df.index,
            columns=[f"{c}_delta" for c in existing_var_cols],
        )
        self.obj_param_df = pd.concat([data_cols, delta_cols], axis=1)

        # 7) Create torch Dataset (noise settings are optional in config)
        self.dataset = ModelDataset(
            self.param_df,
            self.var_df,
            self.obj_param_df,
            noise_mean=self.model_config_dict.get("noise_mean", 0.0),
            noise_std=self.model_config_dict.get("noise_std", 0.0),
            noise_scale=self.model_config_dict.get("noise_scale", 1.0),
            apply_noise=self.model_config_dict.get("apply_noise", 0),
            num_points=self.model_config_dict.get("num_points", "all"),
            save_path= self.config_dir + "/dataset_with_noise.csv" if self.model_config_dict.get("apply_noise", 0) else None
        )


    # def _create_dataloader(self):
    #     batch_size = self.model_config_dict["batch_size"]
    #     total_size = len(self.dataset)
    #     train_size = int(self.model_config_dict["train_split_size"] * total_size)
    #     val_size = int(self.model_config_dict["val_split_size"] * total_size)
    #     test_size = total_size - train_size - val_size

    #     train_set, val_set, test_set = random_split(
    #         self.dataset, [train_size, val_size, test_size],
    #         generator=torch.Generator()
    #     )

    #     # Save indices
    #     self.test_indices = test_set.indices if hasattr(test_set, "indices") else np.arange(test_size)

    #     # -------------------------------
    #     # 🔹 Group test samples by KKT combination
    #     # -------------------------------
    #     grouped_indices = {k: [] for k in self.kkt_combinations.keys()}

    #     for idx in self.test_indices:
    #         x_vals, y_vals, y_data = self.dataset[idx]

    #         # # Convert tensor to numpy
    #         # x_vals_np = x_vals.detach().cpu().numpy()
    #         #
    #         # # Safe extraction of parameters (support 1D, 2D, ... cases)
    #         # x1_val = int(x_vals_np[0]) if len(x_vals_np) > 0 and not np.isnan(x_vals_np[0]) else None
    #         # x2_val = int(x_vals_np[1]) if len(x_vals_np) > 1 and not np.isnan(x_vals_np[1]) else None
    #         #
    #         # # Build key
    #         # use_ics = False  # or extract properly if encoded somewhere
    #         # key_parts = [f"ICs={use_ics}"]
    #         #
    #         # if x1_val is not None:
    #         #     key_parts.append(f"x1={x1_val}")
    #         # if x2_val is not None:
    #         #     key_parts.append(f"x2={x2_val}")
    #         #
    #         # # Fall back gracefully if only t exists
    #         # if len(x_vals_np) == 1:
    #         #     key_parts.append(f"t={int(x_vals_np[0]) if not np.isnan(x_vals_np[0]) else 'None'}")
    #         #
    #         # key = " | ".join(key_parts)

    #         # Convert tensor to numpy
    #         x_vals_np = x_vals.detach().cpu().numpy()

    #         # Extract x1, x2 from input (assuming order is consistent: x1 -> col 0, x2 -> col 1)
    #         x1_val = int(x_vals_np[0]) if not np.isnan(x_vals_np[0]) else None
    #         x2_val = int(x_vals_np[1]) if not np.isnan(x_vals_np[1]) else None

    #         # Build key
    #         use_ics = False  # or extract properly if encoded somewhere
    #         key = f"ICs={use_ics} | x1={x1_val if x1_val is not None else 'None'} | x2={x2_val if x2_val is not None else 'None'}"

    #         if key in grouped_indices:
    #             grouped_indices[key].append(idx)
    #         else:
    #             # 🚨 Assign to OG if not matching
    #             if "OG" in grouped_indices:
    #                 grouped_indices["OG"].append(idx)
    #             else:
    #                 print(f"⚠️ No matching KKT combination for key {key} and no 'OG' fallback")

    #     # -------------------------------
    #     # 🔹 Create loaders per KKT combination
    #     # -------------------------------
    #     self.test_loaders = []
    #     for key, indices in grouped_indices.items():
    #         if not indices:
    #             continue  # skip empty groups

    #         subset = Subset(self.dataset, indices)
    #         loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    #         # Attach metadata
    #         # print("KKT variables :: ", self.kkt_combinations_sympy[key]["kkt_variables"])
    #         # print("KKT parameters :: ", self.kkt_combinations_sympy[key]["parameters"])
            
    #         loader.kkt_info = {
    #             "key": key,
    #             "selection": self.kkt_combinations[key]["selection"],
    #             "use_ics": self.kkt_combinations[key]["use_ics"],
    #             "counts": self.kkt_combinations[key]["counts"],
    #             "kkt_system": self.kkt_combinations_sympy[key]["kkt_system"],
    #             "variables": self.kkt_combinations_sympy[key]["kkt_variables"],
    #             "parameters": self.kkt_combinations_sympy[key]["parameters"],
    #         }

    #         self.test_loaders.append(loader)

    #     # -------------------------------
    #     # 🔹 Train & Validation Loaders
    #     # -------------------------------
    #     self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    #     self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    #     self.test_loader_normal = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    #     # print(f"✅ Created {len(self.test_loaders)} test loaders based on KKT combinations")
    #     # kkt_system = self.test_loaders[0].kkt_info
    #     # print("Test Loader 0 info: ", kkt_system if self.test_loaders else "No test loaders")
    #     # print("System: ", self.system)

    def _create_dataloader(self):
        batch_size = self.model_config_dict["batch_size"]
        total_size = len(self.dataset)
        train_size = int(self.model_config_dict["train_split_size"] * total_size)
        val_size = int(self.model_config_dict["val_split_size"] * total_size)
        test_size = total_size - train_size - val_size

        train_set, val_set, test_set = random_split(
            self.dataset, [train_size, val_size, test_size],
            generator=torch.Generator()
        )

        # Save indices
        self.test_indices = test_set.indices if hasattr(test_set, "indices") else np.arange(test_size)

        # -------------------------------
        # 🔹 Group test samples by KKT combination
        # -------------------------------
        grouped_indices = {k: [] for k in self.kkt_combinations.keys()}

        for idx in self.test_indices:
            x_vals, y_vals, y_data = self.dataset[idx]

            # Convert tensor to numpy
            x_vals_np = x_vals.detach().cpu().numpy()
            # print("x_vals shape:", x_vals_np.shape)

            # Dynamically extract x1, x2, ..., xN
            x_entries = []
            for i, val in enumerate(x_vals_np):
                xi_val = int(val) if not np.isnan(val) else None
                x_entries.append(f"x{i+1}={xi_val if xi_val is not None else 'None'}")

            # Build key dynamically
            use_ics = False  # TODO: replace with proper flag if available
            key = f"ICs={use_ics} | " + " | ".join(x_entries)

            if key in grouped_indices:
                grouped_indices[key].append(idx)
            else:
                # 🚨 Assign to OG if not matching
                if "OG" in grouped_indices:
                    grouped_indices["OG"].append(idx)
                else:
                    print(f"⚠️ No matching KKT combination for key {key} and no 'OG' fallback")

        # -------------------------------
        # 🔹 Create loaders per KKT combination
        # -------------------------------
        self.test_loaders = []
        for key, indices in grouped_indices.items():
            if not indices:
                continue  # skip empty groups

            subset = Subset(self.dataset, indices)
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
            
            loader.kkt_info = {
                "key": key,
                "selection": self.kkt_combinations[key]["selection"],
                "use_ics": self.kkt_combinations[key]["use_ics"],
                "counts": self.kkt_combinations[key]["counts"],
                "kkt_system": self.kkt_combinations_sympy[key]["kkt_system"],
                "variables": self.kkt_combinations_sympy[key]["kkt_variables"],
                "parameters": self.kkt_combinations_sympy[key]["parameters"],
            }

            self.test_loaders.append(loader)

        # -------------------------------
        # 🔹 Train & Validation Loaders
        # -------------------------------
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        self.test_loader_normal = DataLoader(test_set, batch_size=batch_size, shuffle=False)


#############################################################################################################################################################################################################################
#############################################################################################################################################################################################################################

    def _explicit_axis_values(self):
        """
        Gather the set of explicit BC values per axis (e.g., {'x1': {0, 10}, 'x2': {5}}).
        Values are converted to numeric where possible.
        """
        axis_vals = {}
        if not getattr(self, "BCs", None):
            return axis_vals
        for ax, vdict in self.BCs.items():
            s = set()
            for v in vdict.keys():
                try:
                    fv = float(v)
                    v = int(fv) if fv.is_integer() else fv
                except Exception:
                    pass
                s.add(v)
            axis_vals[ax] = s
        return axis_vals

    def _mask_for_combo(self, df: pd.DataFrame, selection: dict, use_ics: bool, axis_values: dict):
        """
        Build a boolean mask for a single KKT combination:
          - ICs=True  -> t == 0   (if 't' exists)
          - ICs=False -> t != 0   (if 't' exists)
          - For each axis in `selection`:
              val is None -> df[ax] is NOT in explicit axis_values[ax]
              val is x    -> df[ax] == x
        """
        mask = pd.Series(True, index=df.index)

        # ICs rule (only if 't' column exists)
        if "t" in df.columns:
            if use_ics:
                mask &= (df["t"] == 0)
            else:
                mask &= (df["t"] != 0)

        # BC rules
        for ax, sel_val in selection.items():
            if ax not in df.columns:
                # If axis not present in dataset, ignore this axis
                continue

            if sel_val is None:
                # anything except the explicit BC values for this axis
                disallow = axis_values.get(ax, set())
                mask &= ~df[ax].isin(disallow)
            else:
                mask &= (df[ax] == sel_val)

        return mask

    def build_test_buckets_by_kkt(self, batch_size: int = 32, shuffle: bool = False):
        """
        Partition the test set into MUTUALLY EXCLUSIVE buckets for every KKT combination
        (ICs + BCs). A combination where selection has ax=None means
        'values not equal to any explicit BC value for that axis'.

        # print("Y taylor expr: ", type(self.y_taylor_expr))
        Produces:
            self.test_buckets:        dict[key] -> Subset(dataset, idxs)
            self.test_bucket_loaders: dict[key] -> DataLoader
            self.test_bucket_counts:  dict[key] -> int
        """
        if not hasattr(self, "test_indices"):
            raise RuntimeError("Call _create_dataloader() before bucketing test sets.")

        # Slice only the test rows and keep the global mapping back to dataset indices
        test_df = self.df.iloc[self.test_indices].copy()
        test_df["__global_idx__"] = self.test_indices

        # Collect explicit BC values per axis (used for the 'None' meaning)
        axis_values = self._explicit_axis_values()

        # Build raw masks for each combo
        raw_masks = {}
        for key, pack in self.kkt_combinations.items():
            selection = pack.get("selection", {})
            use_ics = bool(pack.get("use_ics", False))

            raw_masks[key] = self._mask_for_combo(
                df=test_df,
                selection=selection,
                use_ics=use_ics,
                axis_values=axis_values
            )

        # Enforce mutual exclusivity by assigning rows in a stable order
        # (the order of keys from your kkt_combinations dict).
        assigned = pd.Series(False, index=test_df.index)
        exclusive_indices = {k: [] for k in self.kkt_combinations.keys()}

        for key in self.kkt_combinations.keys():
            # rows that match this combo AND have not been assigned yet
            m = raw_masks[key] & (~assigned)
            idxs_local = test_df.index[m]
            assigned.loc[idxs_local] = True
            # map to dataset indices
            gidxs = test_df.loc[idxs_local, "__global_idx__"].tolist()
            exclusive_indices[key] = gidxs

        # Optionally, assert that we covered all test rows (if desired)
        # assert assigned.sum() == len(test_df), "Some test rows were not assigned to any bucket."

        # Build subsets and loaders
        self.test_buckets = {}
        self.test_bucket_loaders = {}
        for key, idxs in exclusive_indices.items():
            if not idxs:
                continue
            subset = Subset(self.dataset, idxs)
            loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
            self.test_buckets[key] = subset
            self.test_bucket_loaders[key] = loader

        self.test_bucket_counts = {k: len(v) for k, v in self.test_buckets.items()}
        return self.test_bucket_loaders

    def _sanitize_key(self, key: str) -> str:
        """Safe filename from KKT combo key."""
        key = key.strip()
        key = key.replace(" | ", "_").replace("=", "_")
        key = re.sub(r"[^A-Za-z0-9_\-]", "_", key)
        return key

    def save_all_test_sets(self, save_csv: bool = False):
        """
        Print every bucketed test set (full rows from the source CSV).
        If save_csv=True, also write each bucket to <config_dir>/testset_<key>.csv
        """
        if not hasattr(self, "test_buckets") or not self.test_buckets:
            raise RuntimeError("No test buckets found. Call build_test_buckets_by_kkt() first.")

        # print("\n=== Test Buckets Summary ===")
        for k, subset in self.test_buckets.items():
            # Subset.indices is the dataset-global index list we created
            idxs = subset.indices if hasattr(subset, "indices") else []
            df_part = self.df.iloc[idxs].copy()

            # print(f"\n--- [{k}] ---")
            # print(f"Samples: {len(df_part)}")
            # if len(df_part) > 0:
            #     with pd.option_context("display.max_rows", None,
            #                            "display.max_columns", None,
            #                            "display.width", 200,
            #                            "display.precision", 10):
            #         print(df_part.to_string(index=False))
            # else:
            #     print("(empty)")
            #     pass

            if save_csv and len(df_part) > 0:
                fname = f"testset_{self._sanitize_key(k)}.csv"
                out_dir = os.path.join(self.config_dir, "test_dataset")
                os.makedirs(out_dir, exist_ok=True)  # create the folder if it doesn't exist
                out_path = os.path.join(out_dir, fname)
                df_part.to_csv(out_path, index=False)
                # print(f"Saved: {out_path}")

#############################################################################################################################################################################################################################
#############################################################################################################################################################################################################################

    def _define_model(self, is_kkt_hardnet: bool = False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Cuda Availability: {torch.cuda.is_available()}: Using {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        # if self.is_linear:
        #     self.model = NewtonModelLinear(
        #         residuals=self.system, 
        #         eq_violation=self.eq_viol, 
        #         ineq_violation=self.ineq_viol,
        #         variables=self.kkt_variables,
        #         parameters=self.kkt_parameters,
        #         input_dim=len(self.param_cols),
        #         config=self.model_config_dict
        #     ).to(device)
        # else:
        #     print("running")
        #     self.model = NewtonModel(
        #         residuals=self.system, 
        #         eq_violation=self.eq_viol, 
        #         ineq_violation=self.ineq_viol,
        #         variables=self.kkt_variables,
        #         parameters=self.kkt_parameters,
        #         input_dim=len(self.param_cols),
        #         config=self.model_config_dict,
        #         is_kkt_model=is_kkt_hardnet
        #     ).to(device)
        # print("running")
        self.model = NewtonModel(
            residuals=self.system, 
            taylor_exprs=self.y_taylor_expr,
            orig_eq_violation=self.orig_eq_viol_fn,
            orig_ineq_violation=self.orig_ineq_viol_fn,
            eq_violation=self.eq_viol, 
            ineq_violation=self.ineq_viol,
            variables=self.kkt_variables,
            parameters=self.kkt_parameters,
            input_dim=len(self.param_cols),
            config=self.model_config_dict,
            is_kkt_model=is_kkt_hardnet
        ).to(device)

    def _define_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config_dict["lr"])
        # self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.model_config_dict["lr"])
        # self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.model_config_dict["lr"])
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.model_config_dict["lr"])
        # self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=self.model_config_dict["lr"])

    def _define_loss(self):
        self.criterion = nn.MSELoss()

    def _plot_losses(self):
        mpl.rcParams['text.usetex'] = False
        mpl.rcParams['mathtext.fontset'] = 'stix'
        loss_files = {
            'MLP': self.config_dir + "/" + 'mlp_losses.npz',
            'PINN': self.config_dir + "/" + 'pinn_losses.npz',
            'KKT-HardNet': self.config_dir + "/" + 'kkt_hardnet_losses.npz'
        }

        data_losses = {}     # {label: (train_data, test_data)}
        physics_losses = {}  # {label: (train_pinn, test_pinn)}
        abs_violation = {}

        # Load all loss files first
        for label, file in loss_files.items():
            if not os.path.exists(file):
                print(f"⚠️ File not found: {file}")
                continue

            data = np.load(file)
            min_val = 1e-12
            
            train_data = np.clip(data['train_data_loss'], min_val, None)
            test_data = np.clip(data['test_data_loss'], min_val, None)
            train_pinn = np.clip(data['train_pinn_loss'], min_val, None)
            test_pinn = np.clip(data['test_pinn_loss'], min_val, None)
            train_abs_violation = np.clip(data['train_abs_violation'], min_val, None)
            test_abs_violation = np.clip(data['test_abs_violation'], min_val, None)
            

            data_losses[label] = (train_data, test_data)
            physics_losses[label] = (train_pinn, test_pinn)
            abs_violation[label] = (train_abs_violation, test_abs_violation)
        
        # ===== 1. Plot Data Losses =====
        plt.figure(figsize=(9, 6))
        for label, (train_data, test_data) in data_losses.items():
            epochs = np.arange(len(train_data))
            plt.plot(epochs, train_data, label=f'{label} Train Data', linestyle='-', linewidth=2.5)
            plt.plot(epochs, test_data, label=f'{label} Test Data', linestyle='--', linewidth=2.5)

        plt.xlabel("Epochs", fontsize=20, fontweight="bold")
        plt.ylabel("Data Loss (MSE)", fontsize=20, fontweight="bold")
        plt.yscale('log')
        plt.legend(fontsize=12)
        plt.tight_layout()
        save_path_data = os.path.join(self.config_dir, "all_models_data_loss_plot.png")
        plt.savefig(save_path_data)
        print(f"📁 Saved Data Loss Plot: {save_path_data}")
        plt.close()

        # ===== 2. Plot Physics Losses =====
        plt.figure(figsize=(9, 6))
        for label, (train_pinn, test_pinn) in physics_losses.items():
            epochs = np.arange(len(train_pinn))
            plt.plot(epochs, train_pinn, label=f'{label} Train Physics Loss', linestyle='-', linewidth=2.5)
            plt.plot(epochs, test_pinn, label=f'{label} Test Physics Loss', linestyle='--', linewidth=2.5)

        plt.xlabel("Epochs", fontsize=20, fontweight="bold")
        plt.ylabel("Physics Loss", fontsize=20, fontweight="bold")
        plt.yscale('log')
        plt.legend(fontsize=12)
        plt.tight_layout()
        save_path_physics = os.path.join(self.config_dir, "all_models_physics_loss_plot.png")
        plt.savefig(save_path_physics)
        print(f"📁 Saved Physics Loss Plot: {save_path_physics}")
        plt.close()
        
        # ===== 3. Plot Absolute Violation =====
        plt.figure(figsize=(9, 6))
        for label, (train_violation, test_violation) in abs_violation.items():
            epochs = np.arange(len(train_violation))
            plt.plot(epochs, train_violation, label=f'{label} Train Absolute Violation', linestyle='-', linewidth=2.5)
            plt.plot(epochs, test_violation, label=f'{label} Test Absolute Violation', linestyle='--', linewidth=2.5)

        plt.xlabel("Epochs", fontsize=20)
        plt.ylabel("Absolute Violation", fontsize=20)
        plt.yscale('log')
        plt.legend(fontsize=12)
        plt.tight_layout()
        save_path_violation = os.path.join(self.config_dir, "all_models_absolute_violation_plot.png")
        plt.savefig(save_path_violation)
        print(f"📁 Saved Physics Loss Plot: {save_path_violation}")
        plt.close()

        # # ===== 3. RMSE and Absolute Violation =====
        
        plt.rcParams.update({
            "font.size": 16,
            "axes.linewidth": 1.5,
            "xtick.major.size": 6, "xtick.major.width": 1.5,
            "ytick.major.size": 6, "ytick.major.width": 1.5,
            "legend.frameon": False
        })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=600)

        # Define colors for consistency
        color_map = {
            "MLP": "g",
            "PINN": "b",
            "KKT-HardNet": "r"
        }

        # RMSE plot
        for label, (train_data_loss, test_data_loss) in data_losses.items():
            color = color_map.get(label, 'o')
            train_rmse = np.sqrt(train_data_loss)
            test_rmse = np.sqrt(test_data_loss)
            epochs = np.arange(len(train_rmse))

            if label == "KKT-HardNet":
                label = "DAE-HardNet"

            ax1.plot(epochs, train_rmse, label=f"{label} (train)", linewidth=0.5, color=color)
            ax1.plot(epochs, test_rmse, label=f"{label} (val)", linestyle="--", linewidth=0.5, color=color)
            # min_val = min(train_rmse.min(), test_rmse.min())
            min_val_train = train_rmse.min()
            min_val_test = test_rmse.min()
            ax1.axhline(y=min_val_train, linestyle="-", linewidth=0.5, color=color)
            ax1.axhline(y=min_val_test, linestyle="--", linewidth=0.5, color=color)

        ax1.set(xlabel="Epoch", ylabel="RMSE")
        ax1.set_yscale("log")

        # Violation plot
        for label, (train_violation, test_violation) in abs_violation.items():
            color = color_map.get(label, 'o')
            epochs = np.arange(len(train_violation))

            if label == "KKT-HardNet":
                label = "DAE-HardNet"

            ax2.plot(epochs, train_violation, label=f"{label} (train)", linewidth=0.5, color=color)
            ax2.plot(epochs, test_violation, label=f"{label} (val)", linestyle="--", linewidth=0.5, color=color)
            # min_val = min(train_violation.min(), test_violation.min())
            min_val_train = train_violation.min()
            min_val_test = test_violation.min()
            ax2.axhline(y=min_val_train, linestyle="-", linewidth=0.5, color=color)
            ax2.axhline(y=min_val_test, linestyle="--", linewidth=0.5, color=color)

        ax2.set(xlabel="Epoch", ylabel="Absolute Violation")
        ax2.set_yscale("log")

        # ===== Shared Legend Below Both Plots =====
        # Get handles and labels from one axis (they’re identical across both)
        handles, labels = ax1.get_legend_handles_labels()

        # Place shared legend centered below both plots
        fig.legend(handles, labels,
                loc='lower center',
                ncol=len(labels)//2,     # number of columns for horizontal layout
                fontsize=14,
                frameon=False,
                bbox_to_anchor=(0.5, -0.05))

        plt.tight_layout(rect=[0, 0.08, 1, 1])  # leave space for legend at bottom
        save_path_rmse_violation = os.path.join(self.config_dir, "all_models_rmse_and_violation.png")
        plt.savefig(save_path_rmse_violation, dpi=600, bbox_inches='tight')
        print(f":file_folder: Saved Manuscript Plot: {save_path_rmse_violation}")
        plt.close()

    def detect_structure(self, df):
        """Detect number of x's, t, and y's."""
        x_cols = [c for c in df.columns if c.startswith("x")]
        y_cols = [c for c in df.columns if c.startswith("y")]
        has_t = "t" in df.columns
        return x_cols, y_cols, has_t
    
    def sample_times(self, t_values, num=3):
        """Pick num time steps at [start, mid, end]."""
        t_values = np.sort(t_values)
        if len(t_values) == 0:
            return []
        if len(t_values) <= num:
            return list(t_values)
        
        # first, middle, and last indices
        idxs = [0, len(t_values) // 2, len(t_values) - 1]
        samples = [t_values[i] for i in idxs]
    
        # Round midpoint (only the middle one)
        samples[1] = round(samples[1], 14)
        return samples

    
    def plot_csv_comparisons(self, model_files, save_path, cmap="inferno"):
        """
        Plot comparisons: Ground truth + 3 models.
        model_files should be dict with keys "MLP", "PINN", "DAE-Hardnet".
        """
        plt.rcParams.update({
            "font.size": 20,
            "axes.linewidth": 1.5,
            "xtick.major.size": 6, "xtick.major.width": 1.5,
            "ytick.major.size": 6, "ytick.major.width": 1.5,
            "legend.frameon": False
        })
        assert set(model_files.keys()) == {"Ground Truth", "MLP", "PINN", "DAE-HardNet"}, \
            "model_files must have keys: 'MLP', 'PINN', 'DAE-HardNet'"

        # Load ground truth and models
        dfs = {}
        for name, f in model_files.items():
            dfs[name] = pd.read_csv(f)

        # Assume structure from GT
        x_cols, y_cols, has_t = self.detect_structure(dfs["Ground Truth"])
        dim_x = len(x_cols)

        # Add violation column for GT
        cutoff_violation = 1e-6
        dfs["Ground Truth"]["abs_violation"] = cutoff_violation

        # Clip violations for models
        for name, df in dfs.items():
            if name != "Ground Truth":
                df["abs_violation"] = df["abs_violation"].clip(lower=cutoff_violation)

        # --- Time-dependent case ---
        if has_t:
            print("Detected time-dependent data.")
            t_vals = np.sort(dfs["Ground Truth"]["t"].unique())
            sampled_ts = self.sample_times(t_vals, 3)

            if dim_x == 1:  # Heatmap visualization for (x, t)
                x_cols = [x_cols[0], "t"]  # ensure order
                # fig, axes = plt.subplots(2, len(dfs), figsize=(6*len(dfs), 12))

                # # Collect vmin/vmax for consistency
                # vmin_y, vmax_y, vmin_v, vmax_v = np.inf, -np.inf, np.inf, -np.inf
                # grids_y, grids_v = {}, {}
                # for model_name, df_model in dfs.items():
                #     grid_y = df_model.pivot(index=x_cols[1], columns=x_cols[0], values="y1").values
                #     grid_v = df_model.pivot(index=x_cols[1], columns=x_cols[0], values="abs_violation").values
                #     grids_y[model_name] = grid_y
                #     grids_v[model_name] = grid_v
                #     vmin_y, vmax_y = min(vmin_y, grid_y.min()), max(vmax_y, grid_y.max())
                #     vmin_v, vmax_v = min(vmin_v, grid_v.min()), max(vmax_v, grid_v.max())

                # for j, (model_name, grid_y) in enumerate(grids_y.items()):
                #     ax_y = axes[0, j]
                #     im_y = ax_y.imshow(grid_y, origin="lower", cmap=cmap, aspect="auto",
                #                     vmin=vmin_y, vmax=vmax_y)
                #     ax_y.set_title(model_name, fontsize=24)  # only once

                #     ax_v = axes[1, j]
                #     im_v = ax_v.imshow(grids_v[model_name]+1e-6, origin="lower", cmap="gray",
                #                     aspect="auto", norm=LogNorm(vmin=max(vmin_v,cutoff_violation), vmax=max(vmax_v,cutoff_violation)))

                # # Shared colorbars
                # cbar_y = fig.colorbar(im_y, ax=axes[0, :], orientation="vertical", fraction=0.025, pad=0.04)
                # cbar_y.set_label("y", fontsize=24)
                # cbar_v = fig.colorbar(im_v, ax=axes[1, :], orientation="vertical", fraction=0.025, pad=0.04)
                # cbar_v.set_label("Abs Violation", fontsize=24)
                
                # Define the base Reds colormap
                base_cmap = plt.get_cmap("Reds")

                # Create a custom colormap with a stronger red above 1e-3
                # We compress the lower values and expand the upper range
                colors = [
                    (0.0, base_cmap(0.0)),   # dark, low errors
                    (0.25, base_cmap(0.15)),
                    (0.5, base_cmap(0.35)),
                    (0.60, base_cmap(0.55)),
                    (0.75, base_cmap(0.75)),
                    (1.0, base_cmap(1.0)),   # bright red for large errors
                ]

                cmap_rmse = LinearSegmentedColormap.from_list("BiasedReds", colors, N=256)

                fig, axes = plt.subplots(3, len(dfs), figsize=(6 * len(dfs), 18))

                # Collect vmin/vmax for consistency
                vmin_y, vmax_y, vmin_v, vmax_v = np.inf, -np.inf, np.inf, -np.inf
                vmin_rmse, vmax_rmse = np.inf, -np.inf
                grids_y, grids_v, grids_rmse = {}, {}, {}

                # Ground truth grid for RMSE calculation
                if "Ground Truth" not in dfs:
                    raise ValueError("Ground Truth data must be included in dfs for RMSE computation.")
                gt_df = dfs["Ground Truth"]
                gt_grid = gt_df.pivot(index=x_cols[1], columns=x_cols[0], values="y1").values

                for model_name, df_model in dfs.items():
                    grid_y = df_model.pivot(index=x_cols[1], columns=x_cols[0], values="y1").values
                    grid_v = df_model.pivot(index=x_cols[1], columns=x_cols[0], values="abs_violation").values
                    grids_y[model_name] = grid_y
                    grids_v[model_name] = grid_v

                    if model_name == "Ground Truth":
                        rmse_grid = np.ones_like(gt_grid) * 1e-6  # Machine precision baseline
                    else:
                        rmse_grid = np.sqrt((grid_y - gt_grid) ** 2) + 1e-6  # Avoid log(0)
                    
                    grids_rmse[model_name] = rmse_grid
                    vmin_rmse = min(vmin_rmse, rmse_grid.min())
                    vmax_rmse = max(vmax_rmse, rmse_grid.max())
                    vmin_y, vmax_y = min(vmin_y, grid_y.min()), max(vmax_y, grid_y.max())
                    vmin_v, vmax_v = min(vmin_v, grid_v.min()), max(vmax_v, grid_v.max())

                cmap_pred = "rainbow"   # for y
                # cmap_rmse = "Reds"   # for log-scale RMSE
                cmap_violation = "Greys" # for abs violation

                for j, (model_name, grid_y) in enumerate(grids_y.items()):
                    # --- Top row: Prediction (y) ---
                    ax_y = axes[0, j]
                    im_y = ax_y.imshow(grid_y, origin="lower", cmap=cmap_pred, aspect="auto",
                                    vmin=vmin_y, vmax=vmax_y)
                    ax_y.set_title(model_name, fontsize=22)

                    # --- Middle row: Log-scaled RMSE ---
                    ax_rmse = axes[1, j]
                    im_rmse = ax_rmse.imshow(grids_rmse[model_name], origin="lower", cmap=cmap_rmse,
                                            aspect="auto", norm=LogNorm(vmin=vmin_rmse, vmax=vmax_rmse))
                    
                    # --- Bottom row: Abs Violation ---
                    ax_v = axes[2, j]
                    im_v = ax_v.imshow(grids_v[model_name] + 1e-6, origin="lower", cmap=cmap_violation,
                                    aspect="auto",
                                    norm=LogNorm(vmin=max(vmin_v, cutoff_violation),
                                                    vmax=max(vmax_v, cutoff_violation)))

                # --- Shared colorbars ---
                cbar_y = fig.colorbar(im_y, ax=axes[0, :], orientation="vertical", fraction=0.025, pad=0.04)
                cbar_y.set_label("$T$", fontsize=24)

                cbar_rmse = fig.colorbar(im_rmse, ax=axes[1, :], orientation="vertical", fraction=0.025, pad=0.04)
                cbar_rmse.set_label("RMSE", fontsize=22)

                cbar_v = fig.colorbar(im_v, ax=axes[2, :], orientation="vertical", fraction=0.025, pad=0.04)
                cbar_v.set_label("Absolute Violation", fontsize=22)

                # --- Common labels and formatting ---
                for ax_row in axes:
                    for ax in ax_row:
                        ax.tick_params(axis="both", labelsize=18)
                # --- Global axis labels ---
                fig.text(0.5, 0.07, r"$t$", ha='center', va='center', fontsize=24)
                fig.text(0.08, 0.5, r"$x$", ha='center', va='center', rotation='vertical', fontsize=24)

                # plt.tight_layout()
                # plt.savefig("heatmap_comparison_logRMSE.png", dpi=600, bbox_inches="tight")
                # plt.show()

                # plt.tight_layout()
                # plt.savefig("heatmap_comparison.png", dpi=600, bbox_inches="tight")
                # plt.show()

            elif dim_x == 2:  # Heatmaps
                fig, axes = plt.subplots(2, len(dfs)*len(sampled_ts),
                                        figsize=(6*len(dfs)*len(sampled_ts), 12))

                for i, t_sel in enumerate(sampled_ts):
                    # Collect vmin/vmax for consistency
                    vmin_y, vmax_y, vmin_v, vmax_v = np.inf, -np.inf, np.inf, -np.inf
                    grids_y, grids_v = {}, {}
                    for model_name, df_model in dfs.items():
                        df_slice = df_model[df_model["t"] == t_sel]
                        grid_y = df_slice.pivot(index=x_cols[1], columns=x_cols[0], values="y1").values
                        grid_v = df_slice.pivot(index=x_cols[1], columns=x_cols[0], values="abs_violation").values
                        grids_y[model_name] = grid_y
                        grids_v[model_name] = grid_v
                        vmin_y, vmax_y = min(vmin_y, grid_y.min()), max(vmax_y, grid_y.max())
                        vmin_v, vmax_v = min(vmin_v, grid_v.min()), max(vmax_v, grid_v.max())

                    for j, (model_name, grid_y) in enumerate(grids_y.items()):
                        col_idx = i*len(dfs)+j
                        ax_y = axes[0, col_idx]
                        im_y = ax_y.imshow(grid_y, origin="lower", cmap=cmap, aspect="auto",
                                        vmin=vmin_y, vmax=vmax_y)
                        if i == 0:  # titles only on first row
                            ax_y.set_title(model_name, fontsize=30)

                        ax_v = axes[1, col_idx]
                        im_v = ax_v.imshow(grids_v[model_name]+1e-6, origin="lower", cmap="gray",
                                        aspect="auto", norm=LogNorm(vmin=max(vmin_v,cutoff_violation), vmax=max(vmax_v,cutoff_violation)))

                # Shared colorbars
                cbar_y = fig.colorbar(im_y, ax=axes[0, :], orientation="vertical", fraction=0.025, pad=0.04)
                cbar_y.set_label("y", fontsize=16)
                cbar_v = fig.colorbar(im_v, ax=axes[1, :], orientation="vertical", fraction=0.025, pad=0.04)
                cbar_v.set_label("Abs Violation", fontsize=16)

        # --- Static case (no time) ---
        else:
            if dim_x == 1: 
                fig, axes = plt.subplots(1, len(y_cols) + 1, figsize=(7 * (len(y_cols) + 1), 6), sharex=True)

                # Ensure axes is iterable
                if len(y_cols) + 1 == 1:
                    axes = [axes]

                # --- Plot y columns ---
                for j, y_col in enumerate(y_cols):
                    ax_y = axes[j] if len(y_cols) > 1 else axes[0]
                    for model_name, df_model in dfs.items():
                        df_model = df_model.sort_values(x_cols[0])
                        ax_y.plot(df_model[x_cols[0]], df_model[y_col], label=model_name)

                    ax_y.set_xlabel("t", fontsize=24)
                    ax_y.set_ylabel(y_col, fontsize=24)
                    ax_y.tick_params(axis="both", which="major", labelsize=24)

                # --- Last column: abs_violation ---
                ax_v = axes[-1]
                for model_name, df_model in dfs.items():
                    df_model = df_model.sort_values(x_cols[0])
                    ax_v.plot(df_model[x_cols[0]], df_model["abs_violation"], label=model_name)

                ax_v.set_xlabel("t", fontsize=24)
                ax_v.set_ylabel("Absolute Violation", fontsize=24)
                ax_v.set_yscale("log")
                ax_v.tick_params(axis="both", which="major", labelsize=24)

                # --- Common legend below all plots ---
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", fontsize=24, ncol=len(dfs), frameon=False)

                plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space at bottom for legend




            elif dim_x == 2:  # Heatmaps
                print("Detected static 2D data.")
                # Define the base Reds colormap
                base_cmap = plt.get_cmap("Reds")

                # Create a custom colormap with a stronger red above 1e-3
                # We compress the lower values and expand the upper range
                colors = [
                    (0.0, base_cmap(0.0)),   # dark, low errors
                    (0.25, base_cmap(0.15)),
                    (0.5, base_cmap(0.35)),
                    (0.60, base_cmap(0.55)),
                    (0.75, base_cmap(0.75)),
                    (1.0, base_cmap(1.0)),   # bright red for large errors
                ]

                cmap_rmse = LinearSegmentedColormap.from_list("BiasedReds", colors, N=256)

                fig, axes = plt.subplots(3, len(dfs), figsize=(6 * len(dfs), 18))

                # Collect vmin/vmax for consistency
                vmin_y, vmax_y, vmin_v, vmax_v = np.inf, -np.inf, np.inf, -np.inf
                vmin_rmse, vmax_rmse = np.inf, -np.inf
                grids_y, grids_v, grids_rmse = {}, {}, {}

                # Ground truth grid for RMSE calculation
                if "Ground Truth" not in dfs:
                    raise ValueError("Ground Truth data must be included in dfs for RMSE computation.")
                gt_df = dfs["Ground Truth"]
                gt_grid = gt_df.pivot(index=x_cols[1], columns=x_cols[0], values="y1").values

                for model_name, df_model in dfs.items():
                    grid_y = df_model.pivot(index=x_cols[1], columns=x_cols[0], values="y1").values
                    grid_v = df_model.pivot(index=x_cols[1], columns=x_cols[0], values="abs_violation").values
                    grids_y[model_name] = grid_y
                    grids_v[model_name] = grid_v

                    if model_name == "Ground Truth":
                        rmse_grid = np.ones_like(gt_grid) * 1e-5  # Machine precision baseline
                    else:
                        rmse_grid = np.sqrt((grid_y - gt_grid) ** 2) + 1e-12  # Avoid log(0)
                    
                    grids_rmse[model_name] = rmse_grid
                    vmin_rmse = min(vmin_rmse, rmse_grid.min())
                    vmax_rmse = max(vmax_rmse, rmse_grid.max())
                    vmin_y, vmax_y = min(vmin_y, grid_y.min()), max(vmax_y, grid_y.max())
                    vmin_v, vmax_v = min(vmin_v, grid_v.min()), max(vmax_v, grid_v.max())

                cmap_pred = "rainbow"   # for y
                # cmap_rmse = "Reds"   # for log-scale RMSE
                cmap_violation = "Greys" # for abs violation

                for j, (model_name, grid_y) in enumerate(grids_y.items()):
                    # --- Top row: Prediction (y) ---
                    ax_y = axes[0, j]
                    im_y = ax_y.imshow(grid_y, origin="lower", cmap=cmap_pred, aspect="auto",
                                    vmin=vmin_y, vmax=vmax_y)
                    ax_y.set_title(model_name, fontsize=22)

                    # --- Middle row: Log-scaled RMSE ---
                    ax_rmse = axes[1, j]
                    im_rmse = ax_rmse.imshow(grids_rmse[model_name], origin="lower", cmap=cmap_rmse,
                                            aspect="auto", norm=LogNorm(vmin=vmin_rmse, vmax=vmax_rmse))
                    
                    # --- Bottom row: Abs Violation ---
                    ax_v = axes[2, j]
                    im_v = ax_v.imshow(grids_v[model_name] + 1e-6, origin="lower", cmap=cmap_violation,
                                    aspect="auto",
                                    norm=LogNorm(vmin=max(vmin_v, cutoff_violation),
                                                    vmax=max(vmax_v, cutoff_violation)))

                # --- Shared colorbars ---
                cbar_y = fig.colorbar(im_y, ax=axes[0, :], orientation="vertical", fraction=0.025, pad=0.04)
                cbar_y.set_label("$y$", fontsize=24)

                cbar_rmse = fig.colorbar(im_rmse, ax=axes[1, :], orientation="vertical", fraction=0.025, pad=0.04)
                cbar_rmse.set_label("RMSE", fontsize=22)

                cbar_v = fig.colorbar(im_v, ax=axes[2, :], orientation="vertical", fraction=0.025, pad=0.04)
                cbar_v.set_label("Absolute Violation", fontsize=22)

                # --- Common labels and formatting ---
                for ax_row in axes:
                    for ax in ax_row:
                        ax.tick_params(axis="both", labelsize=18)
                # --- Global axis labels ---
                fig.text(0.5, 0.07, r"$x_1$", ha='center', va='center', fontsize=24)
                fig.text(0.08, 0.5, r"$x_2$", ha='center', va='center', rotation='vertical', fontsize=24)

                # plt.tight_layout()
                # plt.savefig("heatmap_comparison_logRMSE.png", dpi=600, bbox_inches="tight")
                # plt.show()

                # plt.tight_layout()
                # plt.savefig("heatmap_comparison.png", dpi=600, bbox_inches="tight")
                # plt.show()


        # plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        print(f"Saved comparison plot with GT: {save_path}")
        plt.close()

    def _plot_comparison(self):
        model_files = {
            "Ground Truth": os.path.join(self.config_dir, self.file_name),
            "MLP": self.config_dir + "/" + "mlp_predictions.csv",
            "PINN": self.config_dir + "/" + "pinn_predictions.csv",
            "DAE-HardNet": self.config_dir + "/" + "kkt_hardnet_predictions.csv"
        }
        save_path = os.path.join(self.config_dir + "/" + "all_models_comparison.png")
        self.plot_csv_comparisons(model_files, save_path)

    def train_mlp(self):
        print("==============Training MLP System=======================")
        self._define_model()
        self._define_optimizer()
        self._define_loss()
        self.mlp_trainer = MLP_Trainer(
            self.config_dir, self.model, self.train_loader, self.val_loader, self.test_loader_normal,
            self.optimizer, self.criterion,
            parameters_list=self.parameters_list,
            variables_list=self.variables_list,
            pinn_reg_factor=self.model_config_dict["pinn_reg_factor"],
            num_epochs=self.model_config_dict["num_epochs"],
            eta=self.model_config_dict["eta"],
            taylor_offset=self.model_config_dict["taylor_offset"],
            gaussian_mean=self.model_config_dict["noise_mean"],
            gaussian_var=self.model_config_dict["noise_std"],
            gaussian_scale=self.model_config_dict["noise_scale"],
            model_loss_tolerance=self.model_config_dict["model_loss_tolerance"],
            checkpoint_path=self.model_config_dict["mlp_checkpoint_path"],
            save_checkpoint_iter=self.model_config_dict["save_checkpoint_iter"]
        )
        self.mlp_trainer.train()

    def train_pinn(self):
        print("==============Training PINN System=======================")
        self._define_model()
        self._define_optimizer()
        self._define_loss()
        self.pinn_trainer = PINN_Trainer(
            self.config_dir, self.model, self.train_loader, self.val_loader, self.test_loader_normal,
            self.optimizer, self.criterion,
            parameters_list=self.parameters_list,
            variables_list=self.variables_list,
            pinn_reg_factor=self.model_config_dict["pinn_reg_factor"],
            num_epochs=self.model_config_dict["num_epochs"],
            eta=self.model_config_dict["eta"],
            taylor_offset=self.model_config_dict["taylor_offset"],
            gaussian_mean=self.model_config_dict["noise_mean"],
            gaussian_var=self.model_config_dict["noise_std"],
            gaussian_scale=self.model_config_dict["noise_scale"],
            model_loss_tolerance=self.model_config_dict["model_loss_tolerance"],
            checkpoint_path=self.model_config_dict["pinn_checkpoint_path"],
            save_checkpoint_iter=self.model_config_dict["save_checkpoint_iter"]
        )
        self.pinn_trainer.train()

    def train_kkt_hardnet(self):
        print("==============Training KKT-HardNet System=======================")
        self._define_model(is_kkt_hardnet=True)
        self._define_optimizer()
        self._define_loss()
        self.kkt_trainer = KKT_HardNet_Trainer(
            self.config_dir, self.model, self.train_loader, self.val_loader, self.test_loaders,
            self.optimizer, self.criterion,
            parameters_list=self.parameters_list,
            variables_list=self.variables_list,
            pinn_reg_factor=self.model_config_dict["pinn_reg_factor"],
            hardnet_reg_factor=self.model_config_dict["hardnet_reg_factor"],
            num_epochs=self.model_config_dict["num_epochs"],
            eta=self.model_config_dict["eta"],
            model_loss_tolerance=self.model_config_dict["model_loss_tolerance"],
            taylor_offset=self.model_config_dict["taylor_offset"],
            gaussian_mean=self.model_config_dict["noise_mean"],
            gaussian_var=self.model_config_dict["noise_std"],
            gaussian_scale=self.model_config_dict["noise_scale"],
            checkpoint_path=self.model_config_dict["kkt_hardnet_checkpoint_path"],
            save_checkpoint_iter=self.model_config_dict["save_checkpoint_iter"],
            is_linear = self.is_linear
        )
        self.kkt_trainer.train()

    def run(self, mode="mlp"):
        """mode can be 'mlp', 'pinn', or 'dae'."""
        self._create_dataset()
        self._create_dataloader()
        self.build_test_buckets_by_kkt(batch_size=32, shuffle=False)
        self.save_all_test_sets(save_csv=True)
        
        if mode.lower() == "mlp":
            self.train_mlp()
        elif mode.lower() == "pinn":
            self.train_pinn()
        elif mode.lower() == "dae":
            self.train_kkt_hardnet()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        pass
