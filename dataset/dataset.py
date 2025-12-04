import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class ModelDataset(Dataset):
    def __init__(self, param_df, var_df, obj_param_df, 
                 noise_std=0.0, noise_mean=0.0, noise_scale=0.1, 
                 apply_noise=0, num_points="all", save_path=None):
        
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.noise_scale = noise_scale
        self.apply_noise = apply_noise
        self.save_path = save_path
        
        assert len(param_df) == len(var_df), "Mismatched number of rows"
        
        self.param_df = param_df.copy()
        self.var_df = var_df.copy()
        # self.obj_param_df = obj_param_df.copy()
        
        self.x = torch.tensor(param_df.values, dtype=torch.float32)
        self.y = torch.tensor(var_df.values, dtype=torch.float32)
        self.obj_param_y = torch.tensor(obj_param_df.values, dtype=torch.float32)

        # 🔀 Random subsampling if num_points is specified
        if num_points != "all":
            print(f"🔀 Subsampling dataset to {num_points} points.")
            if isinstance(num_points, int) or (isinstance(num_points, str) and num_points.isdigit()):
                n = int(num_points)
                assert n <= len(self.x), "num_points exceeds dataset size"

                # generate random indices
                idx = torch.randperm(len(self.x))[:n]

                # select random subset
                self.x = self.x[idx]
                self.y = self.y[idx]
                self.obj_param_y = self.obj_param_y[idx]
                
                self.param_df = self.param_df.iloc[idx.numpy()].reset_index(drop=True)
                self.var_df = self.var_df.iloc[idx.numpy()].reset_index(drop=True)
                # self.obj_param_df = self.obj_param_df.iloc[idx.numpy()].reset_index(drop=True)
            else:
                raise ValueError("num_points should be 'all' or an integer value")
        
        # Optionally add Gaussian noise to targets
        if self.apply_noise and self.noise_std > 0:
            noise = torch.normal(mean=self.noise_mean, std=self.noise_std, size=self.y.shape)
            self.y = self.y + self.noise_scale * noise
            
            # update dataframe
            self.var_df = pd.DataFrame(self.y.numpy(), columns=self.var_df.columns)

            # save if path is provided
            if self.save_path:
                out_df = pd.concat([self.param_df, self.var_df], axis=1)
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                out_df.to_csv(self.save_path, index=False)
                # print(f"💾 Dataset with noise saved at: {self.save_path}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.obj_param_y[idx]
