import re
import torch.nn as nn
from model.newton import NewtonLayer



class Backbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, model_depth):
        super(Backbone, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers (repeat model_depth - 1 times)
        for _ in range(model_depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)
    

class NewtonModel(nn.Module):
    def __init__(self, residuals, taylor_exprs, orig_eq_violation, orig_ineq_violation, eq_violation, ineq_violation, variables, parameters, input_dim, config, taylor_offset=1e-6, is_kkt_model=False):
        super(NewtonModel, self).__init__()
        self.hidden_dims = config["hidden_dim"]
        self.step_length = config["newton_step_length"]
        self.tol = config["newton_tol"]
        self.reg_factor = config["newton_reg_factor"]
        self.max_newton_iter = config["max_newton_iter"]
        self.model_depth = config["model_depth"]
        self.taylor_offset = taylor_offset
        self.is_kkt_model = is_kkt_model
        self.taylor_exprs = taylor_exprs  # New addition for Taylor expansions

        # print(variables)

        # ===============================
        # Separate differential and non-differential variables
        # ===============================
        self.symbolic_vars = variables + parameters
        self.required_derivatives = []
        self.has_differential_terms = False
        self.max_diff_order = 0

        self.diff_variable_names = []
        self.non_diff_variable_names = []

        for name in variables:
            name = str(name)

            # Match first-order: dy1dx1 or dy1dt
            if re.fullmatch(r"dy\d+d[xt]\d*", name):
                self.has_differential_terms = True
                self.diff_variable_names.append(name)

                y_idx, wrt = re.findall(r"\d+|t", name)  # captures numbers + 't'
                target = f"y{y_idx}"
                wrt_var = "t" if wrt == "t" else f"x{wrt}"
                
                self.required_derivatives.append({
                    'target': target,
                    'order': 1,
                    'wrt': [wrt_var],
                    'symbol': name
                })
                self.max_diff_order = max(self.max_diff_order, 1)

            # Match higher-order: d2y3dx1dx2 or d2y1dt2
            elif match := re.fullmatch(r"d(\d+)y(\d+)(d[xt]\d*)+", name):
                self.has_differential_terms = True
                order = int(match.group(1))
                y_idx = int(match.group(2))
                diff_terms = re.findall(r"d([xt]\d*)", name)  # e.g. ['x1','t','t']

                wrt_vars = []
                for term in diff_terms:
                    wrt_vars.append("t" if term == "t" else f"x{term}")

                self.diff_variable_names.append(name)
                self.required_derivatives.append({
                    'target': f'y{y_idx}',
                    'order': order,
                    'wrt': wrt_vars,
                    'symbol': name
                })
                self.max_diff_order = max(self.max_diff_order, order)

            else:
                self.non_diff_variable_names.append(name)


        self.num_diff_terms = len(self.diff_variable_names)
        # print("Required derivatives: ", self.required_derivatives)
        # print("Non differentiable variable name: ", self.non_diff_variable_names)

        # ===============================
        # Define the neural network (only for non-differential vars)
        # ===============================
        # self.nn = nn.Sequential(
        #     nn.Linear(input_dim, self.hidden_dims),
        #     # nn.ReLU(),
        #     nn.Tanh(),
        #     nn.Linear(self.hidden_dims, self.hidden_dims),  # exclude differential vars
        #     # nn.ReLU(),
        #     nn.Tanh(),
        #     nn.Linear(self.hidden_dims, self.hidden_dims),  # exclude differential vars
        #     # nn.ReLU(),
        #     nn.Tanh(),
        #     nn.Linear(self.hidden_dims, len(self.non_diff_variable_names)),  # exclude differential vars
        #     # nn.ReLU()
        # )

        self.nn = Backbone(input_dim=input_dim, 
                           hidden_dim=self.hidden_dims, 
                           output_dim=len(self.non_diff_variable_names), 
                           model_depth=self.model_depth)
        
        # print("NN output dim variables: ", self.non_diff_variable_names)
        # print("NN output dim: ", len(self.non_diff_variable_names))

        # ===============================
        # Define the NewtonLayer
        # ===============================
        self.newton = NewtonLayer(
            residuals=residuals,
            taylor_exprs=taylor_exprs,
            orig_eq_violation=orig_eq_violation,
            orig_ineq_violation=orig_ineq_violation,
            eq_violation = eq_violation,
            ineq_violation= ineq_violation,
            variables=variables,
            parameters=parameters,
            step_length=self.step_length,
            taylor_offset=self.taylor_offset,
            tol=self.tol,
            reg_factor=self.reg_factor,
            max_iter=self.max_newton_iter
        )

    def forward(self, x):
        # Only predict non-differential vars
        y_nn = self.nn(x)  # (B, len(non_diff_variable_names))

        # You are expected to reconstruct full y vector here using autograd
        # Including manually computing required gradients and concatenating with y_nn

        # Call Newton layer with full vector
        y_tilde = self.newton(y_nn, x)
        return y_tilde
