import re
from sympy import Symbol
from KKT.creator import KKTSystemCreator
from KKT.kkt_combination import KKT_combination_creator
from run.utils import categorize

# class NewtonRunner:
#     def __init__(self, parameters, variables, binary_variables, objective, constraints,
#                  formulation_index):
#         self.parameters = parameters
#         self.variables = variables
#         self.binary_variables = binary_variables
#         self.problem_variables = variables + binary_variables
#         self.objective = objective
#         self.constraints = constraints
#         self.index = formulation_index
#         self.creator = None
#         self.residuals = None
#         self.create_kkt_system()
#         self.formulate_kkt_system()
#         print(self.creator.model.kkt_variables)
#         self.variables = sorted(
#             self.creator.model.kkt_variables,
#             key=lambda sym: self.creator.model.symbol_map[str(sym)]
#         )

#         self.parameters = sorted(
#             self.creator.model.parameters,
#             key=lambda sym: self.creator.model.symbol_map[str(sym)]
# )
#         self.residuals = self.creator.model.residuals
        
        
#     def create_kkt_system(self):
#         self.creator = KKTSystemCreator()
#         self.creator.add_parameters(self.parameters)
#         self.creator.add_variables(self.variables)
#         self.creator.add_binary_variables(self.binary_variables)
#         self.creator.add_objective(self.objective)
#         self.creator.add_constraints(self.constraints)

#     def formulate_kkt_system(self):
#         self.creator.formulate(self.index)


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


class NewtonRunner:
    def __init__(self, 
                 parameters, 
                 variables, 
                 binary_variables, 
                 objective, 
                 constraints,
                 ICs,
                 BCs,
                 formulation_index,
                 taylor_offset,
                 taylor_order="auto",
                 step_length=1.0, 
                 tol=1e-12, 
                 reg_factor=1e-8, 
                 max_iter=1000):
        self.parameters = parameters
        self.variables = variables
        self.binary_variables = binary_variables
        self.problem_variables = variables + binary_variables
        self.objective = objective
        self.constraints = constraints
        self.ICs = ICs
        self.BCs = BCs
        self.index = formulation_index
        self.step_length = step_length
        self.tol = tol
        self.taylor_offset = taylor_offset
        self.taylor_order = taylor_order
        self.reg_factor = reg_factor
        self.max_iter = max_iter
        self.creator = None
        self.kkt_combinations = {}
        self.kkt_combinations_sympy = {}
        self.create_kkt_system()
        self.formulate_kkt_system()
        self.formulate_kkt_combination_system()
        
        # Filter only sympy.Symbols before sorting
        self.variables = sorted(
            [v for v in self.creator.model.kkt_variables if isinstance(v, Symbol)],
            key=categorize
        )

        self.parameters = sorted(
            [self.creator.model.symbol_map[p] for p in self.creator.model.parameters if isinstance(self.creator.model.symbol_map[p], Symbol)],
            key=categorize
        )

        self.number_of_parameters = self.creator.model.number_of_parameters
        self.parameters_list = self.parameters[:self.number_of_parameters]
        self.variables_list = self.variables[:]

    def create_kkt_system(self):
        self.creator = KKTSystemCreator(order=self.taylor_order, offset=self.taylor_offset)
        self.creator.add_parameters(self.parameters)
        self.creator.add_variables(self.variables)
        self.creator.add_binary_variables(self.binary_variables)
        self.creator.add_objective(self.objective)
        self.creator.add_constraints(self.constraints)
        
    def formulate_kkt_system(self):
        self.creator.formulate(self.index)
        # self.creator.print_kkt_system()
        # self.creator.print_kkt_variables()
        
    def formulate_kkt_combination_system(self):
        self.KKT = KKT_combination_creator(objective=self.objective, constraints = self.constraints, parameters = self.parameters,
                                                        variables = self.variables, binary_variables = self.binary_variables,
                                                        ICs = self.ICs, BCs = self.BCs, formulation_index = self.index, taylor_offset = self.taylor_offset, taylor_order=self.taylor_order)
        self.kkt_combinations = self.KKT.kkt_combinations
        self.kkt_combinations_sympy = self.KKT.kkt_combinations_sympy

