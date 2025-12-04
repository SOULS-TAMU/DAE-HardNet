from typing import List
from KKT.formulation1 import Formulation1
from KKT.formulation2 import Formulation2
from KKT.formulation3 import Formulation3


class KKTSystemCreator:
    def __init__(self, order, offset):
        self.parameters = None
        self.variables = None
        self.objective = None
        self.binary_variables = None
        self.constraints = None
        self.model = None
        self.offset = offset
        self.order = order

    def add_parameters(self, parameters: List[str]):
        self.parameters = parameters

    def add_variables(self, variables: List[str]):
        self.variables = variables

    def add_binary_variables(self, binary_variables: List[str]):
        self.binary_variables = binary_variables

    def add_objective(self, objective: str):
        self.objective = objective

    def add_constraints(self, constraints: List[str]):
        self.constraints = constraints

    def get_formulation1(self):
        self.model = Formulation1(self.objective, self.constraints,
                                  self.parameters, self.variables, self.binary_variables, self.offset, self.order)
        self.model.formulate()

    def get_formulation2(self):
        self.model = Formulation2(self.objective, self.constraints,
                                  self.parameters, self.variables, self.binary_variables, self.offset, self.order)
        self.model.formulate()

    def get_formulation3(self):
        self.model = Formulation3(self.objective, self.constraints,
                                  self.parameters, self.variables, self.binary_variables, self.offset, self.order)
        self.model.formulate()

    def formulate(self, index=1):
        if index == 1:
            self.get_formulation1()
        elif index == 2:
            self.get_formulation2()
        elif index == 3:
            self.get_formulation3()

    def print_kkt_system(self):
        print("\n=== KKT System ===")
        for eq in self.model.kkt_system:
            print(eq)

    def print_kkt_variables(self):
        print("\n=== KKT Variables ===")
        print(self.model.kkt_variables)
        
    def print_parameters(self):
        print("\n=== Parameters ===")
        print(self.model.parameters)
        
    def print_objective(self):
        print("\n=== Objective ===")
        print(self.model.objective)
        
    def print_constraints(self):
        print("\n=== Constraints ===")
        print(self.model.constraints_list)
