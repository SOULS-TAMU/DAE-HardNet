from KKT.creator import KKTSystemCreator
import itertools
import re
import sympy as sp
from copy import deepcopy


def categorize(sym):
    name = str(sym)

    # Pure y variables like y1, y2
    if re.fullmatch(r"y\d+", name):
        return (0, 0, name)

    # Spatial derivatives: d...dx...
    elif re.fullmatch(r"(d\d*)y\d+dx\d+", name) or re.fullmatch(r"dy\d+dx\d+", name):
        return (1, 0, name)

    # Time derivatives: d...dt...
    elif re.fullmatch(r"(d\d*)y\d+dt\d*", name) or re.fullmatch(r"dy\d+dt\d*", name):
        return (1, 1, name)

    # Other y-prefixed variables
    elif name.startswith("y") and not name.endswith("data"):
        return (2, 0, name)

    elif name.startswith("mu"):
        return (3, 0, name)
    elif name.startswith("s"):
        return (4, 0, name)
    elif name.startswith("delta"):
        return (5, 0, name)
    elif name.startswith("sigma"):
        return (6, 0, name)
    elif name.startswith("lambda"):
        return (7, 0, name)
    elif name.startswith("t"):
        return (8, 0, name)
    elif name.startswith("x"):
        return (9, 0, name)
    elif name.endswith("data"):
        return (10, 0, name)
    else:
        return (11, 0, name)

    

class KKT_combination_creator:
    def __init__(self, taylor_offset, taylor_order, objective=None, constraints=None, parameters=None, variables=None,
                 binary_variables=None, ICs=None, BCs=None, formulation_index=1):
        # Normalize inputs
        self.objective = objective
        self.constraints = list(constraints or [])
        self.parameters = list(parameters or [])
        self.variables = list(variables or [])
        self.binary_variables = list(binary_variables or [])
        self.ICs = list(ICs or [])
        self.BCs = dict(BCs or {})
        self.index = formulation_index
        self.taylor_offset = taylor_offset
        self.order = taylor_order

        self.kkt_combinations = {}
        self.kkt_combinations_sympy = {}
        self.build_all_kkt_combinations()

    # ---- helpers for BC enumeration ----
    def _enumerate_bc_selections(self):
        if not self.BCs:         # handles {} or None safely
            return [], []
        axes = sorted(self.BCs.keys())
        options_per_axis = []
        for ax in axes:
            vals = sorted(self.BCs[ax].keys(), key=lambda v: (self._try_float(v), v))
            options_per_axis.append([None] + vals)  # None = pick nothing
        return axes, options_per_axis

    @staticmethod
    def _try_float(s):
        try:
            return (0, float(s))
        except Exception:
            return (1, s)

    @staticmethod
    def _coerce_num(v):
        if v is None:
            return None
        try:
            f = float(v)
            return int(f) if f.is_integer() else f
        except Exception:
            return v

    def _collect_selected_bc_strings(self, selection: dict) -> list[str]:
        out = []
        for ax, val in selection.items():
            if val is None:
                continue
            out.extend(self.BCs[ax][val])
        return out

    def _build_creator_for(self, constraints: list[str]) -> KKTSystemCreator:
        cr = KKTSystemCreator(offset=self.taylor_offset, order=self.order)
        cr.add_parameters(self.parameters)
        cr.add_variables(self.variables)
        cr.add_binary_variables(self.binary_variables)
        cr.add_objective(self.objective)
        cr.add_constraints(constraints)
        cr.formulate(self.index)
        return cr

    def build_all_kkt_combinations(self) -> dict:
        axes, options_per_axis = self._enumerate_bc_selections()
        all_packs = {}

        # Only allow ICs=True when ICs actually exist
        use_ics_options = (False, True) if self.ICs else (False,)

        # If no axes, iterate once with an empty selection
        prod_iter = itertools.product(*options_per_axis) if axes else [()]

        for use_ics in use_ics_options:
            # IMPORTANT: re-create the iterator each time if using product
            iter_source = itertools.product(*options_per_axis) if axes else [()]
            for combo in iter_source:
                selection_raw = dict(zip(axes, combo)) if axes else {}
                selection_num = {ax: self._coerce_num(val) for ax, val in selection_raw.items()}

                used = []
                used.extend(self.constraints)

                ic_count = 0
                if use_ics:  # guaranteed self.ICs is non-empty here
                    used.extend(self.ICs)
                    ic_count = len(self.ICs)

                bc_list = self._collect_selected_bc_strings(selection_raw)
                used.extend(bc_list)

                only_constraints = (not use_ics) and (len(bc_list) == 0)
                if only_constraints:
                    key = "OG"
                else:
                    parts = [f"ICs={'True' if use_ics else 'False'}"]
                    for ax in sorted(selection_num.keys()):
                        val = selection_num[ax]
                        parts.append(f"{ax}={val if val is not None else 'None'}")
                    key = " | ".join(parts)

                # Build KKT
                creator = self._build_creator_for(used)
                kkt_eqs = [f"{eqn.lhs} == {eqn.rhs}" for eqn in creator.model.kkt_system]

                pack = {
                    "selection": deepcopy(selection_num),
                    "use_ics": use_ics,
                    "counts": {
                        "OG": len(self.constraints),
                        "ICs": ic_count,
                        "BCs": len(bc_list),
                        "total": len(used),
                    },
                    "kkt_system": kkt_eqs,
                }

                sympy_pack = {
                    "selection": deepcopy(selection_num),
                    "use_ics": use_ics,
                    "counts": {
                        "OG": len(self.constraints),
                        "ICs": ic_count,
                        "BCs": len(bc_list),
                        "total": len(used),
                    },
                    "kkt_system": creator.model.residuals,
                    "kkt_variables": sorted(creator.model.kkt_variables, key=categorize),
                    "parameters": sorted(creator.model.parameters, key=categorize),
                }
                all_packs[key] = pack
                self.kkt_combinations[key] = pack
                self.kkt_combinations_sympy[key] = sympy_pack


