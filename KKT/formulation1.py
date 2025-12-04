import sympy as sp
from sympy import Symbol
from typing import List

class Formulation1:
    def __init__(self, objective: str, constraints: List[str], parameters: List[str], variables: List[str],
                 binary_variables: List[str], offset: float, order):
        self.objective_str = objective
        self.constraints_list = constraints[:]  # raw constraint strings
        self.parameters = parameters[:]         # user-defined parameters
        self.variables = variables[:]           # decision variables
        self.binary_variables = binary_variables[:]
        self.delta = offset
        self.order = order

        # containers
        self.symbol_map = {}
        self.objective = None
        self.original_constraints = []          # constraints in terms of y
        self.expanded_constraints = []          # taylor-expanded constraints
        self.equality_constraints = []
        self.inequality_constraints = []
        self.lagrangian = None
        self.kkt_variables = []
        self.kkt_system = []
        self.residuals = []
        self.slack_syms = []
        self.additional_variables = []          # λ, μ, s
        self.is_linear = True
        self.number_of_parameters = len(self._parameter_names())
        
        if self.order == "auto":
            self.derivative_order = self.max_derivative_order()
        else:
            self.derivative_order = self.order
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _parameter_names(self):
        """Independent parameters (exclude *_data, *_delta)."""
        return [p for p in self.parameters if not (p.endswith("_data") or p.endswith("_delta"))]

    def _initialize_symbols(self):
        """Create all symbols: parameters, y_data, per-parameter y_*_delta, derivatives (1st/2nd), derivative_data."""
        # parameters first
        for p in self.parameters:
            if p not in self.symbol_map:
                self.symbol_map[p] = Symbol(p, real=True)

        y_vars = [v for v in self.variables if v.startswith("y")]
        P = self._parameter_names()  # user parameters (exclude *_data/_delta)

        for y in y_vars:
            # y itself
            if y not in self.symbol_map:
                self.symbol_map[y] = Symbol(y, real=True)

            # y_data
            pname = f"{y}_data"
            if pname not in self.parameters:
                self.parameters.append(pname)
            if pname not in self.symbol_map:
                self.symbol_map[pname] = Symbol(pname, real=True)

            # --- per-parameter deltas y_{param}_delta (e.g., y1_t_delta, y1_x1_delta) ---
            for p in P:
                yp_delta = f"{p}_{y}_delta"
                if yp_delta not in self.parameters:
                    self.parameters.append(yp_delta)
                if yp_delta not in self.symbol_map:
                    self.symbol_map[yp_delta] = Symbol(yp_delta, real=True)

            # --- 1st-order derivatives wrt each parameter ---
            for p in P:
                dname = f"d1{y}d{p}"
                if dname not in self.variables:
                    self.variables.append(dname)
                if dname not in self.symbol_map:
                    self.symbol_map[dname] = Symbol(dname, real=True)

                # # derivative data
                # ddata = f"{dname}_data"
                # if ddata not in self.parameters:
                #     self.parameters.append(ddata)
                # if ddata not in self.symbol_map:
                #     self.symbol_map[ddata] = Symbol(ddata, real=True)

            # --- 2nd-order derivatives if requested (ONLY pure seconds; no mixed) ---
            if getattr(self, "derivative_order", 1) >= 2:
                for p in P:
                    d2name = f"d2{y}d{p}"  # e.g., d2y1dt, d2y1dx1
                    if d2name not in self.variables:
                        self.variables.append(d2name)
                    if d2name not in self.symbol_map:
                        self.symbol_map[d2name] = Symbol(d2name, real=True)

                    # d2data = f"{d2name}_data"
                    # if d2data not in self.parameters:
                    #     self.parameters.append(d2data)
                    # if d2data not in self.symbol_map:
                    #     self.symbol_map[d2data] = Symbol(d2data, real=True)

                    # alias support for legacy doubled form (e.g., d2y1dx1dx1)
                    # legacy = f"d2{y}d{p}"
                    # # legacy_data = f"{legacy}_data"
                    # if legacy not in self.symbol_map:
                    #     self.symbol_map[legacy] = self.symbol_map[d2name]
                    # # if legacy_data not in self.symbol_map:
                    #     self.symbol_map[legacy_data] = self.symbol_map[d2data]

        # binary vars
        for b in self.binary_variables:
            if b not in self.symbol_map:
                self.symbol_map[b] = Symbol(b, real=True)

        # derivatives defined directly as variables (external definitions)
        d_vars = [v for v in self.variables if v.startswith("d")]
        for d in d_vars:
            if d not in self.symbol_map:
                self.symbol_map[d] = Symbol(d, real=True)
            # ddata = f"{d}_data"
            # if ddata not in self.parameters:
            #     self.parameters.append(ddata)
            # if ddata not in self.symbol_map:
            #     self.symbol_map[ddata] = Symbol(ddata, real=True)


    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    def _parse_original_constraints(self):
        """Keep constraints in original form (in terms of y)."""
        self.original_constraints = []
        self.orig_eq_violation = []   # <--- NEW

        for c in self.constraints_list:
            expr = sp.sympify(c, evaluate=False, locals=self.symbol_map)
            self.original_constraints.append(expr)

            # also build orig_eq_violation here
            if isinstance(expr, sp.Equality):
                self.orig_eq_violation.append(expr.lhs - expr.rhs)
            # elif isinstance(expr, (sp.LessThan, sp.StrictLessThan)):
            #     self.orig_eq_violation.append(expr.lhs - expr.rhs)
            # elif isinstance(expr, (sp.GreaterThan, sp.StrictGreaterThan)):
            #     self.orig_eq_violation.append(expr.rhs - expr.lhs)
            # else:
            #     raise ValueError(f"Unsupported constraint: {expr}")
            
        # ------------------------------------------------------------------
        self.orig_ineq_violation = []  # <--- NEW
        for c in self.constraints_list:
            expr = sp.sympify(c, evaluate=False, locals=self.symbol_map)
            if isinstance(expr, (sp.LessThan, sp.StrictLessThan)):
                self.orig_ineq_violation.append(expr.lhs - expr.rhs)
            elif isinstance(expr, (sp.GreaterThan, sp.StrictGreaterThan)):
                self.orig_ineq_violation.append(expr.rhs - expr.lhs)


    def _expand_constraints(self):
        """Use averaged per-parameter update:
        y = (1/|P|) * ( Σ_p y_{p}_delta  -  Σ_p d1y/dp * δ  -  1/2 Σ_p d2y/dp^2 * δ^2 )
        (no mixed partials)."""
        subs_map = {}
        y_vars = [v for v in self.variables if v.startswith("y")]
        P = self._parameter_names()
        nP = len(P)

        y_taylor_list = []

        for y in y_vars:
            if nP == 0:
                expansion = self.symbol_map[y]  # fallback
            else:
                sum_y_deltas = sum(self.symbol_map[f"{p}_{y}_delta"] for p in P)
                sum_d1 = sum(self.symbol_map[f"d1{y}d{p}"] * self.delta for p in P)
                delta2 = (self.delta ** 2)
                sum_d2 = sum(self.symbol_map.get(f"d2{y}d{p}", 0) for p in P)

                expansion = (sum_y_deltas - sum_d1 - sp.Rational(1, 2) * sum_d2 * delta2) / sp.Integer(nP)

            subs_map[self.symbol_map[y]] = expansion
            y_taylor_list.append(expansion)

        # column vector of expansions
        self.y_taylor_expr = sp.Matrix(y_taylor_list)

        # substitute into constraints
        self.expanded_constraints = [expr.xreplace(subs_map) for expr in self.original_constraints]

        # split into equalities/inequalities & build violations
        self.equality_constraints, self.inequality_constraints = [], []
        self.eq_violation, self.ineq_violation = [], []

        for expr in self.expanded_constraints:
            if isinstance(expr, sp.Equality):
                self.equality_constraints.append(expr)
                self.eq_violation.append(expr.lhs - expr.rhs)
            elif isinstance(expr, (sp.LessThan, sp.StrictLessThan)):
                ineq = expr.lhs - expr.rhs
                self.inequality_constraints.append(ineq)
                self.ineq_violation.append(ineq)
            elif isinstance(expr, (sp.GreaterThan, sp.StrictGreaterThan)):
                ineq = expr.rhs - expr.lhs
                self.inequality_constraints.append(ineq)
                self.ineq_violation.append(ineq)
            else:
                raise ValueError(f"Unsupported constraint: {expr}")





    # def _expand_constraints(self):
    #     """Expand y in constraints using Taylor expansion."""
    #     subs_map = {}
    #     y_vars = [v for v in self.variables if str(v).startswith("y")]  # handles str or Symbol

    #     for y in y_vars:
    #         expansion = self.symbol_map[f"{y}_data"] + self.symbol_map[f"{y}_delta"]
    #         for p in self._parameter_names():
    #             expansion += self.symbol_map[f"d1{y}d{p}"] * self.delta

    #         # map y itself → expansion
    #         subs_map[sp.Symbol(str(y))] = expansion

    #     self.expanded_constraints = [expr.xreplace(subs_map) for expr in self.original_constraints]

    #     self.equality_constraints, self.inequality_constraints = [], []
    #     for expr in self.expanded_constraints:
    #         if isinstance(expr, sp.Equality):
    #             self.equality_constraints.append(expr)
    #         elif isinstance(expr, (sp.LessThan, sp.StrictLessThan)):
    #             self.inequality_constraints.append(expr.lhs - expr.rhs)
    #         elif isinstance(expr, (sp.GreaterThan, sp.StrictGreaterThan)):
    #             self.inequality_constraints.append(expr.rhs - expr.lhs)
    #         else:
    #             raise ValueError(f"Unsupported constraint: {expr}")




    def _add_additional_constraints(self):
        """Binary variable constraints."""
        for b in self.binary_variables:
            self.inequality_constraints.append(self.symbol_map[b] * (self.symbol_map[b] - 1))

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def _build_objective(self):
        if self.objective_str not in (None, "", "auto"):
            self.objective = sp.sympify(self.objective_str, evaluate=False, locals=self.symbol_map)
            return

        terms = []
        P = self._parameter_names()
        nP = len(P)

        for y in [v for v in self.variables if v.startswith("y")]:
            if nP == 0:
                approx_y = self.symbol_map[y]
            else:
                sum_y_deltas = sum(self.symbol_map[f"{p}_{y}_delta"] for p in P)
                sum_d1 = sum(self.symbol_map[f"d1{y}d{p}"] * self.delta for p in P)
                delta2 = (self.delta ** 2)
                sum_d2 = sum(self.symbol_map.get(f"d2{y}d{p}", 0) for p in P)
                approx_y = (sum_y_deltas - sum_d1 - sp.Rational(1, 2) * sum_d2 * delta2) / sp.Integer(nP)

            incr = approx_y - self.symbol_map[f"{y}_data"]
            terms.append(incr ** 2)

        self.objective = sp.Rational(1, 2) * sp.Add(*terms) if terms else sp.Integer(0)



    # ------------------------------------------------------------------
    # Lagrangian and KKT
    # ------------------------------------------------------------------
    def _build_lagrangian_and_kkt(self):
        self.lagrangian = self.objective

        # Equality constraints with λ
        for i, eq in enumerate(self.equality_constraints):
            lam = sp.Symbol(f"lambda_{i+1}", real=True)
            self.symbol_map[f"lambda_{i+1}"] = lam
            self.lagrangian += lam * (eq.lhs - eq.rhs)
            self.additional_variables.append(f"lambda_{i+1}")

        # Inequality constraints with μ and slack
        for j, ineq in enumerate(self.inequality_constraints):
            mu = sp.Symbol(f"mu_{j+1}", real=True)
            s = sp.Symbol(f"s_{j+1}", real=True)
            self.symbol_map[f"mu_{j+1}"] = mu
            self.symbol_map[f"s_{j+1}"] = s
            self.lagrangian += mu * (ineq + s)
            self.additional_variables.extend([f"mu_{j+1}", f"s_{j+1}"])
            self.slack_syms.append(s)

        # Collect KKT variables
        vars_ = [self.symbol_map[v] for v in self.variables + self.binary_variables]
        mults_ = [self.symbol_map[v] for v in self.additional_variables]
        self.kkt_variables = vars_ + mults_

        # Stationarity
        primal_vars_no_y = [v for v in self.variables if not v.startswith("y")]
        stationarity_eqns = []
        for v in primal_vars_no_y:
            deriv = sp.diff(self.lagrangian, self.symbol_map[v])
            stationarity_eqns.append(sp.Eq(deriv, 0))

        # Primal feasibility
        primal_eqns = [sp.Eq(eq.lhs - eq.rhs, 0) for eq in self.equality_constraints]
        for ineq, s in zip(self.inequality_constraints, self.slack_syms):
            primal_eqns.append(sp.Eq(ineq + s, 0))

        # Complementarity (μ*s = 0 smoothed)
        mu_syms = [self.symbol_map[v] for v in self.additional_variables if v.startswith("mu")]
        compl_eqns = [sp.Eq(mu + s - sp.sqrt(mu**2 + s**2), 0) for mu, s in zip(mu_syms, self.slack_syms)]

        self.kkt_system = stationarity_eqns + primal_eqns + compl_eqns
        self.residuals = sp.Matrix([eq.lhs for eq in self.kkt_system])

    # Max derivative order
    def max_derivative_order(self):
        """
        Check variables starting with 'd' and return the largest number following 'd'.
        Example: ['d1y1dx1', 'd2y3dx4'] → returns 2
        """
        max_order = 0
        for v in self.variables:
            if v.startswith("d"):
                # extract consecutive digits right after 'd'
                i = 1
                digits = ""
                while i < len(v) and v[i].isdigit():
                    digits += v[i]
                    i += 1
                if digits:
                    order = int(digits)
                else:
                    order = 1   # if it's just like "dy1dx1"
                max_order = max(max_order, order)
        return max_order

    # second order derivative variable name
    def _d2name(self, y: str, p: str, q: str) -> str:
        """
        Canonical 2nd-derivative symbol name.
        - If p == q:  d2{y}d{p}       (e.g., d2y1dt, d2y1dx1)
        - If p != q:  d2{y}d{a}d{b}   (sorted, e.g., d2y1dtdx1)
        """
        a, b = sorted([p, q])
        if a == b:
            return f"d2{y}d{a}"
        return f"d2{y}d{a}d{b}"


    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def formulate(self):
        self._initialize_symbols()
        self._parse_original_constraints()
        self._expand_constraints()
        self._build_objective()
        self._add_additional_constraints()
        self._build_lagrangian_and_kkt()

        # print(f"Objective: {self.objective}\n")
        # print(f"Original Constraints: {self.original_constraints}\n")
        # print(f"Expanded Constraints: {self.expanded_constraints}\n")
        # print(f"Equality Constraints: {self.equality_constraints}\n")
        # print(f"Inequality Constraints: {self.inequality_constraints}\n")
        # print(f"Lagrangian: {self.lagrangian}\n")
        # print(f"KKT System: {self.kkt_system}\n")
        # print(f"Derivative Order: {self.derivative_order}\n")
        # print(f"y_taylor_expr: {self.y_taylor_expr}\n")
        # print(f"variables: {self.variables}\n")
        # print(f"parameters: {self.parameters}\n")