from typing import List, Dict, Iterable, Optional, Tuple
import re

class ICs_BCs:
    """
    Parse IC/BC strings without converting to SymPy expressions.
    - ICs: kept as a flat list[str]
    - BCs: nested dict like {"x1": {"0": ["y1 == ...", ...], "5": [...]}, "x2": {...}}
    """
    _LOC_PATTERN = re.compile(r"""
        ^\s*                # leading space
        (?P<axis>[A-Za-z_]\w*)   # axis name (e.g., x1)
        \s*=\s*
        (?P<value>.+?)      # everything after '=' (e.g., 0, 5, 3, 4)
        \s*$                # trailing space
    """, re.VERBOSE)

    def __init__(
        self,
        ic_list: Iterable[str],
        bc_blocks: Dict[str, Iterable[str]],
        *,
        validate: bool = False
    ):
        self.raw_ics: List[str] = list(ic_list or [])
        self.raw_bcs: Dict[str, List[str]] = {k: list(v) for k, v in (bc_blocks or {}).items()}

        self.ICs: List[str] = []
        # Desired shape: {"x1": {"0": [...], "5": [...]}, "x2": {"3": [...], "4": [...]} }
        self.BCs: Dict[str, Dict[str, List[str]]] = {}

        self.create(validate=validate)

    def create(self, *, validate: bool = False):
        self.parse_ICs(validate=validate)
        self.parse_BCs(validate=validate)

    # ---------- ICs ----------
    def parse_ICs(self, *, validate: bool = False):
        ics: List[str] = []
        for ic in self.raw_ics:
            ic_str = ic.strip()
            if not ic_str:
                raise ValueError("Empty initial condition string found.")
            if validate:
                self._validate_eq_string(ic_str)
            ics.append(ic_str)
        self.ICs = ics

    # ---------- BCs ----------
    def parse_BCs(self, *, validate: bool = False):
        grouped: Dict[str, Dict[str, List[str]]] = {}

        for _, lines in self.raw_bcs.items():
            for bc in lines:
                eq_str, axis, value = self.split_bc_string(bc)
                if validate:
                    self._validate_eq_string(eq_str)
                # Insert into nested structure
                grouped.setdefault(axis, {}).setdefault(value, []).append(eq_str)

        self.BCs = grouped

    def split_bc_string(self, bc: str) -> Tuple[str, str, str]:
        """
        Split 'EQ @ axis = value' into (eq_str, axis, value), keeping strings.
        Example: 'y1 == y1_in @ x1 = 0' -> ('y1 == y1_in', 'x1', '0')
        """
        if "@" not in bc:
            raise ValueError(f"Boundary condition must include '@': {bc}")

        eq_part, loc_part = bc.split("@", 1)
        eq_str = eq_part.strip()
        loc_str = loc_part.strip()

        if not eq_str:
            raise ValueError(f"Empty equation part before '@': {bc}")
        match = self._LOC_PATTERN.match(loc_str)
        if not match:
            raise ValueError(f"Location must be of form '<axis> = <value>': {bc}")

        axis = match.group("axis").strip()
        value = match.group("value").strip().replace(" ", "")  # normalize spaces in numeric/expression
        return eq_str, axis, value

    # ---------- Optional lightweight validation ----------
    @staticmethod
    def _validate_eq_string(eq: str):
        """
        Minimal check: ensure there's a '==' (or '=' used as equality).
        Adjust to your preferred syntax rules.
        """
        if "==" not in eq and "=" not in eq:
            raise ValueError(f"Equation appears malformed (no equality operator): '{eq}'")
