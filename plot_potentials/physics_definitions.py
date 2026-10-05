import ast
from types import SimpleNamespace
import numpy as np

def load_physics_definitions(PHYSICS_STAGE, SELECTED_NAMES) -> dict[str, object]:
    """Execute only the potential-definition nodes, not the training pipeline."""
    source = PHYSICS_STAGE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PHYSICS_STAGE))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            nodes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            names = {
                target.id
                for target in getattr(node, "targets", [])
                if isinstance(target, ast.Name)
            }
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            if names & SELECTED_NAMES:
                nodes.append(node)
    namespace: dict[str, object] = {
        "__name__": "__main__",
        "CFG": SimpleNamespace(k_states=11, kinetic_coefficient=0.5),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PHYSICS_STAGE), "exec"), namespace)
    return namespace


def visible_limits(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [0.005, 0.995])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower, upper = float(np.min(values)), float(np.max(values))
    padding = max((upper - lower) * 0.08, 0.1)
    return float(lower - padding), float(upper + padding)