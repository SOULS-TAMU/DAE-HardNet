"""Check package/version inventory, local Markdown links and workflow structure."""
import ast
from pathlib import Path
import re
import tomllib
from urllib.parse import unquote
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    metadata = tomllib.loads((ROOT/"pyproject.toml").read_text())
    tree = ast.parse((ROOT/"src/__init__.py").read_text())
    versions = [node.value.value for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)]
    assert versions == [metadata["project"]["version"]], "Source/metadata version mismatch"
    citation = yaml.safe_load((ROOT/"CITATION.cff").read_text())
    assert str(citation["version"]) == versions[0], "Citation version mismatch"
    packages = {"daehn" + ("."+".".join(p.parent.relative_to(ROOT/"src").parts) if p.parent != ROOT/"src" else "") for p in (ROOT/"src").rglob("__init__.py")}
    assert packages == set(metadata["tool"]["setuptools"]["packages"]), "Missing/unexpected packaged module"
    for path in [*ROOT.glob("*.md"), *(ROOT/"docs").glob("*.md")]:
        for raw in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            target = raw.split("#", 1)[0]
            if not target or ":" in target or target.startswith("//"):
                continue
            assert (path.parent/unquote(target)).exists(), f"Broken local link in {path.name}: {target}"
    for path in (ROOT/".github/workflows").glob("*.yml"):
        data = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        assert all(key in data for key in ["name", "on", "permissions", "jobs"]), path
        for job in data["jobs"].values():
            assert "runs-on" in job and "steps" in job, path
    print("Package inventory, version/citation, local links and workflow structure passed.")


if __name__ == "__main__":
    main()
