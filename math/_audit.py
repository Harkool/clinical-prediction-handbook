"""Packaging audit runner. Use the repository run.py for the reader workflow."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).absolute().parents[1]
MODE = sys.argv[1] if len(sys.argv)>1 else "quick"
OUT = ROOT / "outputs" / MODE / "math"
entries = json.loads((ROOT / "math" / "catalog.json").read_text(encoding="utf-8"))
environment = dict(os.environ, CPH_MODE=MODE, CPH_WORKERS="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def execute(entry):
    target = OUT / entry['id']
    target.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    completed = subprocess.run([sys.executable,str(ROOT / entry['script'])],cwd=target,env=environment,capture_output=True,encoding="utf-8",timeout=600)
    (target / "stdout.txt").write_text(completed.stdout,encoding="utf-8")
    (target / "stderr.txt").write_text(completed.stderr,encoding="utf-8")
    payload = json.loads((target / "result.json").read_text(encoding="utf-8")) if (target / "result.json").exists() else None
    status = dict(id=entry['id'],exit_code=completed.returncode,seconds=round(time.monotonic()-start,3),checks=payload['checks'] if payload else None)
    print(json.dumps(status,ensure_ascii=False),flush=True)
    if completed.returncode:
        print(completed.stderr,flush=True)
    return status


with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(execute,entries))
import numpy, scipy, sklearn
(OUT / "acceptance.json").write_text(json.dumps(dict(mode=MODE,python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,results=results),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
if any(item['exit_code'] or not item['checks'] or not all(item['checks'].values()) for item in results):
    raise SystemExit(1)
