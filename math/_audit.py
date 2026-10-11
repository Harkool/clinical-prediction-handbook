"""Packaging audit runner. Use the repository run.py for the reader workflow."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).absolute().parents[1]


def execute(entry, mode, output, timeout=600):
    """Every attempted run replaces previous acceptance, even if the child fails."""
    if mode not in ("quick", "book"):
        raise ValueError("mode must be quick or book")
    target = output / entry['id']
    target.mkdir(parents=True, exist_ok=True)
    report = target / "result.json"
    report.write_text(json.dumps(dict(id=entry['id'],mode=mode,status="RUNNING"))+"\n",
                      encoding="utf-8")
    (target / "stdout.txt").write_text("",encoding="utf-8")
    (target / "stderr.txt").write_text("",encoding="utf-8")
    environment = dict(os.environ, CPH_MODE=mode, CPH_WORKERS="1", PYTHONIOENCODING="utf-8",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    start = time.monotonic()
    status = dict(id=entry['id'], mode=mode, exit_code=1, checks=None, status="FAIL")
    try:
        script = ROOT / entry['script']
        completed = subprocess.run([sys.executable,str(script)],cwd=target,env=environment,
            capture_output=True,encoding="utf-8",timeout=timeout)
        (target / "stdout.txt").write_text(completed.stdout,encoding="utf-8")
        (target / "stderr.txt").write_text(completed.stderr,encoding="utf-8")
        payload = json.loads(report.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Child report must be a JSON object")
        checks = payload.get('checks')
        if (completed.returncode or payload.get('id') != entry['id'] or
                payload.get('mode') != mode or not isinstance(checks,dict) or
                not checks or not all(value is True for value in checks.values())):
            raise ValueError("Child did not produce this run's valid acceptance report")
        payload.update(status="PASS", script_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
        report.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        status.update(exit_code=0,checks=checks,status="PASS")
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        status['error'] = str(error)
        if isinstance(error,subprocess.TimeoutExpired):
            for filename,value in (("stdout.txt",error.stdout),("stderr.txt",error.stderr)):
                if value:
                    (target / filename).write_text(value.decode('utf-8',errors='replace') if isinstance(value,bytes) else value,encoding="utf-8")
        report.write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    status['seconds'] = round(time.monotonic()-start,3)
    print(json.dumps(status,ensure_ascii=False),flush=True)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("quick","book"),nargs="?",default="quick")
    parser.add_argument("--output", help="Directory below this repository's outputs/ tree")
    args = parser.parse_args(argv)
    output = Path(os.path.abspath(args.output)) if args.output else ROOT / "outputs" / args.mode / "math"
    try:
        output.relative_to(ROOT / "outputs")
    except ValueError:
        parser.error("--output must stay below the repository outputs/ directory")
    output.mkdir(parents=True,exist_ok=True)
    acceptance = output / "acceptance.json"
    acceptance.write_text(json.dumps(dict(mode=args.mode,status="RUNNING"))+"\n",encoding="utf-8")
    entries = json.loads((ROOT / "math" / "catalog.json").read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda entry: execute(entry,args.mode,output), entries))
    import numpy, scipy, sklearn
    passed = all(item['status']=="PASS" for item in results)
    acceptance.write_text(json.dumps(dict(mode=args.mode,status="PASS" if passed else "FAIL",
        python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,
        sklearn=sklearn.__version__,results=results),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
