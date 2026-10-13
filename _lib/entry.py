import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def launch(module, arguments=None):
    path = ROOT / '_lib' / 'runners' / (module + '.py')
    sys.path.insert(0, str(path.parent))
    os.chdir(ROOT)
    sys.argv = [str(path), *(sys.argv[1:] if arguments is None else arguments)]
    runpy.run_path(str(path), run_name='__main__')


def choose(tasks):
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', choices=tasks)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    launch(tasks[args.experiment], args.arguments)
