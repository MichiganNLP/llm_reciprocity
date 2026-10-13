import ast
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    for path in ROOT.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    with tempfile.TemporaryDirectory(prefix='reciprocity-check-') as output:
        subprocess.run([sys.executable, '-B', str(ROOT/'regan.py'), '--backend', 'mock',
                        '--runs-per-cell', '1', '--output-dir', output], check=True)
    print('Syntax and mock test passed; no inference or API calls.')
