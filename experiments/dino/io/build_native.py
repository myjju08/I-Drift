"""Build the original SSE2 color codec without downloading dependencies."""
from pathlib import Path
import argparse
import os
import platform
import shlex
import subprocess
import tempfile


def build(output=None, compiler=None):
    root = Path(__file__).resolve().parent
    output = Path(output or root / 'native_color_decode.so').resolve()
    if platform.machine().lower() not in {'x86_64', 'amd64', 'i386', 'i686'}:
        raise RuntimeError('The native codec requires x86 SSE2; use --decode-backend numpy on other CPUs')
    output.parent.mkdir(parents=True, exist_ok=True)
    command = shlex.split(compiler or os.environ.get('CC', 'cc'))
    with tempfile.TemporaryDirectory(prefix='.codec-build-', dir=output.parent) as directory:
        candidate = Path(directory) / output.name
        subprocess.run(command + ['-O3', '-shared', '-fPIC', '-msse2',
                       str(root / 'native_color_decode.c'), '-o', str(candidate)], check=True)
        os.replace(candidate, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compiler')
    args = parser.parse_args()
    print(build(args.output, args.compiler))


if __name__ == '__main__':
    main()
