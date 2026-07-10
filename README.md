# chess-again

If you are interested in what's behind this project, check out my 
[blog post here](https://patrickmastorga.github.io/blog/2026/07/09/chess-again/)

## Quick Start

This repository builds a native Rust executable and a Python extension module from the same root
directory.

Clone the repository, then set up the local Python environment and install the Rust Python build
tool:

```bash
uv venv
source .venv/bin/activate
uv pip install maturin
```

Build and install the Python module into the active virtual environment:

```bash
maturin develop
```

Run the native Rust executable:

```bash
cargo run
```

Run the Python entry script against the installed extension:

```bash
python scripts/test.py
```

