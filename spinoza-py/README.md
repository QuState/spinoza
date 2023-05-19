# Spinoza-Py

### Build from Source

1. Create and activate a python virtual environment
```bash
cd spinoza-py
python -m venv .env
source .env/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements-dev.txt
```

3. Build wheels
```bash
maturin build --release
```
Note: This command will build a wheel file and store it in a directory called
target/wheels (the path to the file will also be in the command's output).

4. Install the wheel file
```bash
pip install <path-to-wheel>
```

For example:
```bash
pip install ../target/wheels/spinoza_py-*.whl
```


### Examples

```python
import numpy as np
from spinoza_py import QuantumCircuit, QuantumRegister, show_table

def val_encoding(n, v):
    q = QuantumRegister(n)
    qc = QuantumCircuit(q)

    for i in range(n):
        qc.h(i)

    for i in range(n):
        qc.p(2 * np.pi / (2 ** (i + 1)) * v, i)

    qc.iqft(range(n)[::-1])

    qc.execute()
    return qc.get_statevector()


if __name__ == "__main__":
    state = value_encoding(4, 2.4)
    show_table(state)
```
