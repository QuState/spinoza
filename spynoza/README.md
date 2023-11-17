# Spynoza

### Build from Source

1. Create and activate a python virtual environment
```bash
cd spynoza
python -m venv .env
source .env/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Build wheels
```bash
maturin build --release
```
> [!NOTE]
> This command will build a wheel file and store it the directory `target/wheels`.
> The correct path to the file will also be output to stdout.


4. Install the wheel file
```bash
pip install <path-to-wheel>
```
For example:
```bash
pip install ../target/wheels/spynoza-*.whl
```

### Try it out!
[examples](https://github.com/QuState/spinoza/tree/main/spynoza) can be run using:
```bash
python examples/<example-name>.py
```
