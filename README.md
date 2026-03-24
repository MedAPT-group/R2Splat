# gsplat2ct

3D Gaussian Splatting to CT with Second-Order Optimization

## Project Structure

- `gsplat2ct/`: Main package with CUDA extensions
- `tests/`: Unit tests and integration tests
- `scripts/`: Training and benchmarking scripts

## Installation

```bash
cd gsplat2ct
pip install -e .
```

## Usage

```python
from gsplat2ct import SimpleMLP, NewtonOptimizer

model = SimpleMLP(input_dim=3, hidden_dim=4, output_dim=2).cuda()
optimizer = NewtonOptimizer(model.parameters(), damping=1e-3)
```
