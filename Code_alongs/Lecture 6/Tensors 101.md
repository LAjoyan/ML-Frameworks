## Tensors 101

This notebook is a quick, practical introduction to tensors in PyTorch.

### Goals

* Create tensors from Python data and from random generators
* Inspect shapes and dtypes
* Perform basic operations and broadcasting
* Understand how gradients attach to tensors

```
import torch

torch.manual_seed(0)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.zeros((2, 3))
c = torch.randn((2, 3))

a, b.shape, c.dtype
```
     
### Shapes, indexing, and reshaping

```
x = torch.arange(12).reshape(3, 4)
x, x[0], x[:, 1]
```

### Operations and broadcasting

```
v = torch.tensor([1.0, 2.0, 3.0])
m = torch.ones((2, 3))
m + v
```

### Autograd basics

Tensors can track gradients when requires_grad=True.

```
w = torch.tensor([2.0, -1.0], requires_grad=True)
y = (w ** 2).sum()
y.backward()
w.grad
```     