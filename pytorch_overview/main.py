import torch
import torch.nn.functional as F
from torch.autograd import grad

# ooo it can utilize fancy macs
print("M chips", torch.backends.mps.is_available())

tensor0d = torch.tensor(1)

tensor1d = torch.tensor([1, 2, 3])

tensor2d = torch.tensor([[1, 2], [3, 4]])

tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print("1d dtype", tensor1d.dtype)

floatvec = torch.tensor([1.0, 2.0, 3.0])
print("float dtype", floatvec.dtype)

print("2d tensor", tensor2d)
print("2d tensor shape: ", tensor2d.shape)  # 2x2 (RxC)
# compact way to .matmul
print("Compact matmul using @", tensor2d @ tensor2d.T)

print("Logistic regression forward pass below")
print()
print()

# PyTorch autograd engine constructs a comoputational graph in the background by tracking every operation performed on tensors.
y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
# will build a comp graph internally by default if one of its terminal nodes has requires_grad=True
w1 = torch.tensor([2.2], requires_grad=True)  # requires_grad for running gradient below
b = torch.tensor([0.0], requires_grad=True)
z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)

# grad* are resulting values of loss gradients given model's params
grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print("grad for L w.r.t w1", grad_L_w1)
print("grad for L w.r.t b", grad_L_b)

# above can be automated with .backward()
loss.backward()
print("grad for L w.r.t w1 after loss.backward()", w1.grad)
print("grad for L w.r.t b after loss.backward()", b.grad)
