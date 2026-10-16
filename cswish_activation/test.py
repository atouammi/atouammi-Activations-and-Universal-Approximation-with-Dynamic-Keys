from cswish import CSwish

import torch
act = CSwish(k=3, beta=1.0, seed=42)   # clé fixe dérivée d'une graine


x = torch.randn(8, 16)
y = act(x)

print("Input:", x)
print("Output:", y)

#plot the input and output
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Input")
plt.hist(x.flatten().numpy(), bins=30, color='blue', alpha=0.5)
plt.subplot(1, 2, 2)
plt.title("Output")
plt.hist(y.flatten().detach().numpy(), bins=30, color='orange', alpha=0.5)
plt.show()


#compare with ReLU
relu = torch.nn.ReLU()
y_relu = relu(x)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("CSwish Output")
plt.hist(y.flatten().detach().numpy(), bins=30, color='orange', alpha=0.5)
plt.subplot(1, 2, 2)
plt.title("ReLU Output")
plt.hist(y_relu.flatten().detach().numpy(), bins=30, color='green', alpha=0.5)
plt.show()

