import torch
import matplotlib.pyplot as plt

import math



a = torch.linspace(0,math.pi*2,10,requires_grad=True)

b = torch.sin(a) ### modyfikuje każdą wartośc z tensra a


b.backward() ### wywołuje liczenie do tyłu. Funkcja liczy każdy gradient aż to wartości argumentu na samym starcie.



plt.plot(a.detach(),a.grad.detach())
plt.show()