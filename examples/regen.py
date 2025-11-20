import numpy as np
from tinygrad.tensor import Tensor
from tinygrad import nn
from PIL import Image
import matplotlib.pyplot as plt

img=np.array(Image.open("examples/flux1_seed0.png").convert("RGB")).astype(np.float32)/255.0;H,W,_=img.shape
coords=Tensor(np.stack(np.meshgrid(np.linspace(-1,1,W),np.linspace(-1,1,H)),axis=-1).reshape(-1,2).astype(np.float32))
target=Tensor(img.reshape(-1,3))

class M:
  def __init__(s): s.W1=Tensor.kaiming_uniform(2,2560,requires_grad=True);s.b1=Tensor.zeros(2560,requires_grad=True);s.W2=Tensor.kaiming_uniform(2560,3,requires_grad=True);s.b2=Tensor.zeros(3,requires_grad=True)
  def __call__(s,x): return ((x.dot(s.W1)+s.b1).relu().dot(s.W2)+s.b2).sigmoid()
  def parameters(s): return [s.W1,s.b1,s.W2,s.b2]

m=M();opt=nn.optim.Adam(m.parameters(),lr=1e-3)
plt.ion()
with Tensor.train():
  for i in range(3000):
    p=m(coords);loss=(p-target).square().mean();opt.zero_grad();loss.backward();opt.step()
    if i%50==0:
      print(i,loss.numpy());plt.imshow(np.clip(p.numpy().reshape(H,W,3),0,1));plt.pause(0.01)
plt.ioff();plt.show()
