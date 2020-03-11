
import  torch

a = torch.tensor([1,2])
d = float(a)/0.7
print(d)
print(torch.nonzero(torch.gt(a, 0.9)))