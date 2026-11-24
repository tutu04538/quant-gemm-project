import os,sys,torch
sys.path.insert(0, os.environ.get('VLLM_REFERENCE_ROOT', '/data/syh/infra/vllm-from-scratch'))
from minivllm.layers.quantization.marlin import marlin_permute_scales,marlin_zero_points
K,N=256,128
s=torch.empty((K//128,N),dtype=torch.bfloat16)
z=torch.empty((K//128,N),dtype=torch.int32)
for g in range(K//128):
 for n in range(N):
  s[g,n]=1+(n%64)/128+g/2
  z[g,n]=(n*7+n//8+g*3)%16
rs=marlin_permute_scales(s,K,N,128);rz=marlin_zero_points(z,N)
se=torch.empty_like(s);ze=torch.zeros((K//128,N//8),dtype=torch.int64)
for g in range(K//128):
 for n in range(N):
  chunk,a,b=n//64,n%8,(n%64)//8
  se[g,chunk*64+a*8+b]=s[g,n]
  nib=b//2 if b%2==0 else 4+b//2
  ze[g,chunk*8+a] |= int(z[g,n])<<(4*nib)
print('scale bit mismatches',torch.count_nonzero(rs.view(torch.int16)!=se.view(torch.int16)).item(), 'zero word mismatches',torch.count_nonzero(rz.to(torch.int64).bitwise_and(0xffffffff)!=ze).item())
assert torch.equal(rs.view(torch.int16),se.view(torch.int16)) and torch.equal(rz.to(torch.int64).bitwise_and(0xffffffff),ze)
