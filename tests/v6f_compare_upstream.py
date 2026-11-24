import os, sys, torch, numpy as np
sys.path.insert(0, os.environ.get('VLLM_REFERENCE_ROOT', '/data/syh/infra/vllm-from-scratch'))
from minivllm.quant_ops.ops import gptq_marlin_repack, prepare_full_extension
K,N=256,128
k=torch.arange(K, dtype=torch.int32).view(K,1)
n=torch.arange(N, dtype=torch.int32).view(1,N)
q=(k+3*n+5*(k//128))%16
p=torch.zeros((K//8,N),dtype=torch.int32)
for j in range(8): p |= q[j::8] << (4*j)
prepare_full_extension()
x=gptq_marlin_repack(p.cuda(),torch.empty(0,dtype=torch.int32,device='cuda'),K,N)
torch.cuda.synchronize()
got=np.fromfile(sys.argv[1] if len(sys.argv)>1 else '/tmp/quant_v6f_words.bin',dtype=np.uint32)
ref=x.cpu().numpy().view(np.uint32).reshape(-1)
print('words',ref.size,'mismatches',np.count_nonzero(got!=ref),'reference_shape',tuple(x.shape))
if not np.array_equal(got,ref): sys.exit(1)
