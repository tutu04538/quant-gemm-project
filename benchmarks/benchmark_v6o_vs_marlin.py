"""Fixed M16/K256/N128 V6o vs local minivllm Marlin; GPU-only graph timing."""
import ctypes, json, os, pathlib, statistics, sys, time
import torch
sys.path.insert(0, os.environ.get('VLLM_REFERENCE_ROOT','/data/syh/infra/vllm-from-scratch'))
from minivllm.quant_ops.ops import prepare_full_extension, gptq_marlin_repack, MARLIN_UINT4_ID
from minivllm.layers.quantization.marlin import marlin_permute_scales, marlin_zero_points, marlin_permute_bias

M,N,K=16,128,256
seed=int(os.environ.get('BENCH_SEED','20261125'))
scale_mode=os.environ.get('BENCH_SCALES','exact')
torch.manual_seed(seed)
dev=torch.device('cuda:0')
identity=prepare_full_extension()
q=torch.randint(0,16,(K,N),device=dev,dtype=torch.int32)
z=torch.randint(0,16,(2,N),device=dev,dtype=torch.int32)
a=(torch.randint(-8,9,(M,K),device=dev,dtype=torch.int32).float()*0.125).bfloat16()
s=(torch.randint(1,9,(2,N),device=dev,dtype=torch.int32).float()*0.03125).bfloat16()
if scale_mode=='inexact':
    choices=torch.tensor([0.1,0.7,0.03125,0.25],device=dev,dtype=torch.float32).bfloat16()
    s=choices[torch.randint(0,len(choices),(2,N),device=dev)]
bias=(torch.randint(-8,9,(N,),device=dev,dtype=torch.int32).float()*0.125).bfloat16()
packed=torch.zeros((K//8,N),device=dev,dtype=torch.int32)
for i in range(8): packed |= q[i::8] << (4*i)
empty=torch.empty(0,device=dev,dtype=torch.int32)
rq=gptq_marlin_repack(packed,empty,K,N)
rz=marlin_zero_points(z,N); rs=marlin_permute_scales(s,K,N,128); rb=marlin_permute_bias(bias,N)
out=torch.empty((M,N),device=dev,dtype=torch.bfloat16)
refout=torch.empty_like(out)
diag=torch.empty((M,N),device=dev,dtype=torch.float32)
tmp=torch.empty_like(diag); locks=torch.zeros(2,device=dev,dtype=torch.int32)
workspace=torch.zeros(torch.cuda.get_device_properties(dev).multi_processor_count,device=dev,dtype=torch.int32)
lib=ctypes.CDLL('/tmp/quant_v6o_benchmark.so')
fn=lib.launch_v6o
fn.argtypes=[ctypes.c_void_p]*10; fn.restype=ctypes.c_int

def ours():
    stream=torch.cuda.current_stream().cuda_stream
    result=fn(*(x.data_ptr() for x in (a,rq,rz,rs,bias,out,diag,tmp,locks)),stream)
    if result: raise RuntimeError(f'v6o launch error {result}')

def marlin():
    return torch.ops.minivllm_quant.marlin_gemm(a,refout,rq,rb,rs,None,None,rz,empty,empty,
                                                workspace,MARLIN_UINT4_ID,M,N,K,True,False,True,False)
ours(); marlin(); torch.cuda.synchronize()
# Independent logical oracle with BF16-rounded per-weight dequantization.
w=((q.float()-z.repeat_interleave(128,dim=0).float())*s.repeat_interleave(128,dim=0).float()).bfloat16()
oracle=(a.float() @ w.float()).bfloat16().float().add(bias.float()).bfloat16()
ours_f=out.float(); marlin_f=refout.float(); oracle_f=oracle.float()
diff=(ours_f-marlin_f).abs()
metrics={'max_abs_ours_marlin':diff.max().item(),'mean_abs_ours_marlin':diff.mean().item(),
         'different_bf16_elements':torch.count_nonzero(out.view(torch.int16)!=refout.view(torch.int16)).item(),
         'max_abs_ours_oracle':(ours_f-oracle_f).abs().max().item(),
         'max_abs_marlin_oracle':(marlin_f-oracle_f).abs().max().item(),
         'locks_after_check':locks.cpu().tolist()}
print('correctness',json.dumps(metrics),flush=True)
if not torch.allclose(ours_f,marlin_f,atol=0.125,rtol=0.02) or not torch.allclose(ours_f,oracle_f,atol=0.125,rtol=0.02):
    raise RuntimeError('correctness mismatch; timings intentionally not run')
# Repeated operators in a graph reduce host launch gaps. Both use fixed preallocated outputs.
for _ in range(50): ours(); marlin()
torch.cuda.synchronize()
graphs={}
for name,func in [('v6o',ours),('marlin',marlin)]:
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(128): func()
    graphs[name]=graph
for _ in range(5): graphs['v6o'].replay(); graphs['marlin'].replay()
torch.cuda.synchronize()
measure={'v6o':[],'marlin':[]}
for batch in range(21):
    order=['v6o','marlin'] if batch%2==0 else ['marlin','v6o']
    for name in order:
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record(); graphs[name].replay(); end.record(); end.synchronize()
        measure[name].append(start.elapsed_time(end)*1000/128)
summary={name:{'median_us':statistics.median(v),'min_us':min(v),'max_us':max(v),'samples_us':v} for name,v in measure.items()}
summary['ratio_v6o_over_marlin']=summary['v6o']['median_us']/summary['marlin']['median_us']
result={'shape':[M,N,K],'seed':seed,'scale_mode':scale_mode,'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__,
        'reference_build_identity':identity['identity_sha256'],'reference_library':identity['library'],
        'warmup':50,'graph_nodes':128,'graph_warmup':5,'alternating_batches':21,'correctness':metrics,'timing':summary,
        'timing_scope':'graph CUDA events: per call kernel and internal GPU operations, excluding repack and allocation outside graph; V6o includes diagnostic write and locking; Marlin reference BF16 bias is permuted beforehand'}
path=pathlib.Path(os.environ.get('BENCH_OUTPUT','learning_notes/v6o_results/benchmark.json'));path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
print('result',json.dumps(result,indent=2),flush=True)
