"""V6o diagnostic-store ablation; compare both variants and Marlin in one run."""
import ctypes, hashlib, json, os, pathlib, statistics, sys
import torch
sys.path.insert(0, os.environ.get('VLLM_REFERENCE_ROOT','/data/syh/infra/vllm-from-scratch'))
from minivllm.quant_ops.ops import prepare_full_extension, gptq_marlin_repack, MARLIN_UINT4_ID
from minivllm.layers.quantization.marlin import marlin_permute_scales, marlin_zero_points, marlin_permute_bias

M,N,K=16,128,256
seed=int(os.environ.get('BENCH_SEED','20261126'))
torch.manual_seed(seed); dev=torch.device('cuda:0'); identity=prepare_full_extension()
q=torch.randint(0,16,(K,N),device=dev,dtype=torch.int32)
z=torch.randint(0,16,(2,N),device=dev,dtype=torch.int32)
a=(torch.randint(-8,9,(M,K),device=dev,dtype=torch.int32).float()*0.125).bfloat16()
choices=torch.tensor([0.1,0.7,0.03125,0.25],device=dev,dtype=torch.float32).bfloat16()
s=choices[torch.randint(0,len(choices),(2,N),device=dev)]
bias=(torch.randint(-8,9,(N,),device=dev,dtype=torch.int32).float()*0.125).bfloat16()
packed=torch.zeros((K//8,N),device=dev,dtype=torch.int32)
for i in range(8): packed |= q[i::8] << (4*i)
empty=torch.empty(0,device=dev,dtype=torch.int32)
rq=gptq_marlin_repack(packed,empty,K,N); rz=marlin_zero_points(z,N); rs=marlin_permute_scales(s,K,N,128)
rb=marlin_permute_bias(bias,N)
outs={x:torch.empty((M,N),device=dev,dtype=torch.bfloat16) for x in ('diag','nodiag','marlin')}
diags={x:torch.full((M,N),float('nan'),device=dev) for x in ('diag','nodiag')}
tmps={x:torch.empty((M,N),device=dev) for x in ('diag','nodiag')}
locks={x:torch.zeros(2,device=dev,dtype=torch.int32) for x in ('diag','nodiag')}
workspace=torch.zeros(torch.cuda.get_device_properties(dev).multi_processor_count,device=dev,dtype=torch.int32)
libs={x:ctypes.CDLL('/tmp/v6o_'+x+'.so') for x in ('diag','nodiag')}
funcs={x:libs[x].launch_v6o for x in libs}
for fn in funcs.values(): fn.argtypes=[ctypes.c_void_p]*10; fn.restype=ctypes.c_int

def launch(name):
    ptrs=(a,rq,rz,rs,bias,outs[name],diags[name],tmps[name],locks[name])
    rc=funcs[name](*(x.data_ptr() for x in ptrs),torch.cuda.current_stream().cuda_stream)
    if rc: raise RuntimeError(f'{name}: CUDA launch {rc}')

def marlin():
    return torch.ops.minivllm_quant.marlin_gemm(a,outs['marlin'],rq,rb,rs,None,None,rz,empty,empty,
                                               workspace,MARLIN_UINT4_ID,M,N,K,True,False,True,False)
launch('diag'); launch('nodiag'); marlin(); torch.cuda.synchronize()
w=((q.float()-z.repeat_interleave(128,dim=0).float())*s.repeat_interleave(128,dim=0).float()).bfloat16()
oracle=(a.float() @ w.float()).bfloat16().float().add(bias.float()).bfloat16()
metrics={}
for name in ('diag','nodiag','marlin'):
    metrics[name+'_vs_oracle_max_abs']=(outs[name].float()-oracle.float()).abs().max().item()
    if not torch.allclose(outs[name].float(),oracle.float(),atol=0.125,rtol=0.02): raise RuntimeError(name+' oracle mismatch')
for name in ('nodiag','marlin'):
    metrics[name+'_vs_diag_bits']=torch.count_nonzero(outs[name].view(torch.int16)!=outs['diag'].view(torch.int16)).item()
metrics['nodiag_buffer_untouched']=torch.isnan(diags['nodiag']).all().item()
metrics['diag_buffer_finite']=torch.isfinite(diags['diag']).all().item()
metrics['locks']={name:locks[name].cpu().tolist() for name in locks}
print('correctness',json.dumps(metrics),flush=True)
if metrics['nodiag_vs_diag_bits'] or not metrics['nodiag_buffer_untouched'] or not metrics['diag_buffer_finite'] or any(v!=[0,0] for v in metrics['locks'].values()):
    raise RuntimeError('variant verification failed')
runs={'diag':lambda:launch('diag'),'nodiag':lambda:launch('nodiag'),'marlin':marlin}
for _ in range(50):
    for f in runs.values():f()
torch.cuda.synchronize()
graphs={}
for name,func in runs.items():
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(128):func()
    graphs[name]=graph
for _ in range(5):
    for graph in graphs.values():graph.replay()
torch.cuda.synchronize()
measure={name:[] for name in runs}
orders=[('diag','nodiag','marlin'),('marlin','nodiag','diag'),('nodiag','diag','marlin'),('marlin','diag','nodiag'),('diag','marlin','nodiag'),('nodiag','marlin','diag')]
for batch in range(36):
    for name in orders[batch%len(orders)]:
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();graphs[name].replay();end.record();end.synchronize()
        measure[name].append(start.elapsed_time(end)*1000/128)
summary={n:{'median_us':statistics.median(v),'min_us':min(v),'max_us':max(v),'samples_us':v} for n,v in measure.items()}
summary['nodiag_over_diag']=summary['nodiag']['median_us']/summary['diag']['median_us']
summary['nodiag_over_marlin']=summary['nodiag']['median_us']/summary['marlin']['median_us']
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
result={'shape':[M,N,K],'seed':seed,'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__,
        'source_sha256':sha('kernels/marlin_from_scratch/v6o_four_block_template.cu'),
        'libraries_sha256':{n:sha('/tmp/v6o_'+n+'.so') for n in libs},'reference_build_identity':identity['identity_sha256'],
        'warmup':50,'graph_nodes':128,'graph_warmup':5,'batches':36,'correctness':metrics,'timing':summary,
        'scope':'CUDA graph events; same input, fixed BF16 output, reference-repacked Q/Z/S; V6o variants differ only diagnostic store and have separate identical workspaces; excludes repack and host launch'}
path=pathlib.Path('learning_notes/v6o_results/diagnostic_ablation.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
print('result',json.dumps(result,indent=2),flush=True)
