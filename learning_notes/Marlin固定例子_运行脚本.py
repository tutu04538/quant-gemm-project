import sys, importlib.util, json, torch
from pathlib import Path
root=Path('/data/syh/infra/vllm-from-scratch')
sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location('step122_ref', root/'tests/step122/test_step122_cuda_integration.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
layer, weight, bias=mod.make_layer(n=128,k=256,dtype=torch.bfloat16)
gen=torch.Generator().manual_seed(456)
x=(torch.randn((16,256),generator=gen)*.3).to(torch.bfloat16).cuda()
with torch.inference_mode():
    result,_=layer(x)
    torch.cuda.synchronize()
    expected=x.cpu().float() @ weight + bias.float()
    torch.testing.assert_close(result.cpu().float(),expected,rtol=2e-2,atol=5e-2)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        result,_=layer(x)
        torch.cuda.synchronize()
trace='/tmp/quant_marlin_trace.json'
prof.export_chrome_trace(trace)
data=json.load(open(trace))
for event in data['traceEvents']:
    if 'Marlin<' in event.get('name',''):
        print('KERNEL',json.dumps({'name':event.get('name'),'args':event.get('args')},ensure_ascii=False))
from minivllm.quant_ops.build import build_identity
identity,digest=build_identity('full')
print('BUILD_ID',digest)
print('DEVICE',torch.cuda.get_device_properties(0))
print('CHECK', 'passed rtol=0.02 atol=0.05', 'max_abs_error', (result.cpu().float()-expected).abs().max().item())
print('TRACE',trace)
