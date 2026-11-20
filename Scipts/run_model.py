"""Run a user-selected adapter. No credentials, model IDs, or outputs are invented."""
import argparse, importlib, json, time, hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--adapter',required=True);p.add_argument('--model',required=True);p.add_argument('--revision');p.add_argument('--out',required=True);a=p.parse_args()
    module,name=a.adapter.split(':',1);predict=getattr(importlib.import_module(module),name)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    plan=Path(a.plan);digest=hashlib.sha256(plan.read_bytes()).hexdigest()
    with out.open('x',encoding='utf-8') as f:
        for index,line in enumerate(plan.read_text(encoding='utf-8').splitlines()):
            request=json.loads(line);start=time.time()
            # Ground truth stays in the log, never in the adapter's target input.
            model_request={k:v for k,v in request.items() if k not in ['label','truth','constraint','source']}
            try:
                for img in ([request] if 'path' in request else [])+request.get('exemplars',[]):
                    if hashlib.sha256(Path(img['path']).read_bytes()).hexdigest()!=img['sha256']:raise ValueError('Image changed since plan creation')
                response=predict(model_request,model=a.model,revision=a.revision)
                if not isinstance(response,dict) or 'output' not in response:raise ValueError('Adapter must return dict containing output')
                status='ok'
            except Exception as exc:response={'error':type(exc).__name__+': '+str(exc)};status='error'
            record={'index':index,'plan_sha256':digest,'request':request,'requested_model':a.model,'requested_revision':a.revision,'adapter':a.adapter,'started_unix':start,'elapsed_seconds':time.time()-start,'status':status,'response':response}
            f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
    print(f'Raw outputs saved to {out}; expert evaluation is still required.')
if __name__=='__main__':main()
