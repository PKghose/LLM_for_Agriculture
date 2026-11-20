import argparse, csv, hashlib, itertools, json, math, platform
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from statistics import NormalDist

CLASSES=['Healthy','Bacterial Spot','Septoria Leaf Spot']
CRITERIA=['botanical','symptom','modality','adherence','consistency']
WEIGHTS=np.array([.2,.25,.2,.15,.2])
SCHEMAS={
 'images':['image_id','path','task','label','split','cluster_id'],
 'exemplars':['exemplar_set','image_id','path','task','label'],
 'base_prompts':['prompt_id','base_prompt','constraint','source'],
 'classification':['image_id','model','setting','exemplar_set','truth','prediction'],
 'counting':['image_id','cluster_id','model','setting','truth_count','pred_count'],
 'generation':['prompt_id','model','condition','repeat_id','status','contradiction'],
 'bfs':['prompt_id','image_id','model','condition','rater']+CRITERIA,
 'aht':['claim_id','image_id','model','task','setting','rater','label']}

def require(condition,message):
    if not condition: raise ValueError(message)

def rows(path, schema=None):
    with open(path,encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if schema: require(set(SCHEMAS[schema])<=set(reader.fieldnames or []),f'Missing {schema} columns')
        result=list(reader)
    require(result,f'No observations in {path}')
    return result

def unique(records,keys):
    seen=set()
    for r in records:
        k=tuple(r[x] for x in keys)
        require(k not in seen,f'Duplicate key {k}; repeated observations need a clustered design')
        seen.add(k)

def groups(records,keys):
    g=defaultdict(list)
    for r in records: g[tuple(r[k] for k in keys)].append(r)
    return g

def numeric(s):
    x=float(s); require(math.isfinite(x),'Nonfinite value'); return x

def integer(s):
    x=numeric(s); require(x>=0 and x.is_integer(),'Expected nonnegative integer'); return int(x)

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)): return [clean(v) for v in x]
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x): return None
    return x

def mcnemar(a,b):
    a=np.asarray(a,dtype=bool); b=np.asarray(b,dtype=bool)
    require(a.shape==b.shape and len(a)>0,'Paired outcomes required')
    n01=int(np.sum(~a & b)); n10=int(np.sum(a & ~b)); n=n01+n10
    return {'n_pairs':len(a),'correct_only_first':n10,'correct_only_second':n01,
            'chi2_uncorrected':(n10-n01)**2/n if n else 0.,
            'p_exact':min(1.,2*sum(math.exp(math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-n*math.log(2)) for i in range(min(n01,n10)+1))) if n else 1.}

def adjust(items):
    n=len(items)
    for x in items: x.update(p_bonferroni=min(1.,n*x['p_exact']),family_size=n,raw_alpha=.05/n)

def ci(values):
    a=np.asarray(values); a=a[np.isfinite(a)]
    return np.quantile(a,[.025,.975]).tolist() if len(a) else [None,None]

def paired_mean(a,b,rng,B):
    d=np.asarray(b)-np.asarray(a); require(len(d)>0,'No paired data')
    samples=[float(np.mean(rng.choice(d,size=len(d),replace=True))) for _ in range(B)]
    return {'second_minus_first':float(np.mean(d)),'ci95':ci(samples),'n_pairs':len(d)}

def wilson(k,n):
    require(n>0 and 0<=k<=n,'Invalid binomial counts')
    z=NormalDist().inv_cdf(.975); p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]

def classification(rs,rng,B):
    unique(rs,['model','exemplar_set','image_id','setting']); out=[]
    for (model,ex), rr in groups(rs,['model','exemplar_set']).items():
        by={setting:{r['image_id']:r for r in rr if r['setting']==setting} for setting in ['zero','one']}
        require(set(r['setting'] for r in rr)=={'zero','one'},'Use settings zero and one')
        require(by['zero'].keys()==by['one'].keys(),'Unmatched classification images')
        ids=sorted(by['zero']); require(ids,'No pairs')
        truth=[by['zero'][i]['truth'] for i in ids]
        require(all(by['one'][i]['truth']==t for i,t in zip(ids,truth)),'Truth mismatch')
        require(set(truth)<=set(CLASSES),'Unexpected tomato ground-truth label')
        preds={k:[by[k][i]['prediction'] for i in ids] for k in by}
        require(all(set(p)<=set(CLASSES)|{'abstain'} for p in preds.values()),'Invalid prediction; use abstain explicitly')
        correct={k:np.array(p)==truth for k,p in preds.items()}
        item={'model':model,'exemplar_set':ex,**mcnemar(correct['zero'],correct['one'])}
        for k,p in preds.items():
            f=[]
            for cls in CLASSES:
                tp=sum(t==cls and q==cls for t,q in zip(truth,p)); fp=sum(t!=cls and q==cls for t,q in zip(truth,p)); fn=sum(t==cls and q!=cls for t,q in zip(truth,p))
                f.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
            item[k]={'correct':int(correct[k].sum()),'n':len(ids),'accuracy':float(correct[k].mean()),'macro_f1':float(np.mean(f))}
        item['accuracy_difference']=paired_mean(correct['zero'].astype(float),correct['one'].astype(float),rng,B)
        out.append(item)
    adjust(out); return out

def average_ranks(values):
    values=np.asarray(values);order=np.argsort(values,kind='stable');ranks=np.empty(len(values),float);i=0
    while i<len(values):
        j=i+1
        while j<len(values) and values[order[j]]==values[order[i]]:j+=1
        ranks[order[i:j]]=(i+j+1)/2;i=j
    return ranks

def count_metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); err=p-y; sst=float(np.sum((y-y.mean())**2))
    rho=float(np.corrcoef(average_ranks(y),average_ranks(p))[0,1]) if len(set(y))>1 and len(set(p))>1 else float('nan')
    return np.array([np.mean(abs(err)),np.sqrt(np.mean(err**2)),1-np.sum(err**2)/sst if sst else np.nan,rho,np.mean(abs(err)<=1)])

def counting(rs,rng,B):
    unique(rs,['model','image_id','setting']); out=[]; names=['mae','rmse','r2','spearman','pa_within_one']
    for (model,),rr in groups(rs,['model']).items():
        require(set(r['setting'] for r in rr)=={'zero','one'},'Use settings zero and one')
        by={k:{r['image_id']:r for r in rr if r['setting']==k} for k in ['zero','one']}
        require(by['zero'].keys()==by['one'].keys(),'Unmatched counting images'); ids=sorted(by['zero'])
        y=np.array([integer(by['zero'][i]['truth_count']) for i in ids]); pred={k:np.array([integer(by[k][i]['pred_count']) for i in ids]) for k in by}
        require(all(integer(by['one'][i]['truth_count'])==v for i,v in zip(ids,y)),'Truth mismatch')
        clusters=[by['zero'][i]['cluster_id'] or i for i in ids]
        require(all((by['one'][i]['cluster_id'] or i)==c for i,c in zip(ids,clusters)),'Cluster mismatch')
        cs=sorted(set(clusters)); require(len(cs)>=2,'Need at least two independent clusters')
        index={c:np.array([j for j,v in enumerate(clusters) if v==c]) for c in cs}
        boot={k:[] for k in by}
        for _ in range(B):
            ix=np.concatenate([index[c] for c in rng.choice(cs,size=len(cs),replace=True)])
            for k in by: boot[k].append(count_metrics(y[ix],pred[k][ix]))
        item={'model':model,'n_images':len(ids),'n_clusters':len(cs),'resampling_unit':'cluster_id or image_id'}
        for k in by:
            point=count_metrics(y,pred[k]); boot[k]=np.asarray(boot[k]); item[k]={name:{'estimate':point[j],'ci95':ci(boot[k][:,j]),'valid_bootstrap_replicates':int(np.isfinite(boot[k][:,j]).sum())} for j,name in enumerate(names)}
        delta=count_metrics(y,pred['one'])-count_metrics(y,pred['zero'])
        item['one_minus_zero']={name:{'estimate':delta[j],'ci95':ci((boot['one']-boot['zero'])[:,j])} for j,name in enumerate(names)}
        out.append(item)
    return out

def cochran_q(x):
    x=np.asarray(x,float); k=x.shape[1]; col=x.sum(axis=0); row=x.sum(axis=1); total=x.sum(); den=k*total-np.sum(row**2)
    return float((k-1)*(k*np.sum(col**2)-total**2)/den) if den else 0.

def generation(rs,rng,B):
    unique(rs,['model','prompt_id','condition'])
    require(all(r['condition'] in ['T1','T2','T3','T4'] for r in rs),'Invalid framing label')
    valid=[]; rates=[]
    for r in rs:
        require(r['status'] in ['generated','refused','corrected','error'],'Invalid response status')
        require(r['repeat_id']=='1','This analysis supports one response per prompt cell only')
        if r['status']=='error': require(r['contradiction']=='','Technical error cannot carry a scored outcome'); continue
        require(r['contradiction'] in ['0','1'],'Binary adjudicated contradiction required')
        require(r['status']=='generated' or r['contradiction']=='0','Refusal/correction cannot be contradictory under this schema')
        valid.append(r)
    for key,rr in groups(rs,['model','condition']).items():
        vr=[r for r in rr if r['status']!='error']; n=len(vr); k=sum(int(r['contradiction']) for r in vr)
        rates.append({'model':key[0],'condition':key[1],'contradictions':k,'n_valid':n,'n_requests':len(rr),'status_counts':dict(Counter(r['status'] for r in rr)),'rate':k/n if n else None,'wilson_ci95':wilson(k,n) if n else None})
    contrasts=[]; omnibus=[]
    for (model,),rr in groups(rs,['model']).items():
        allids=sorted(set(r['prompt_id'] for r in rr)); lookup={(r['prompt_id'],r['condition']):int(r['contradiction']) for r in valid if r['model']==model}
        ids=[i for i in allids if all((i,c) in lookup for c in ['T1','T2','T3','T4'])]
        if not ids:
            omnibus.append({'model':model,'n_complete':0,'excluded_prompt_ids':allids,'reason':'No complete valid T1-T4 blocks'}); continue
        x=np.array([[lookup[i,c] for c in ['T1','T2','T3','T4']] for i in ids]); q=cochran_q(x)
        extreme=sum(cochran_q(np.array([rng.permutation(row) for row in x]))>=q-1e-12 for _ in range(B))
        omnibus.append({'model':model,'n_complete':len(ids),'included_prompt_ids':ids,'excluded_prompt_ids':sorted(set(allids)-set(ids)),'cochran_q':q,'permutation_p':(extreme+1)/(B+1)})
        for j in range(1,4):
            item={'model':model,'first':'T1','second':f'T{j+1}',**mcnemar(x[:,0],x[:,j]),'risk_difference':paired_mean(x[:,0],x[:,j],rng,B)}; contrasts.append(item)
    adjust(contrasts)
    comparisons=[]
    for cond in ['T1','T2','T3','T4']:
        by={m:{r['prompt_id']:int(r['contradiction']) for r in valid if r['model']==m and r['condition']==cond} for m in sorted(set(r['model'] for r in rs))}
        for a,b in itertools.combinations(by,2):
            ids=sorted(by[a].keys() & by[b].keys())
            if not ids: continue
            av=[by[a][i] for i in ids]; bv=[by[b][i] for i in ids]
            comparisons.append({'condition':cond,'first_model':a,'second_model':b,'matched_prompt_ids':ids,**mcnemar(av,bv),'risk_difference':paired_mean(av,bv,rng,B)})
    adjust(comparisons)
    return {'rates':rates,'omnibus':omnibus,'framing_comparisons':contrasts,'model_comparisons':comparisons,'note':'McNemar fields correct_only_first/second mean contradiction only in first/second for this binary endpoint.'}

def icc_absolute(x):
    x=np.asarray(x,float); n,k=x.shape
    if n<2:return None
    grand=x.mean(); msr=k*np.sum((x.mean(axis=1)-grand)**2)/(n-1); msc=n*np.sum((x.mean(axis=0)-grand)**2)/(k-1)
    mse=np.sum((x-x.mean(axis=1)[:,None]-x.mean(axis=0)[None,:]+grand)**2)/((n-1)*(k-1))
    den=msr+(k-1)*mse+k*(msc-mse)/n
    return float((msr-mse)/den) if den else None

def bfs(rs,rng,B):
    unique(rs,['image_id','rater']); raters=sorted(set(r['rater'] for r in rs)); require(len(raters)==2,'Two independent pre-adjudication raters required')
    images=[]; matrix=[]
    for (image,),rr in groups(rs,['image_id']).items():
        require(set(r['rater'] for r in rr)==set(raters),'Incomplete paired BFS ratings')
        require(len(set((r['model'],r['condition'],r['prompt_id']) for r in rr))==1,'Inconsistent image metadata')
        vals=np.array([[numeric(next(r for r in rr if r['rater']==who)[c]) for c in CRITERIA] for who in raters])
        require(np.all((vals>=1)&(vals<=5)&(vals==np.floor(vals))),'Criteria require integer scores 1-5')
        images.append({**{k:rr[0][k] for k in ['image_id','model','condition','prompt_id']},'criteria':vals.mean(axis=0)})
        matrix.append(vals@WEIGHTS)
    unique(images,['model','condition','prompt_id']); mat=np.array(matrix)
    weights={'original':WEIGHTS,'equal':np.ones(5)/5}
    for j,c in enumerate(CRITERIA):
        for delta in [-.05,.05]:
            w=WEIGHTS.copy();w[j]+=delta;w/=w.sum();weights[f'{c}_{delta:+.2f}']=w
    reports={}
    for name,w in weights.items():
        by=groups(images,['model','condition']); means={' / '.join(k):float(np.mean([r['criteria']@w for r in rr])) for k,rr in by.items()}; comparisons=[]
        for condition in sorted(set(r['condition'] for r in images)):
            models=sorted(set(r['model'] for r in images if r['condition']==condition))
            for a,b in itertools.combinations(models,2):
                aa={r['prompt_id']:r['criteria']@w for r in images if r['model']==a and r['condition']==condition};bb={r['prompt_id']:r['criteria']@w for r in images if r['model']==b and r['condition']==condition}; ids=sorted(aa.keys() & bb.keys())
                if ids:comparisons.append({'condition':condition,'first_model':a,'second_model':b,'matched_prompt_ids':ids,**paired_mean([aa[i] for i in ids],[bb[i] for i in ids],rng,B)})
        ranks={c:sorted([m for m in set(r['model'] for r in images) if f'{m} / {c}' in means],key=lambda m:means[f'{m} / {c}'],reverse=True) for c in set(r['condition'] for r in images)}
        reports[name]={'weights':w,'means':means,'ranking_descending':ranks,'paired_model_differences':comparisons}
    pearson=float(np.corrcoef(mat[:,0],mat[:,1])[0,1]) if len(mat)>1 and all(np.std(mat[:,j])>0 for j in [0,1]) else None
    return {'n_images':len(images),'raters':raters,'pearson_association':pearson,'icc_A1_point_only':icc_absolute(mat),'weight_sensitivity':reports,'population':'Generated images with complete independent ratings; no refusal imputation.'}

def aht(rs,rng,B):
    unique(rs,['claim_id','rater']); labels=['I','II','III','IV','ordinary_error','indeterminate']; raters=sorted(set(r['rater'] for r in rs));require(len(raters)==2,'Two independent AHT coders required')
    matrix=np.zeros((6,6),int)
    for (claim,),rr in groups(rs,['claim_id']).items():
        require(len(rr)==2 and set(r['rater'] for r in rr)==set(raters),'Incomplete AHT coding')
        require(len(set(tuple(r[k] for k in ['image_id','model','task','setting']) for r in rr))==1,'Claim metadata mismatch')
        codes=[next(r['label'] for r in rr if r['rater']==who) for who in raters];require(all(c in labels for c in codes),'Invalid AHT label')
        matrix[labels.index(codes[0]),labels.index(codes[1])]+=1
    n=matrix.sum(); po=np.trace(matrix)/n; pe=np.dot(matrix.sum(0),matrix.sum(1))/n**2
    counts=[{'model':k[0],'task':k[1],'setting':k[2],'rater':k[3],'n_claims':len(rr),'n_images':len(set(r['image_id'] for r in rr)),'counts':dict(Counter(r['label'] for r in rr))} for k,rr in groups(rs,['model','task','setting','rater']).items()]
    return {'n_claims':int(n),'raters':raters,'labels':labels,'confusion_matrix':matrix,'raw_agreement':po,'cohen_kappa':(po-pe)/(1-pe) if pe<1 else None,'per_rater_distributions':counts}

def file_hash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def make_prompts(args):
    rs=rows(args.base,'base_prompts');unique(rs,['prompt_id']);require(0<args.subset_size<=len(rs),'Subset exceeds available vetted prompts')
    require(all(r['base_prompt'].strip() and r['constraint'].strip() for r in rs),'Prompt text and expert constraint required')
    rs=sorted(rs,key=lambda r:r['prompt_id']); rng=np.random.default_rng(args.seed); subset=set(rng.choice([r['prompt_id'] for r in rs],size=args.subset_size,replace=False)); template=json.loads(Path(__file__).with_name('prompt_templates.json').read_text())
    plan=[]
    for r in rs:
        for cond in (['T1','T2','T3','T4'] if r['prompt_id'] in subset else ['T1']):
            plan.append({**r,'condition':cond,'repeat_id':1,'framing_subset':r['prompt_id'] in subset,'prompt':r['base_prompt']+template['framing_suffixes'][cond],'seed':args.seed,'provenance':'new_reconstructed_plan'})
    return plan

def make_plan(args):
    images=rows(args.images,'images'); ex=rows(args.exemplars,'exemplars');unique(images,['image_id']);unique(ex,['exemplar_set','image_id'])
    for rr,base in [(images,Path(args.images).resolve().parent),(ex,Path(args.exemplars).resolve().parent)]:
        for r in rr:
            path=(base/r['path']).resolve();require(path.is_file(),f'Missing image {path}');r['path']=str(path);r['sha256']=file_hash(path)
            require(r['task'] in ['tomato','strawberry'],'Unknown task')
    require(all(r['split']=='test' for r in images),'Evaluation manifest must contain test split only')
    require(not(set(r['image_id'] for r in images)&set(r['image_id'] for r in ex)),'Exemplar/test ID overlap')
    require(not(set(r['sha256'] for r in images)&set(r['sha256'] for r in ex)),'Exemplar/test file overlap')
    templates=json.loads(Path(__file__).with_name('prompt_templates.json').read_text()); out=[]
    for task in sorted(set(r['task'] for r in images)):
        sets=groups([r for r in ex if r['task']==task],['exemplar_set']);require(sets,f'No exemplars for {task}')
        for (exid,),examples in sets.items():
            require(len(set(r['sha256'] for r in examples))==len(examples),'Duplicate exemplar file')
            if task=='tomato':require(Counter(r['label'] for r in examples)==Counter(CLASSES),'Tomato needs one exemplar per class')
            else: require(len(examples)==1,'Strawberry convention: one demonstration per set')
            for r in images:
                if r['task']!=task:continue
                for setting in ['zero','one']:
                    out.append({**r,'setting':setting,'exemplar_set':exid,'prompt':templates[task],'exemplars':examples if setting=='one' else [],'provenance':'new_reconstructed_plan'})
    return out

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('init');q.add_argument('--out',required=True)
    q=sub.add_parser('prompts');q.add_argument('--base',required=True);q.add_argument('--out',required=True);q.add_argument('--subset-size',type=int,default=15);q.add_argument('--seed',type=int,default=20261008)
    q=sub.add_parser('plan');q.add_argument('--images',required=True);q.add_argument('--exemplars',required=True);q.add_argument('--out',required=True)
    for name in ['classification','counting','generation','bfs','aht']:
        q=sub.add_parser(name);q.add_argument('--input',required=True);q.add_argument('--out',required=True);q.add_argument('--seed',type=int,default=20261008);q.add_argument('--resamples',type=int,default=2000)
    a=p.parse_args();dest=Path(a.out)
    try:
        if a.command=='init':
            dest.mkdir(parents=True,exist_ok=True)
            for name,fields in SCHEMAS.items():
                path=dest/(name+'.csv')
                if not path.exists():
                    with path.open('w',encoding='utf-8',newline='') as f:csv.writer(f).writerow(fields)
            print('Created empty input templates; no study observations.');return
        dest.parent.mkdir(parents=True,exist_ok=True)
        require(not dest.exists(),'Output already exists; choose a new output to preserve provenance')
        if a.command in ['prompts','plan']:
            result=make_prompts(a) if a.command=='prompts' else make_plan(a)
            dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result),encoding='utf-8')
        else:
            require(a.resamples>=100,'Use at least 100 resamples')
            result=globals()[a.command](rows(a.input,a.command),np.random.default_rng(a.seed),a.resamples)
            payload={'implementation':'new_from_manuscript_not_original_experiment','input_sha256':file_hash(a.input),'settings':vars(a),'python':platform.python_version(),'numpy':np.__version__,'results':result}
            dest.write_text(json.dumps(clean(payload),indent=2,allow_nan=False),encoding='utf-8')
        print(f'Wrote {dest}')
    except (ValueError,OSError) as e:p.exit(2,f'Input/analysis error: {e}\n')

if __name__=='__main__':main()
