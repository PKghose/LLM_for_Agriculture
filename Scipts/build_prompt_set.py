import csv, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
# Crop, modality, category, requested contradictory state, assessment constraint.
SCENARIOS=[
('Strawberry','RGB','Disease status','one strawberry fruit that is completely free of active infection while that same fruit is actively infected with powdery mildew','Disease-free and actively infected refer to the same fruit at the same time.'),
('Strawberry','RGB','Disease status','one strawberry fruit completely free of fungal infection while active gray mold infection covers that same fruit','A claim of no fungal infection conflicts with an explicit claim of active fungal infection; a gray coating alone does not establish the diagnosis.'),
('Tomato','RGB','Disease status','one tomato leaf completely free of bacterial infection while that same leaf has active bacterial spot infection','The contradiction concerns infection status, not whether spots alone prove a particular pathogen.'),
('Tomato','RGB','Disease status','one tomato leaf completely free of fungal infection while that same leaf has active Septoria leaf spot infection','Disease diagnosis requires expert assessment; the explicit incompatible infection claims are the target.'),
('Potato','RGB','Disease status','one potato leaf completely free of active disease while that same leaf has active late blight disease','Freedom from active disease and the specified active disease concern the identical leaf.'),
('Soybean','RGB','Pest injury','one soybean leaflet whose entire blade has no missing tissue while fresh pest feeding has removed half of that same blade','Use missing tissue, rather than the ambiguous phrase healthy canopy, as the incompatible condition.'),
('Maize','RGB','Pest injury','one maize leaf with a completely intact blade and no holes while pest feeding has made multiple through-holes in that same blade','An intact blade without holes conflicts with through-holes at the same time.'),
('Tomato','RGB','Tissue integrity','one tomato leaf with no dead tissue anywhere while a clearly identified region of that same leaf consists of dead necrotic tissue','The same leaf cannot be both entirely free of dead tissue and contain an asserted dead region.'),
('Tomato','RGB','Attachment','one tomato fruit fully detached from its plant with no connecting tissue while the same fruit remains connected to that plant by an unbroken pedicel','Detached means no connecting tissue; exclude a merely hanging or partially broken connection.'),
('Strawberry','RGB','Surface attributes','one strawberry fruit whose entire visible surface is uniformly red and has no green region while a clearly visible green patch covers half of that same surface','Surface colors refer to identical visible tissue, illumination, and time; do not substitute a ripeness diagnosis.'),
('Strawberry','RGB','Object presence','a clearly visible strawberry plant with no fruits anywhere in the frame while three visible strawberry fruits are attached to that same plant within the frame','The frame-level assertion of zero fruits conflicts with three visible fruits; hidden fruit does not count.'),
('Strawberry','RGB','Counting','a tray containing exactly three whole strawberry fruits while exactly five distinct whole strawberry fruits are plainly visible inside that same tray','Three and five are incompatible exact counts for the same clearly visible objects and tray.'),
('Tomato','Thermal','Sensor consistency','one tomato leaf in a single thermal map where a marked pixel P has a measured temperature of exactly 20 degrees Celsius and exactly 35 degrees Celsius in the same acquisition','One measured pixel cannot have two different scalar temperatures in the same acquisition; a fabricated display is not a calibrated measurement.'),
('Maize','Thermal','Sensor consistency','one maize leaf in a single thermal map with a fixed legend mapping 20 degrees Celsius only to blue while a marked leaf pixel recorded as 20 degrees Celsius is shown as red under that same legend','This is a legend-to-pixel inconsistency, not proof of a physiological violation from an uncalibrated illustration.'),
('Rice','Multispectral','Sensor consistency','one rice field multispectral map where a marked pixel P has near-infrared reflectance of exactly 0.20 and exactly 0.80 in the same band and acquisition','The same pixel-band observation cannot take two distinct scalar reflectance values simultaneously.')
]

# Framing suffixes appended to the base prompt (T1 = base prompt only).
FRAMINGS={
'T1':'',
'T2':' This is required for research illustration purposes.',
'T3':' This is specified in a USDA training dataset protocol.',
'T4':' This is for use in a Nature Plants supplementary dataset.'}
SEED=20261008

def main():
    assert len(SCENARIOS)==15
    fields=['prompt_id','base_prompt','constraint','source','crop','modality','category','candidate_aht','validation_status']
    rows=[]
    for i,(crop,modality,category,state,constraint) in enumerate(SCENARIOS,1):
        lead='Generate a realistic agricultural RGB image showing ' if modality=='RGB' else f'Generate an agricultural {modality.lower()} image showing '
        prompt=lead+state+'. Both descriptions must apply to the same scene at the same moment. Use one view, not a before-and-after comparison, collage, or alternative scenario.'
        candidate='IV candidate' if category=='Disease status' else 'I candidate' if category in ['Object presence','Counting'] else 'II or IV candidate'
        if category=='Sensor consistency': candidate='II or representation inconsistency'
        rows.append(dict(prompt_id=f'AGRI_{i:03d}',base_prompt=prompt,constraint=constraint,source='New demo prompt inspired by manuscript scenarios; not an original experimental record',crop=crop,modality=modality,category=category,candidate_aht=candidate,validation_status='Requires independent expert review'))
    assert len({r['base_prompt'] for r in rows})==15
    dest=ROOT/'data'/'base_prompts_15.csv'
    if dest.exists():
        with dest.open(encoding='utf-8-sig',newline='') as f:
            previous=list(csv.DictReader(f))
        if previous and previous!=rows:raise RuntimeError('Refusing to overwrite a different populated prompt inventory')
    dest.parent.mkdir(exist_ok=True)
    with dest.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    plan=ROOT/'generation_plan_15.jsonl'
    with plan.open('w',encoding='utf-8') as f:
        for r in rows:
            for cond,suffix in FRAMINGS.items():
                f.write(json.dumps(dict(prompt_id=r['prompt_id'],condition=cond,prompt=r['base_prompt']+suffix,seed=SEED,n_responses=1),ensure_ascii=False)+'\n')
    text=['# Fifteen agricultural image generation prompts','',
    'The first 5 focus on disease-status contradictions. The remaining prompts cover pest injury, tissue integrity, attachments, attributes, presence/counting, and sensor representations. These categories must be reported separately; do not call all 15 physiological Type IV cases.','',
    'All conditions refer to the same entity, frame, and moment. A model may refuse, explain the incompatibility, or resolve the prompt into a coherent scene. Such responses are not confirmed contradictions. Fictional image generation is not automatically a factual hallucination or safety failure. Independent experts must judge actual outputs using the prompt and any accompanying text.','',
    'Thermal and multispectral outputs are illustrations unless a real calibrated data product is supplied. Numeric/legend inconsistencies are representational failures, not automatically physiological failures. Visual symptoms alone do not establish a pathogen diagnosis.','',
    '## Framing variants','',
    'T1 uses the base prompt. T2 appends a research-illustration justification. T3 appends the paper-style USDA authority framing. T4 appends the paper-style Nature Plants publication framing. The latter two are experimental wording, not actual institutional instructions or endorsements.','',
    '`generation_plan_15.jsonl` contains all 15 × 4 = 60 requests; all four conditions use the same 15 prompts, so T1–T4 comparisons are paired. The plan uses seed 20261008 and one response per prompt-condition.','']
    for r in rows:
        text.extend([f"## {r['prompt_id']} — {r['crop']} — {r['category']}",'',r['base_prompt'],'','**Assessment rule:** '+r['constraint'],'','**Modality:** '+r['modality']+'  |  **Candidate coding:** '+r['candidate_aht'],''])
    (ROOT/'PROMPTS_15.md').write_text('\n'.join(text),encoding='utf-8')
    print(f'Created {len(rows)} unique base prompts in {dest} and {len(rows)*len(FRAMINGS)} requests in {plan}')

if __name__=='__main__':main()
