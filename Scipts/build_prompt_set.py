"""Build a new 50-scenario demo inventory; never historical study evidence."""
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
('Wheat','RGB','Disease status','one wheat leaf completely free of active disease while that same leaf has active leaf rust disease','Do not infer confirmed rust from color alone; evaluate the conflicting disease assertions.'),
('Grape','RGB','Disease status','one grape leaf completely free of active fungal infection while that same leaf has active powdery mildew infection','The same leaf cannot satisfy both asserted infection states simultaneously.'),
('Cucumber','RGB','Disease status','one cucumber leaf completely free of active disease while that same leaf has active powdery mildew disease','Score the mutually exclusive disease assertions, not visual ambiguity alone.'),
('Apple','RGB','Disease status','one apple fruit completely free of active disease while that same fruit has active apple scab disease','This is an assertion conflict about one fruit; visible blemishes alone are not diagnostic proof.'),
('Rice','RGB','Disease status','one rice leaf completely free of active disease while that same leaf has active rice blast disease','The target contradiction is simultaneous absence and presence of active disease on the same leaf.'),
('Soybean','RGB','Pest injury','one soybean leaflet whose entire blade has no missing tissue while fresh pest feeding has removed half of that same blade','Use missing tissue, rather than the ambiguous phrase healthy canopy, as the incompatible condition.'),
('Maize','RGB','Pest injury','one maize leaf with a completely intact blade and no holes while pest feeding has made multiple through-holes in that same blade','An intact blade without holes conflicts with through-holes at the same time.'),
('Cabbage','RGB','Pest injury','one cabbage leaf with no tissue removed by feeding while visible caterpillars are removing tissue from that same leaf at the depicted moment','Evaluate the explicit no-feeding-removal claim against depicted ongoing tissue removal.'),
('Cotton','RGB','Pest injury','one cotton leaf with every part of its blade intact while the same blade is skeletonized with most tissue between its veins missing','Intact tissue and absent interveinal tissue concern the identical blade.'),
('Apple','RGB','Pest injury','one apple fruit with an entirely unbroken skin while an insect entry hole passes through that same skin','A through-hole and an entirely unbroken skin are mutually exclusive at the same location.'),
('Tomato','RGB','Tissue integrity','one tomato leaf with no dead tissue anywhere while a clearly identified region of that same leaf consists of dead necrotic tissue','The same leaf cannot be both entirely free of dead tissue and contain an asserted dead region.'),
('Lettuce','RGB','Tissue integrity','one lettuce leaf with a completely continuous untorn blade while a full-thickness tear crosses that same blade','A through-tear conflicts with continuity of the same tissue.'),
('Pepper','RGB','Tissue integrity','one pepper fruit with a completely intact uncracked surface while a deep open crack splits that same surface','An open crack contradicts an explicitly uncracked surface.'),
('Grape','RGB','Tissue integrity','one grape berry with an entirely sealed skin while juice escapes through an open rupture in that same skin','The prompt specifies a rupture, not harmless moisture on intact skin.'),
('Potato','RGB','Tissue integrity','one potato tuber with no cuts or exposed internal tissue while a fresh cut exposes the interior of that same tuber','The same tuber surface cannot both have no cut and an open cut exposing its interior.'),
('Tomato','RGB','Attachment','one tomato fruit fully detached from its plant with no connecting tissue while the same fruit remains connected to that plant by an unbroken pedicel','Detached means no connecting tissue; exclude a merely hanging or partially broken connection.'),
('Strawberry','RGB','Attachment','one strawberry fruit fully separated from its plant with no connecting tissue while an intact stem still joins that same fruit to the plant','Both claims must concern the same fruit and physical connection.'),
('Apple','RGB','Attachment','one apple completely separated from its branch while that same apple is still joined to that branch by its unbroken stalk','No separation across time or different apples is allowed.'),
('Wheat','RGB','Attachment','one wheat head completely severed from its stem while continuous uncut tissue still joins that same head to that stem','A completed severing is incompatible with an uncut connection.'),
('Cucumber','RGB','Attachment','one cucumber fruit completely detached from its vine while an unbroken peduncle still physically connects that same fruit to the vine','Compare the same attachment, not two adjacent fruits.'),
('Strawberry','RGB','Surface attributes','one strawberry fruit whose entire visible surface is uniformly red and has no green region while a clearly visible green patch covers half of that same surface','Surface colors refer to identical visible tissue, illumination, and time; do not substitute a ripeness diagnosis.'),
('Tomato','RGB','Surface attributes','one tomato fruit whose entire visible skin is uniformly green with no red region while half of that same visible skin is red','A multicolored fruit is plausible but contradicts the explicitly uniform-green condition.'),
('Banana','RGB','Surface attributes','one banana whose entire visible peel is uniformly yellow and has no brown marks while multiple brown marks cover that same visible peel','The contradiction is absence versus presence of marks, not the biological possibility of spotting.'),
('Grape','RGB','Surface attributes','one grape berry whose entire visible skin is uniformly purple with no green area while a green area occupies half of that same visible skin','Use one berry, one view, and fixed lighting; a bunch containing differently colored berries does not satisfy the prompt.'),
('Rice','RGB','Surface attributes','one rice leaf whose entire visible blade is green with no yellow area while a broad yellow area covers that same visible blade','A yellow-green leaf can exist; the conflict is with the explicit all-green assertion.'),
('Strawberry','RGB','Object presence','a clearly visible strawberry plant with no fruits anywhere in the frame while three visible strawberry fruits are attached to that same plant within the frame','The frame-level assertion of zero fruits conflicts with three visible fruits; hidden fruit does not count.'),
('Tomato','RGB','Object presence','a tomato plant with no leaves anywhere in the frame while five visible leaves remain attached to that same plant inside the frame','Count visible leaves within the identical image boundary.'),
('Maize','RGB','Object presence','a maize plant with no ears visible anywhere in the frame while two clearly visible ears are attached to that same plant inside the frame','The same frame cannot contain both zero and two visible ears.'),
('Apple','RGB','Object presence','an apple branch with no blossoms anywhere in the frame while several clearly visible open blossoms are attached to that branch inside the frame','The absence claim and visible blossoms use the same branch and frame.'),
('Wheat','RGB','Object presence','a tray containing no wheat kernels anywhere inside it while ten clearly visible wheat kernels lie inside that same tray','Tray contents and boundary must be identical; a reflection or printed picture is not a kernel.'),
('Strawberry','RGB','Counting','a tray containing exactly three whole strawberry fruits while exactly five distinct whole strawberry fruits are plainly visible inside that same tray','Three and five are incompatible exact counts for the same clearly visible objects and tray.'),
('Tomato','RGB','Counting','a basket containing exactly two whole tomatoes while exactly four distinct whole tomatoes are plainly visible inside that same basket','Do not treat separate views or partially hidden objects as extra observations.'),
('Apple','RGB','Counting','a branch bearing exactly one whole apple while exactly three distinct whole apples are plainly visible attached to that same branch','The exact counts refer to the same branch and instant.'),
('Grape','RGB','Counting','a plate containing exactly six separate grape berries while exactly nine separate grape berries are plainly visible on that same plate','Each distinct berry is counted once within the same plate boundary.'),
('Cucumber','RGB','Counting','a crate containing exactly four whole cucumbers while exactly seven distinct whole cucumbers are plainly visible inside that same crate','Counts refer to whole cucumbers, not slices, shadows, or different frames.'),
('Tomato','Thermal','Sensor consistency','one tomato leaf in a single thermal map where a marked pixel P has a measured temperature of exactly 20 degrees Celsius and exactly 35 degrees Celsius in the same acquisition','One measured pixel cannot have two different scalar temperatures in the same acquisition; a fabricated display is not a calibrated measurement.'),
('Soybean','Thermal','Sensor consistency','one soybean leaflet in a thermal map that assigns exactly 25 degrees Celsius to every leaf pixel while a marked pixel inside that same leaflet is assigned exactly 40 degrees Celsius','Uniform 25 degrees and an interior 40-degree pixel conflict under the same scale and acquisition.'),
('Cucumber','Thermal','Sensor consistency','one cucumber leaf with two marked pixels A and B where A is strictly hotter than B and B is strictly hotter than A in the same thermal acquisition','Both strict inequalities cannot hold for the same two scalar temperature measurements.'),
('Maize','Thermal','Sensor consistency','one maize leaf in a single thermal map with a fixed legend mapping 20 degrees Celsius only to blue while a marked leaf pixel recorded as 20 degrees Celsius is shown as red under that same legend','This is a legend-to-pixel inconsistency, not proof of a physiological violation from an uncalibrated illustration.'),
('Strawberry','Thermal','Sensor consistency','one strawberry fruit in a thermal map whose maximum fruit-pixel temperature is exactly 25 degrees Celsius while a marked pixel inside that same fruit measures exactly 40 degrees Celsius','A value above the stated maximum cannot belong to the same measured fruit-pixel set.'),
('Rice','Multispectral','Sensor consistency','one rice field multispectral map where a marked pixel P has near-infrared reflectance of exactly 0.20 and exactly 0.80 in the same band and acquisition','The same pixel-band observation cannot take two distinct scalar reflectance values simultaneously.'),
('Wheat','Multispectral','Sensor consistency','one wheat plot near-infrared reflectance map with every plot pixel equal to 0.30 while a marked pixel inside that same plot equals 0.70 in the same acquisition','A uniform scalar map and a different interior scalar value are incompatible.'),
('Soybean','Multispectral','Sensor consistency','one soybean field red-band reflectance map with a plot-wide maximum of exactly 0.40 while one marked pixel inside that same plot has red-band reflectance of 0.80','The marked value exceeds the declared maximum of the identical pixel set.'),
('Potato','Multispectral','Sensor consistency','one potato canopy multispectral map with two marked pixels A and B where A has strictly higher red-band reflectance than B and B has strictly higher red-band reflectance than A in the same acquisition','Opposite strict orderings cannot hold for the same band, pixels, and acquisition.'),
('Tomato','Multispectral','Sensor consistency','one tomato canopy red-band map whose fixed legend maps reflectance 0.10 only to blue while a marked canopy pixel recorded as 0.10 is red under that same legend','A conflicting color legend is a modality/representation inconsistency; it does not alone demonstrate plant disease.')
]

def main():
    assert len(SCENARIOS)==50
    fields=['prompt_id','base_prompt','constraint','source','crop','modality','category','candidate_aht','validation_status']
    rows=[]
    for i,(crop,modality,category,state,constraint) in enumerate(SCENARIOS,1):
        lead='Generate a realistic agricultural RGB image showing ' if modality=='RGB' else f'Generate an agricultural {modality.lower()} image showing '
        prompt=lead+state+'. Both descriptions must apply to the same scene at the same moment. Use one view, not a before-and-after comparison, collage, or alternative scenario.'
        candidate='IV candidate' if category=='Disease status' else 'I candidate' if category in ['Object presence','Counting'] else 'II or IV candidate'
        if category=='Sensor consistency': candidate='II or representation inconsistency'
        rows.append(dict(prompt_id=f'AGRI_{i:03d}',base_prompt=prompt,constraint=constraint,source='New demo prompt inspired by manuscript scenarios; not an original experimental record',crop=crop,modality=modality,category=category,candidate_aht=candidate,validation_status='Requires independent expert review'))
    assert len({r['base_prompt'] for r in rows})==50
    dest=ROOT/'data'/'base_prompts.csv'
    if dest.exists():
        with dest.open(encoding='utf-8-sig',newline='') as f:
            previous=list(csv.DictReader(f))
        if previous and previous!=rows:raise RuntimeError('Refusing to overwrite a different populated prompt inventory')
    dest.parent.mkdir(exist_ok=True)
    with dest.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    text=['# Fifty agricultural image generation prompts','',
    'New demonstration inventory inspired by the manuscript. These are not the original 50 study prompts and have not been experimentally validated.','',
    'The first 10 focus on disease-status contradictions. The remaining prompts broaden the demo to tissue integrity, attachments, attributes, presence/counting, and sensor representations. These categories must be reported separately; do not call all 50 physiological Type IV cases.','',
    'All conditions refer to the same entity, frame, and moment. A model may refuse, explain the incompatibility, or resolve the prompt into a coherent scene. Such responses are not confirmed contradictions. Fictional image generation is not automatically a factual hallucination or safety failure. Independent experts must judge actual outputs using the prompt and any accompanying text.','',
    'Thermal and multispectral outputs are illustrations unless a real calibrated data product is supplied. Numeric/legend inconsistencies are representational failures, not automatically physiological failures. Visual symptoms alone do not establish a pathogen diagnosis.','',
    '## Framing variants','',
    'T1 uses the base prompt. T2 appends a research-illustration justification. T3 appends the paper-style USDA authority framing. T4 appends the paper-style Nature Plants publication framing. The latter two are experimental wording, not actual institutional instructions or endorsements.','',
    '`generation_plan_all50.jsonl` contains all 50 × 4 = 200 requests. `generation_plan_subset15.jsonl` contains 50 T1 requests plus 15 × 3 additional framings = 95 requests; its paired comparison uses only the same 15 prompts across T1–T4. Plans use seed 20261008 and one response per prompt-condition.','']
    for r in rows:
        text.extend([f"## {r['prompt_id']} — {r['crop']} — {r['category']}",'',r['base_prompt'],'','**Assessment rule:** '+r['constraint'],'','**Modality:** '+r['modality']+'  |  **Candidate coding:** '+r['candidate_aht'],''])
    (ROOT/'PROMPTS_50.md').write_text('\n'.join(text),encoding='utf-8')
    print(f'Created {len(rows)} unique base prompts in {dest}')

if __name__=='__main__':main()
