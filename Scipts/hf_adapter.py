_CACHE={}
def predict(request,model,revision=None):
    import torch, transformers
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    if 'path' not in request:raise ValueError('This adapter supports interpretation only')
    if not revision:raise ValueError('Supply the verified immutable checkpoint revision')
    key=(model,revision)
    if key not in _CACHE:
        processor=AutoProcessor.from_pretrained(model,revision=revision,trust_remote_code=False)
        net=AutoModelForImageTextToText.from_pretrained(model,revision=revision,trust_remote_code=False,device_map='auto',torch_dtype='auto')
        net.eval();_CACHE[key]=(processor,net)
    processor,net=_CACHE[key];messages=[];images=[]
    for example in request.get('exemplars',[]):
        images.append(Image.open(example['path']).convert('RGB'))
        messages.extend([{'role':'user','content':[{'type':'image'},{'type':'text','text':request['prompt']}]},{'role':'assistant','content':[{'type':'text','text':example['label']}]}])
    images.append(Image.open(request['path']).convert('RGB'))
    messages.append({'role':'user','content':[{'type':'image'},{'type':'text','text':request['prompt']}]})
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[text],images=images,return_tensors='pt').to(net.device)
    with torch.inference_mode():output=net.generate(**inputs,max_new_tokens=512,do_sample=False)
    generated=output[:,inputs['input_ids'].shape[1]:] if not net.config.is_encoder_decoder else output
    answer=processor.batch_decode(generated,skip_special_tokens=True)[0]
    return {'output':answer,'metadata':{'checkpoint':model,'revision':revision,'transformers':transformers.__version__,'torch':torch.__version__,'decoding':'greedy','max_new_tokens':512,'image_count':len(images)}}
