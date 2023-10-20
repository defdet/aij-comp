import transformers, librosa

def get_captions(audio_content, model_caption, tokenizer, caption_feature_extractor):
    audio, sampling_rate = librosa.load(audio_content, sr=caption_feature_extractor.sampling_rate)
    features = caption_feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
    style_prefix = "clotho > caption: "
    style_prefix_tokens = tokenizer("", text_target=style_prefix, return_tensors="pt", add_special_tokens=False).labels
    outputs = model_caption.generate(
        inputs=features.to(model_caption.device),
        forced_ac_decoder_ids=style_prefix_tokens,
        max_length=256,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


def format_replic(replic, model_caption, tokenizer, no_space=False):
    prompt = []
    for data in replic:
        data_type, content = data['type'], data['content']
        if data_type == 'text':
            if no_space:
                prompt.append(f"User: {content}")
            else:
                prompt.append(f"\nUser: {content}")
        elif data_type == 'image':
            prompt.append(content)
        elif data_type == 'audio':
            captions = get_captions(content)
            prompt.append("Since you cannot listen to audio, here's audio captioning: {captions}")
    prompt.append("<end_of_utterance>")
    prompt.append('\nAssistant:')
    return prompt

def get_text_ans(prompt, text_model, processor):
    inputs = processor(prompt, return_tensors="pt").cuda()
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
    generated_ids = text_model.generate(**inputs, bad_words_ids=bad_words_ids, max_length=256)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_text

def get_ans(test_history, full_history):
        lines1 = test_history.splitlines()
        lines2 = full_history.splitlines()
    
        differences = []
        for line in lines2:
            if line not in lines1:
                differences.append(line)
    
        return '\n'.join(differences)
