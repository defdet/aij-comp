import subprocess
import sys, os
import transformers
import torch
import librosa
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_captions(audio_content, model_caption, tokenizer, caption_feature_extractor):
    audio, sampling_rate = librosa.load(audio_content, sr=caption_feature_extractor.sampling_rate)
    features = caption_feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
    style_prefix = "clotho > caption: "
    style_prefix_tokens = tokenizer("", text_target=style_prefix, return_tensors="pt", add_special_tokens=False).labels
    outputs = model_caption.generate(
        inputs=features.to(device),
        forced_ac_decoder_ids=style_prefix_tokens.to(device),
        max_length=512,
    )
    captions = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].replace('clotho > caption:', '')
    return captions


def format_replic(replic, model_caption, tokenizer, caption_feature_extractor, no_space=False):
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
            captions = get_captions(content, model_caption, tokenizer, caption_feature_extractor)
            prompt.append("Since you cannot listen to audio, here's audio captioning: {captions}")
    prompt.append("<end_of_utterance>")
    prompt.append('\nAssistant:')
    return prompt

def get_text_ans(prompt, text_model, processor):
    inputs = processor(prompt, return_tensors="pt").to(device)
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
    generated_ids = text_model.generate(**inputs, bad_words_ids=bad_words_ids, max_length=40)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_text

def get_ans(test_history, full_history):
        return full_history.replace(test_history, '')

def get_perp_of_text(text, model, tokenizer):
    encodings = tokenizer(text, return_tensors="pt").to(device)
    max_length = 1024
    stride = 512
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=input_ids, image_attention_mask = encodings.image_attention_mask, pixel_values=encodings.pixel_values, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean()).detach().cpu().item()
    return ppl
