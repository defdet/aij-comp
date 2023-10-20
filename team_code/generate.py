import torch, transformers
from transformers import IdeficsForVisionText2Text, AutoProcessor
from configs import WhisperForAudioCaptioning
from utils import format_replic, get_text_ans, get_ans

IDEFICS_DIR = "models/idefics-9b-instruct"
WHISPER_CAPTION_DIR = "models/whisper-caption"
def setup_model_and_tokenizer():

    model_idefics = IdeficsForVisionText2Text.from_pretrained(IDEFICS_DIR, torch_dtype=torch.bfloat16).cuda()
    processor_idefics = AutoProcessor.from_pretrained(IDEFICS_DIR)
    model_caption = WhisperForAudioCaptioning.from_pretrained(WHISPER_CAPTION_DIR).cuda()
    tokenizer = transformers.WhisperTokenizer.from_pretrained(WHISPER_CAPTION_DIR, language="en", task="transcribe")
    caption_feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(WHISPER_CAPTION_DIR)
    model_caption.eval()
    model_idefics.eval()

    return (model_idefics, model_caption), (processor_idefics, caption_feature_extractor)

def generate_text(model, tokenizer, cur_query_list, history_list=None):
    no_space = False if history_list is None else True
    history_list = [[]] if history_list is None else history_list[0]
    model_idefics, model_caption = model
    processor_idefics, caption_feature_extractor = tokenizer
    prompt = format_replic(cur_query_list, model_caption, tokenizer, no_space)
    for data in prompt:
        history_list[0].append(data)
    full_history = get_text_ans(history_list, model_idefics, processor_idefics)
    answer = get_ans(history_list, full_history)
    return answer, history_list

def get_ppl(model, tokenizer, cur_query_tuple, history=None):
    ppl = 0
    return ppl
