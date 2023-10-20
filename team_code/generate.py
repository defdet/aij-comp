import torch, transformers
from transformers import IdeficsForVisionText2Text, AutoProcessor
from configs import WhisperForAudioCaptioning

IDEFICS_DIR = "models/idefics-9b-instruct"
WHISPER_CAPTION_DIR = "models/whisper-caption"
device = "cuda" if torch.cuda.is_available() else "cpu"
def setup_model_and_tokenizer():

    model_idefics = IdeficsForVisionText2Text.from_pretrained(IDEFICS_DIR, torch_dtype=torch.bfloat16).to(device)
    processor_idefics = AutoProcessor.from_pretrained(IDEFICS_DIR)
    model_caption = WhisperForAudioCaptioning.from_pretrained(WHISPER_CAPTION_DIR)
    tokenizer = transformers.WhisperTokenizer.from_pretrained(WHISPER_CAPTION_DIR, language="en", task="transcribe")
    caption_feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(WHISPER_CAPTION_DIR)

    return (model_idefics, model_caption), (processor_idefics, caption_feature_extractor)

def generate_text(model, tokenizer, cur_query_list, history_list=None):
    setup_output = setup_model_and_tokenizer()
    model_idefics, model_caption = setup_output[0]
    processor_idefics, caption_feature_extractor = setup_output[1]
    model_idefics, model_caption = model_idefics.cuda(), model_caption.cuda()
    answer, history_on_current_round = '', ''
    return answer, history_on_current_round

def get_ppl(model, tokenizer, cur_query_tuple, history=None):
    ppl = 0
    return ppl