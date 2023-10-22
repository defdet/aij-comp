import subprocess
import sys
def install():
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'transformers', '--no-index', '--find-links', 'file:///packages/transformers'])
install()
import torch, transformers
from transformers import IdeficsForVisionText2Text, AutoProcessor
from configs import WhisperForAudioCaptioning
from utils import format_replic, get_text_ans, get_ans, get_perp_of_text
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IDEFICS_DIR = "models\idefics-test"
WHISPER_CAPTION_DIR = "models\captioning-whisper"
def setup_model_and_tokenizer():

    model_idefics = IdeficsForVisionText2Text.from_pretrained(IDEFICS_DIR, torch_dtype=torch.bfloat16).to(device)
    processor_idefics = AutoProcessor.from_pretrained(IDEFICS_DIR)
    model_caption = WhisperForAudioCaptioning.from_pretrained(WHISPER_CAPTION_DIR).to(device)
    tokenizer = transformers.WhisperTokenizer.from_pretrained(WHISPER_CAPTION_DIR, language="en", task="transcribe")
    caption_feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(WHISPER_CAPTION_DIR)
    model_caption.eval()
    model_idefics.eval()

    return (model_idefics, model_caption), (processor_idefics, caption_feature_extractor, tokenizer)

def generate_text(model, tokenizer, cur_query_list, history_list=None):
    no_space = False if history_list is None else True
    history_list = [[]] if history_list is None else history_list
    model_idefics, model_caption = model
    processor_idefics, caption_feature_extractor, tokenizer_whisper = tokenizer
    prompt = format_replic(cur_query_list, model_caption, tokenizer_whisper, caption_feature_extractor, no_space)
    for data in prompt:
        history_list[0].append(data)
    full_history = get_text_ans(history_list, model_idefics, processor_idefics)
    answer = get_ans(history_list, full_history)
    return answer, history_list

def get_ppl(model, tokenizer, cur_query_tuple, history=None):
    history = "" if history is None else history
    cur_query_list = cur_query_tuple[0]
    model_idefics, model_caption = model
    processor_idefics, caption_feature_extractor, tokenizer_whisper = tokenizer
    prompt = format_replic(cur_query_list, model_caption, tokenizer_whisper, caption_feature_extractor, True)
    for data in prompt:
        history += data
    ppl = get_perp_of_text(history)
    return ppl, history
