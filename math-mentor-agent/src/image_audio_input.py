from pix2text import Pix2Text
from faster_whisper import WhisperModel


def img_to_text(path):
    p2t = Pix2Text.from_config()

    img_path = path 
    text = p2t.recognize(img_path, file_type='text_formula',)
    return text
    
        

#%%
def audio_to_text(path):
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(path)
    text = ''
    for segment in segments:
        text = text + segment.text

    return text    
