import re
import torch



from .utils import get_hparams_from_file, load_checkpoint
from .models import SynthesizerTrn

def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result





# class TTS():
#     def __init__(self, model_path, model_config) -> None:
#        hps = get_hparams_from_file(model_config)
#        model = SynthesizerTrn