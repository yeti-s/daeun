import io
import torch 
from scipy.io import wavfile
from .utils import get_hparams_from_file, load_checkpoint
from .models import SynthesizerTrn
from .text.symbols import symbols
from .text import text_to_sequence
from .commons import intersperse

class TTSModel():
    def __init__(self, model_path, model_config, playback_speed=1., device='cuda') -> None:
        self.bytes_io = io.BytesIO()
        self.hps = get_hparams_from_file(model_config)
        self.playback_speed = playback_speed
        self.device = torch.device(device)
        self.model = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model).to(device)
        self.model.eval()
        load_checkpoint(model_path, self.model, None)

    def __ndarr_to_wav__(self, ndarr):
        wavfile.write(self.bytes_io, self.hps.data.sampling_rate, ndarr)
        wav_bin = self.bytes_io.getvalue()
        return wav_bin

    @torch.no_grad()
    def generate(self, text):
        print(text)
        stn_tst = text_to_sequence(f'[KO]{text}[KO]', self.hps.data.text_cleaners)
        if self.hps.data.add_blank:
            stn_tst = intersperse(stn_tst, 0)
        stn_tst = torch.LongTensor(stn_tst)
        
        x_tst = stn_tst.to(self.device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
        audio_ndarr = self.model.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=self.playback_speed)[0][0,0].data.cpu().float().numpy()
        return self.__ndarr_to_wav__(audio_ndarr)
    