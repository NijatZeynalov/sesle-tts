from vits import utils
from vits import commons
from vits.models import SynthesizerTrn
import soundfile as sf
import numpy as np
import os
import re
import tempfile
import torch
from num2azerbaijani import convert as num2aze
import random

class TextMapper(object):
    def __init__(self, vocab_file):
        self.symbols = [
            x.replace("\n", "") for x in open(vocab_file, encoding="utf-8").readlines()
        ]
        self.SPACE_ID = self.symbols.index(" ")
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_sequence(self, text, cleaner_names):
        """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
        Returns:
        List of integers corresponding to the symbols in the text
        """
        sequence = []
        clean_text = text.strip()
        for symbol in clean_text:
            symbol_id = self._symbol_to_id[symbol]
            sequence += [symbol_id]
        return sequence

    def uromanize(self, text, uroman_pl):
        iso = "xxx"
        with tempfile.NamedTemporaryFile() as tf, tempfile.NamedTemporaryFile() as tf2:
            with open(tf.name, "w") as f:
                f.write("\n".join([text]))
            cmd = f"perl " + uroman_pl
            cmd += f" -l {iso} "
            cmd += f" < {tf.name} > {tf2.name}"
            os.system(cmd)
            outtexts = []
            with open(tf2.name) as f:
                for line in f:
                    line = re.sub(r"\s+", " ", line).strip()
                    outtexts.append(line)
            outtext = outtexts[0]
        return outtext

    def get_text(self, text, hps):
        text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def filter_oov(self, text, lang=None):
        text = self.preprocess_char(text, lang=lang)
        val_chars = self._symbol_to_id
        txt_filt = "".join(list(filter(lambda x: x in val_chars, text)))
        return txt_filt

    def preprocess_char(self, text, lang=None):
        """
        Special treatement of characters in certain languages
        """
        if lang == "ron":
            text = text.replace("ț", "ţ")
            print(f"{lang} (ț -> ţ): {text}")
        return text

def synthesize(text, lang, speed):

    if speed is None:
        speed = 1.0

    lang_code = lang.split(":")[0].strip()

    vocab_file = r'model_files/vocab.txt'
    config_file = r'model_files/config.json'
    g_pth = r'model_files/G_100000.pth'


    device = torch.device("cpu")

    print(f"Run inference with {device}")

    assert os.path.isfile(config_file), f"{config_file} doesn't exist"
    hps = utils.get_hparams_from_file(config_file)
    text_mapper = TextMapper(vocab_file)
    net_g = SynthesizerTrn(
        len(text_mapper.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    )
    net_g.to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(g_pth, net_g, None)

    is_uroman = hps.data.training_files.split(".")[-1] == "uroman"

    if is_uroman:
        uroman_dir = "uroman"
        assert os.path.exists(uroman_dir)
        uroman_pl = os.path.join(uroman_dir, "bin", "uroman.pl")
        text = text_mapper.uromanize(text, uroman_pl)

    text = text.lower()
    text = text_mapper.filter_oov(text, lang=lang)
    stn_tst = text_mapper.get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        hyp = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                noise_scale=0.3,
                noise_scale_w=0.3,
                length_scale=1.0 / speed,
            )[0][0, 0]
            .cpu()
            .float()
            .numpy()
        )

    return (hps.data.sampling_rate, hyp), text


def generate_voice(string, speed = 1.2):
    numbers = re.findall(r'\d+', string)

    numbers = [int(num) for num in numbers]
    result = {}

    for number in numbers:
        result[number] = num2aze(number)

    for num, word in result.items():
        string = string.replace(str(num), word)

    string = string.replace('.', '. ')

    gra = synthesize(string,'azj-script_latin',speed)

    sampling_rate = gra[0][0]
    audio_data = np.array(gra[0][1])

    number = random.randint(1000, 9999)
    output_file = f"audio_temp/{gra[1].replace(' ', '_')[:10]+ str(number)}.wav"

    # Save the audio_temp file
    sf.write(output_file, audio_data, sampling_rate)
    return output_file
