class _Model:
    sr = 22050
    def to(self, device):
        return self
    def synthesise(self, text, ref_speech=None, exaggeration_factor=0.5):
        import torch
        return torch.tensor([0.0])
    def get_ref_speech(self, wav_path):
        return None

class TTS:
    @staticmethod
    def from_pretrained(model_dir):
        return _Model()
