import scipy
from audiocraft.models import musicgen
from audiocraft.utils.notebook import display_audio
import torch

device = int(input("Choose device:\n1.CPU\n2.GPU\n"))

if device ==1:
    model = musicgen.MusicGen.get_pretrained('medium',device='cpu')
    while True:
        prompt = input("Write melody description:")
        time = int(input("Specify length of the music in seconds:"))
        model.set_generation_params(duration=time)
        res = model.generate([prompt,], progress=True)
        scipy.io.wavfile.write(f"cpu_{prompt.replace(' ','_')}.wav", rate=32000, data=res[0,0].numpy())
elif device==2:
    model = musicgen.MusicGen.get_pretrained('medium', device='cuda')
    while True:
        prompt = input("Write melody description:")
        time = int(input("Specify length of the music in seconds:"))
        model.set_generation_params(duration=time)
        res = model.generate([prompt,], progress=True)
        dat = res.detach().cpu()
        scipy.io.wavfile.write(f"gpu_{prompt.replace(' ','_')}.wav", rate=32000, data=dat[0,0].numpy())
else:
    print("Invalid option")
