import os
from pathlib import Path
import shutil
import random


root = Path('.')
ls = ['MT_Free', 'MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Uneven']
# rebuild dataset
if os.path.exists('dataset'):
    shutil.rmtree(root/'dataset', ignore_errors=True)
os.makedirs('dataset')
for f in ls:
    os.makedirs(Path('dataset', f))
# shuffle dataset
for c in ls:
    imgs = list((root/c/'Imgs').glob('*.jpg'))
    random.shuffle(imgs)
    for idx, img in enumerate(imgs):
        shutil.copyfile(img, root/'dataset'/c/(str(idx)+'.jpg'))
