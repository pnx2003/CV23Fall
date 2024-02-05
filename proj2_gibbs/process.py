from PIL import Image
import os

# Source and destination directories
source_dir = './pde/result/'
destination_dir = './save/'

names = ["stone","sce","room"]
sizes = ["small","big"]
norms = ["L1","L2"]
seq = [0, 20, 90]
for name in names:
    for size in sizes:
        for norm in norms:
            for s in seq:
                source_file = os.path.join(source_dir,f'{name}/{size}/{norm}/pde_{s}.bmp')
                destination_file = os.path.join(destination_dir,f'pde_{name}_{size}_{norm}_{s}.png')
                img = Image.open(source_file)
                img.save(destination_file, 'PNG')
        

