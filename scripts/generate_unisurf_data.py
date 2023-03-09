import os
from tqdm import tqdm
from pathlib import Path
# (Path(os.getcwd()) / "unisurf_data_pc").mkdir(parents=True)

for i in tqdm(range(5, 10)): # 0 to (num_scenes-1)
    os.system("python scripts/generate_data_parallel.py --scene pile --object-set pile/train --num-grasps 5 --save-scene ../potato-net/data/scene{0:03d} --random".format(i))