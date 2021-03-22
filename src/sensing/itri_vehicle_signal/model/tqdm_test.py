from tqdm import tqdm
import time

pbar = tqdm(["a","b","c", "d"])

for c in pbar:
	time.sleep(1)
	pbar.update(1)
	pbar.set_description("Processing %s" % c)


