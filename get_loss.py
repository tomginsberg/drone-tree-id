from sys import argv 
import json

infile = argv[1]
outfile = argv[2]
if len(argv) < 4:
	keyw = ["total_loss", "loss_cls", "loss_box_reg", "loss_mask", "loss_rpn_cls", "loss_rpn_loc", "time"]
else:
	keyw = [argv[3]]

with open(infile, 'r') as f:
	data = f.read()

import re
outs = []
for k in keyw:
	outs.append(re.findall(f'{keyw}: (\d.\d*)', data))

with open(outfile, 'w') as f:
	f.write(json.dumps(outs))
