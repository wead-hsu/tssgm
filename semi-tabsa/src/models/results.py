from collections import defaultdict

rest  = defaultdict(dict)
rest['tdlstm']             = {'acc': [77.03, 78.01, 77.12, 77.75, 76.50], 'f1': [66.03, 65.21, 65.36, 67.84, 65.54]}
rest['tdlstm(cbow)']       = {'acc': [77.03, 77.21, 76.14, 77.12, 76.94], 'f1': [64.90, 64.68, 65.49, 65.84, 65.99]}
rest['tssvae(tdlstm)']     = {'acc': [78.10, 77.75, 77.48, 77.93, 78.10], 'f1': [68.33, 66.25, 66.55, 68.03, 67.53]}

rest['memnet']             = {'acc': [78.80, 78.90, 78.90, 78.40, 78.40], 'f1': [67.80, 67.40, 67.70, 66.70, 64.30]}
rest['memnet(cbow)']       = {}
rest['tssvae(memnet)']     = {'acc': [80.20, 79.70, 80.20, 80.40, 80.40], 'f1': [69.30, 68.70, 69.60, 69.90, 69.80]}

rest['bilstmattg']         = {'acc': [80.00, 79.80, 79.60, 79.40, 79.90], 'f1': [68.80, 68.81, 67.80, 67.80, 67.60]}
rest['bilstmattg(cbow)']   = {}
rest['tssvae(bilstmattg)'] = {'acc': [80.00, 80.10, 80.50, 80.33, 80.20], 'f1': [69.40, 70.10, 71.60, 70.30, 70.10]}

rest['ian']                = {}
rest['ian(cbow)']          = {}
rest['tssvae(ian)']        = {}

lapt  = defaultdict(dict)
lapt['tdlstm']             = {'acc': [69.40, 67.82, 68.29, 67.98, 68.61], 'f1': [64.48, 61.19, 61.89, 62.23, 62.34]}
lapt['tdlstm(cbow)']       = {'acc': [67.67, 67.82, 66.25, 68.45, 67.35], 'f1': [61.28, 60.42, 58.08, 61.79, 59.98]}
lapt['tssvae(tdlstm)']     = {'acc': [69.24, 68.61, 68.61, 68.92, 70.03], 'f1': [62.61, 63.07, 62.38, 63.60, 63.66]}

lapt['memnet']             = {'acc': [70.80, 70.30, 70.20, 69.80, 70.30], 'f1': [64.40, 63.80, 65.20, 63.10, 65.40]}
lapt['memnet(cbow)']       = {}
lapt['tssvae(memnet)']     = {'acc': [71.70, 71.70, 71.70, 70.50, 70.50], }

lapt['bilstmattg']         = {'acc': [74.50, 74.30, 74.60, 74.30, 73.60], 'f1': [69.70, 69.30, 70.40, 69.50, 68.80]}
lapt['bilstmattg(cbow)']   = {}
lapt['tssvae(bilstmattg)'] = {}

lapt['ian']                = {}
lapt['ian(cbow)']          = {}
lapt['tssvae(ian)']        = {}

import numpy as np

rest_summary = defaultdict(dict)
for k in rest:
    for g in rest[k]:
        rest_summary[k][g + '_mean'] = np.mean(rest[k][g])
        rest_summary[k][g + '_std'] = np.std(rest[k][g])

lapt_summary = defaultdict(dict)
for k in lapt:
    for g in lapt[k]:
        lapt_summary[k][g + '_mean'] = np.mean(lapt[k][g])
        lapt_summary[k][g + '_std'] = np.std(lapt[k][g])

from pprint import pprint
pprint(rest_summary)
pprint(lapt_summary)
