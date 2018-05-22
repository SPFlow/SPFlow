'''
Created on March 26, 2018

@author: Alejandro Molina
'''
import glob
import json
import os
import platform

import numpy as np
from natsort import natsorted

np.set_printoptions(precision=50)

path = os.path.dirname(__file__)
OS_name = platform.system()

if __name__ == '__main__':


    times = {}

    i = 0
    for fpath in natsorted(glob.glob(path + '/spns/*/tf_timelines2/*.json')):

        dspath = os.path.dirname(os.path.dirname(fpath))

        exp = os.path.basename(dspath)


        if exp not in times:
            times[exp] = []

        with open(fpath, "r") as ffile:
            traceEvents = json.load(ffile)["traceEvents"]
            #run_time = sum([o["dur"] for o in traceEvents if "dur" in o])

            #run_time = max([o["ts"] + o["dur"] for o in traceEvents if "ts" in o and "dur" in o]) - min([o["ts"] for o in traceEvents if "ts" in o])

            trace_events_time = [o for o in traceEvents if "ts" in o]
            maximum = max([o["ts"] for o in trace_events_time])
            maximum = max([o["ts"] + o["dur"] for o in trace_events_time if "dur" in o]+[maximum])
            minimum = min([o["ts"] for o in trace_events_time])
            #print(exp, maximum, minimum, maximum - minimum)
            run_time = maximum - minimum


            times[exp].append(run_time)

        i += 1

        if i == 10:
            pass
            # break

    for k in times.keys():
        del times[k][0]

    #print(times)
    for k in sorted(times.keys()):
        v = times[k]
        del v[0]
        print(k, ((sum(v) / len(v)) )*1000)
