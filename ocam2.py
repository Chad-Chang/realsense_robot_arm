#! /usr/bin/env python2

import liboCams
import cv2
import time
import sys

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-a", "--alltest", dest="alltest",
                  action='store_true', help="test all resolution in playtime")
parser.add_option("-t", "--time", dest="playtime", default=1, type="int",
                  help="playtime for streaming [sec] intValue, 0 means forever")
parser.add_option("-i", "--index", dest="index", default=0, type="int",
                  help="index of resolusion mode")

(options, args) = parser.parse_args()

devpath = liboCams.FindCamera('oCam')
if devpath is None:
    exit()

test = liboCams.oCams(devpath, verbose=1)

print
'Format List'
fmtlist = test.GetFormatList()
for fmt in fmtlist:
    print
    '\t', fmt

print
'Control List'
ctrlist = test.GetControlList()
for key in ctrlist:
    print
    '\t', key, '\tID:', ctrlist[key]

test.Close()

if options.alltest is True:
    len_range = range(len(fmtlist))
else:
    if options.index >= len(fmtlist):
        print
        'INDEX error', options.index, 'index reset to default value 0'
        options.index = 0
    len_range = {options.index}

for i in len_range:
    test = liboCams.oCams(devpath, verbose=0)

    print
    'SET', i, fmtlist[i]
    test.Set(fmtlist[i])
    name = test.GetName()
    test.Start()

    # example code for camera control
    # val = test.GetControl(ctrlist['Exposure (Absolute)'])
    # test.SetControl(ctrlist['Exposure (Absolute)'], 2)

    start_time = time.time()
    stop_time = start_time + float(options.playtime)

    frame_cnt = 0
    while True:
        if name == 'oCamS-1CGN-U':
            frame = test.GetFrame(mode=1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BAYER_GB2BGR)
        elif name == 'oCam-1CGN-U':
            frame = test.GetFrame()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BAYER_GB2BGR)
        elif name == 'oCam-1MGN-U':
            frame = test.GetFrame()
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif name == 'oCam-5CRO-U':
            frame = test.GetFrame()
            rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
        else:
            frame = test.GetFrame()
            rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)

        cv2.imshow(test.cam.card, rgb)
        char = cv2.waitKey(1)
        if char == 27:
            break
        if time.time() > stop_time:
            break
        frame_cnt += 1

    print
    'Result Frame Per Second:', frame_cnt / (time.time() - start_time)
    test.Stop()
    cv2.destroyAllWindows()
    char = cv2.waitKey(1)
    test.Close()




