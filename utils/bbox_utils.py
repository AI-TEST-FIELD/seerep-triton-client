import numpy as np

def cxcy2xyxy(box):
    xtl = int(box[0] - (box[2]/2))
    ytl = int(box[1] - (box[3]/2))
    xbr = int(box[0] + (box[2]/2))
    ybr = int(box[1] + (box[3]/2))

    return [xtl, ytl, xbr, ybr]

def xyxy2cxcy(box):
    print('hold')