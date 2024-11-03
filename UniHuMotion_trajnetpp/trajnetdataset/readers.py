""" Read Raw files as TrackRows """

import json
import os
import xml.etree.ElementTree

import numpy as np
import scipy.interpolate

from .trajnetpp_data import TrackRow_unihumotion


def UHM(line):
    line = [e for e in line.split(',') if e != '']
    
    int_elements = [int(float(line[0])), int(float(line[1]))]
    float_elements = list(map(float, line[2:]))
    
    return TrackRow_unihumotion(*int_elements, *float_elements)


def get_trackrows(line):
    line = json.loads(line)
    track = line.get('track')
    if track is not None:
        return TrackRow_unihumotion(track['f'], track['p'], track['x'], track['y'],
                                    track['h'], track['w'], track['l'], track['rot_z'],
                                    track['bb_left'], track['bb_top'], track['bb_width'], track['bb_height'],
                                    track['x0'], track['x1'], track['x2'], track['x3'], track['x4'], track['x5'], track['x6'], track['x7'], track['x8'], track['x9'], track['x10'], track['x11'], track['x12'], track['x13'], track['x14'], track['x15'],  track['x16'], track['x17'], track['x18'], track['x19'], track['x20'], track['x21'], track['x22'], track['x23'], track['x24'], track['x25'], track['x26'], track['x27'], track['x28'], track['x29'], track['x30'], track['x31'], track['x32'], track['x33'], track['x34'], track['x35'], track['x36'], track['x37'], track['x38'], 
                                    track['y0'], track['y1'], track['y2'], track['y3'], track['y4'], track['y5'], track['y6'], track['y7'], track['y8'], track['y9'], track['y10'], track['y11'], track['y12'], track['y13'], track['y14'], track['y15'],  track['y16'], track['y17'], track['y18'], track['y19'], track['y20'], track['y21'], track['y22'], track['y23'], track['y24'], track['y25'], track['y26'], track['y27'], track['y28'], track['y29'], track['y30'], track['y31'], track['y32'], track['y33'], track['y34'], track['y35'], track['y36'], track['y37'], track['y38'],
                                    track['z0'], track['z1'], track['z2'], track['z3'], track['z4'], track['z5'], track['z6'], track['z7'], track['z8'], track['z9'], track['z10'], track['z11'], track['z12'], track['z13'], track['z14'], track['z15'],  track['z16'], track['z17'], track['z18'], track['z19'], track['z20'], track['z21'], track['z22'], track['z23'], track['z24'], track['z25'], track['z26'], track['z27'], track['z28'], track['z29'], track['z30'], track['z31'], track['z32'], track['z33'], track['z34'], track['z35'], track['z36'], track['z37'], track['z38'], 
                                    track['xx0'], track['xx1'], track['xx2'], track['xx3'], track['xx4'], track['xx5'], track['xx6'], track['xx7'], track['xx8'], track['xx9'], track['xx10'], track['xx11'], track['xx12'], track['xx13'], track['xx14'], track['xx15'],  track['xx16'], track['xx17'], track['xx18'], track['xx19'], track['xx20'], track['xx21'], track['xx22'], track['xx23'], track['xx24'], track['xx25'], track['xx26'], track['xx27'], track['xx28'], track['xx29'], track['xx30'], track['xx31'], track['xx32'], track['xx33'], track['xx34'], track['xx35'], track['xx36'], track['xx37'], track['xx38'],
                                    track['yy0'], track['yy1'], track['yy2'], track['yy3'], track['yy4'], track['yy5'], track['yy6'], track['yy7'], track['yy8'], track['yy9'], track['yy10'], track['yy11'], track['yy12'], track['yy13'], track['yy14'], track['yy15'],  track['yy16'], track['yy17'], track['yy18'], track['yy19'], track['yy20'], track['yy21'], track['yy22'], track['yy23'], track['yy24'], track['yy25'], track['yy26'], track['yy27'], track['yy28'], track['yy29'], track['yy30'], track['yy31'], track['yy32'], track['yy33'], track['yy34'], track['yy35'], track['yy36'], track['yy37'], track['yy38'],
                        track.get('prediction_number'))
    return None



