import json
from .trajnetpp_data import SceneRow, TrackRow_unihumotion

def trajnet_tracks(row):
    x = row.x
    y = row.y

    h = row.h
    w = row.w
    l = row.l
    rot_z = row.rot_z
    bb_left = row.bb_left
    bb_top = row.bb_top
    bb_width = row.bb_width
    bb_height = row.bb_height

    x0 = row.x0
    x1 = row.x1
    x2 = row.x2
    x3 = row.x3
    x4 = row.x4
    x5 = row.x5
    x6 = row.x6
    x7 = row.x7
    x8 = row.x8
    x9 = row.x9
    x10 = row.x10
    x11 = row.x11
    x12 = row.x12
    x13 = row.x13
    x14 = row.x14
    x15 = row.x15
    x16 = row.x16
    x17 = row.x17
    x18 = row.x18
    x19 = row.x19
    x20 = row.x20
    x21 = row.x21
    x22 = row.x22
    x23 = row.x23
    x24 = row.x24
    x25 = row.x25
    x26 = row.x26
    x27 = row.x27
    x28 = row.x28
    x29 = row.x29
    x30 = row.x30
    x31 = row.x31
    x32 = row.x32
    x33 = row.x33
    x34 = row.x34
    x35 = row.x35
    x36 = row.x36
    x37 = row.x37
    x38 = row.x38


    y0 = row.y0
    y1 = row.y1
    y2 = row.y2
    y3 = row.y3
    y4 = row.y4
    y5 = row.y5
    y6 = row.y6
    y7 = row.y7
    y8 = row.y8
    y9 = row.y9
    y10 = row.y10
    y11 = row.y11
    y12 = row.y12
    y13 = row.y13
    y14 = row.y14
    y15 = row.y15
    y16 = row.y16
    y17 = row.y17
    y18 = row.y18
    y19 = row.y19
    y20 = row.y20
    y21 = row.y21
    y22 = row.y22
    y23 = row.y23
    y24 = row.y24
    y25 = row.y25
    y26 = row.y26
    y27 = row.y27
    y28 = row.y28
    y29 = row.y29
    y30 = row.y30
    y31 = row.y31
    y32 = row.y32
    y33 = row.y33
    y34 = row.y34
    y35 = row.y35
    y36 = row.y36
    y37 = row.y37
    y38 = row.y38


    z0 = row.z0
    z1 = row.z1
    z2 = row.z2
    z3 = row.z3
    z4 = row.z4
    z5 = row.z5
    z6 = row.z6
    z7 = row.z7
    z8 = row.z8
    z9 = row.z9
    z10 = row.z10
    z11 = row.z11
    z12 = row.z12
    z13 = row.z13
    z14 = row.z14
    z15 = row.z15
    z16 = row.z16
    z17 = row.z17
    z18 = row.z18
    z19 = row.z19
    z20 = row.z20
    z21 = row.z21
    z22 = row.z22
    z23 = row.z23
    z24 = row.z24
    z25 = row.z25
    z26 = row.z26
    z27 = row.z27
    z28 = row.z28
    z29 = row.z29
    z30 = row.z30
    z31 = row.z31
    z32 = row.z32
    z33 = row.z33
    z34 = row.z34
    z35 = row.z35
    z36 = row.z36
    z37 = row.z37
    z38 = row.z38



    xx0 = row.xx0
    xx1 = row.xx1
    xx2 = row.xx2
    xx3 = row.xx3
    xx4 = row.xx4
    xx5 = row.xx5
    xx6 = row.xx6
    xx7 = row.xx7
    xx8 = row.xx8
    xx9 = row.xx9
    xx10 = row.xx10
    xx11 = row.xx11
    xx12 = row.xx12
    xx13 = row.xx13
    xx14 = row.xx14
    xx15 = row.xx15
    xx16 = row.xx16
    xx17 = row.xx17
    xx18 = row.xx18
    xx19 = row.xx19
    xx20 = row.xx20
    xx21 = row.xx21
    xx22 = row.xx22
    xx23 = row.xx23
    xx24 = row.xx24
    xx25 = row.xx25
    xx26 = row.xx26
    xx27 = row.xx27
    xx28 = row.xx28
    xx29 = row.xx29
    xx30 = row.xx30
    xx31 = row.xx31
    xx32 = row.xx32
    xx33 = row.xx33
    xx34 = row.xx34
    xx35 = row.xx35
    xx36 = row.xx36
    xx37 = row.xx37
    xx38 = row.xx38
    
    yy0 = row.yy0
    yy1 = row.yy1
    yy2 = row.yy2
    yy3 = row.yy3
    yy4 = row.yy4
    yy5 = row.yy5
    yy6 = row.yy6
    yy7 = row.yy7
    yy8 = row.yy8
    yy9 = row.yy9
    yy10 = row.yy10
    yy11 = row.yy11
    yy12 = row.yy12
    yy13 = row.yy13
    yy14 = row.yy14
    yy15 = row.yy15
    yy16 = row.yy16
    yy17 = row.yy17
    yy18 = row.yy18
    yy19 = row.yy19
    yy20 = row.yy20
    yy21 = row.yy21
    yy22 = row.yy22
    yy23 = row.yy23
    yy24 = row.yy24
    yy25 = row.yy25
    yy26 = row.yy26
    yy27 = row.yy27
    yy28 = row.yy28
    yy29 = row.yy29
    yy30 = row.yy30
    yy31 = row.yy31
    yy32 = row.yy32
    yy33 = row.yy33
    yy34 = row.yy34
    yy35 = row.yy35
    yy36 = row.yy36
    yy37 = row.yy37
    yy38 = row.yy38
    

    if row.prediction_number is None:
        return json.dumps({'track': {'f': row.frame, 'p': row.pedestrian, 'x': x, 'y': y, 
                                    'h': h, 'w': w, 'l': l, 'rot_z': rot_z, 'bb_left': bb_left, 'bb_top': bb_top, 'bb_width': bb_width, 'bb_height': bb_height,
                                    'x0': x0, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6, 'x7': x7, 'x8': x8, 'x9': x9, 'x10': x10, 'x11': x11, 'x12': x12, 'x13': x13, 'x14': x14, 'x15': x15, 'x16': x16, 'x17': x17, 'x18': x18, 'x19': x19, 'x20': x20, 'x21': x21, 'x22': x22, 'x23': x23, 'x24': x24, 'x25': x25, 'x26': x26, 'x27': x27, 'x28': x28, 'x29': x29, 'x30': x30, 'x31': x31, 'x32': x32, 'x33': x33, 'x34': x34, 'x35': x35, 'x36': x36, 'x37': x37, 'x38': x38, 
                                    'y0': y0, 'y1': y1, 'y2': y2, 'y3': y3, 'y4': y4, 'y5': y5, 'y6': y6, 'y7': y7, 'y8': y8, 'y9': y9, 'y10': y10, 'y11': y11, 'y12': y12, 'y13': y13, 'y14': y14, 'y15': y15, 'y16': y16, 'y17': y17, 'y18': y18, 'y19': y19, 'y20': y20, 'y21': y21, 'y22': y22, 'y23': y23, 'y24': y24, 'y25': y25, 'y26': y26, 'y27': y27, 'y28': y28, 'y29': y29, 'y30': y30, 'y31': y31, 'y32': y32, 'y33': y33, 'y34': y34, 'y35': y35, 'y36': y36, 'y37': y37, 'y38': y38,
                                    'z0': z0, 'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'z5': z5, 'z6': z6, 'z7': z7, 'z8': z8, 'z9': z9, 'z10': z10, 'z11': z11, 'z12': z12, 'z13': z13, 'z14': z14, 'z15': z15, 'z16': z16, 'z17': z17, 'z18': z18, 'z19': z19, 'z20': z20, 'z21': z21, 'z22': z22, 'z23': z23, 'z24': z24, 'z25': z25, 'z26': z26, 'z27': z27, 'z28': z28, 'z29': z29, 'z30': z30, 'z31': z31, 'z32': z32, 'z33': z33, 'z34': z34, 'z35': z35, 'z36': z36, 'z37': z37, 'z38': z38, 
                                    'xx0': xx0, 'xx1': xx1, 'xx2': xx2, 'xx3': xx3, 'xx4': xx4, 'xx5': xx5, 'xx6': xx6, 'xx7': xx7, 'xx8': xx8, 'xx9': xx9, 'xx10': xx10, 'xx11': xx11, 'xx12': xx12, 'xx13': xx13, 'xx14': xx14, 'xx15': xx15, 'xx16': xx16, 'xx17': xx17, 'xx18': xx18, 'xx19': xx19, 'xx20': xx20, 'xx21': xx21, 'xx22': xx22, 'xx23': xx23, 'xx24': xx24, 'xx25': xx25, 'xx26': xx26, 'xx27': xx27, 'xx28': xx28, 'xx29': xx29, 'xx30': xx30, 'xx31': xx31, 'xx32': xx32, 'xx33': xx33, 'xx34': xx34, 'xx35': xx35, 'xx36': xx36, 'xx37': xx37, 'xx38': xx38, 
                                    'yy0': yy0, 'yy1': yy1, 'yy2': yy2, 'yy3': yy3, 'yy4': yy4, 'yy5': yy5, 'yy6': yy6, 'yy7': yy7, 'yy8': yy8, 'yy9': yy9, 'yy10': yy10, 'yy11': yy11, 'yy12': yy12, 'yy13': yy13, 'yy14': yy14, 'yy15': yy15, 'yy16': yy16, 'yy17': yy17, 'yy18': yy18, 'yy19': yy19, 'yy20': yy20, 'yy21': yy21, 'yy22': yy22, 'yy23': yy23, 'yy24': yy24, 'yy25': yy25, 'yy26': yy26, 'yy27': yy27, 'yy28': yy28, 'yy29': yy29, 'yy30': yy30, 'yy31': yy31, 'yy32': yy32, 'yy33': yy33, 'yy34': yy34, 'yy35': yy35, 'yy36': yy36, 'yy37': yy37, 'yy38': yy38, 
                                    }})
    return json.dumps({'track': {'f': row.frame, 'p': row.pedestrian, 'x': x, 'y': y, 
                                    'h': h, 'w': w, 'l': l, 'rot_z': rot_z, 'bb_left': bb_left, 'bb_top': bb_top, 'bb_width': bb_width, 'bb_height': bb_height,
                                    'x0': x0, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6, 'x7': x7, 'x8': x8, 'x9': x9, 'x10': x10, 'x11': x11, 'x12': x12, 'x13': x13, 'x14': x14, 'x15': x15, 'x16': x16, 'x17': x17, 'x18': x18, 'x19': x19, 'x20': x20, 'x21': x21, 'x22': x22, 'x23': x23, 'x24': x24, 'x25': x25, 'x26': x26, 'x27': x27, 'x28': x28, 'x29': x29, 'x30': x30, 'x31': x31, 'x32': x32, 'x33': x33, 'x34': x34, 'x35': x35, 'x36': x36, 'x37': x37, 'x38': x38, 
                                    'y0': y0, 'y1': y1, 'y2': y2, 'y3': y3, 'y4': y4, 'y5': y5, 'y6': y6, 'y7': y7, 'y8': y8, 'y9': y9, 'y10': y10, 'y11': y11, 'y12': y12, 'y13': y13, 'y14': y14, 'y15': y15, 'y16': y16, 'y17': y17, 'y18': y18, 'y19': y19, 'y20': y20, 'y21': y21, 'y22': y22, 'y23': y23, 'y24': y24, 'y25': y25, 'y26': y26, 'y27': y27, 'y28': y28, 'y29': y29, 'y30': y30, 'y31': y31, 'y32': y32, 'y33': y33, 'y34': y34, 'y35': y35, 'y36': y36, 'y37': y37, 'y38': y38, 
                                    'z0': z0, 'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'z5': z5, 'z6': z6, 'z7': z7, 'z8': z8, 'z9': z9, 'z10': z10, 'z11': z11, 'z12': z12, 'z13': z13, 'z14': z14, 'z15': z15, 'z16': z16, 'z17': z17, 'z18': z18, 'z19': z19, 'z20': z20, 'z21': z21, 'z22': z22, 'z23': z23, 'z24': z24, 'z25': z25, 'z26': z26, 'z27': z27, 'z28': z28, 'z29': z29, 'z30': z30, 'z31': z31, 'z32': z32, 'z33': z33, 'z34': z34, 'z35': z35, 'z36': z36, 'z37': z37, 'z38': z38, 
                                    'xx0': xx0, 'xx1': xx1, 'xx2': xx2, 'xx3': xx3, 'xx4': xx4, 'xx5': xx5, 'xx6': xx6, 'xx7': xx7, 'xx8': xx8, 'xx9': xx9, 'xx10': xx10, 'xx11': xx11, 'xx12': xx12, 'xx13': xx13, 'xx14': xx14, 'xx15': xx15, 'xx16': xx16, 'xx17': xx17, 'xx18': xx18, 'xx19': xx19, 'xx20': xx20, 'xx21': xx21, 'xx22': xx22, 'xx23': xx23, 'xx24': xx24, 'xx25': xx25, 'xx26': xx26, 'xx27': xx27, 'xx28': xx28, 'xx29': xx29, 'xx30': xx30, 'xx31': xx31, 'xx32': xx32, 'xx33': xx33, 'xx34': xx34, 'xx35': xx35, 'xx36': xx36, 'xx37': xx37, 'xx38': xx38,
                                    'yy0': yy0, 'yy1': yy1, 'yy2': yy2, 'yy3': yy3, 'yy4': yy4, 'yy5': yy5, 'yy6': yy6, 'yy7': yy7, 'yy8': yy8, 'yy9': yy9, 'yy10': yy10, 'yy11': yy11, 'yy12': yy12, 'yy13': yy13, 'yy14': yy14, 'yy15': yy15, 'yy16': yy16, 'yy17': yy17, 'yy18': yy18, 'yy19': yy19, 'yy20': yy20, 'yy21': yy21, 'yy22': yy22, 'yy23': yy23, 'yy24': yy24, 'yy25': yy25, 'yy26': yy26, 'yy27': yy27, 'yy28': yy28, 'yy29': yy29, 'yy30': yy30, 'yy31': yy31, 'yy32': yy32, 'yy33': yy33, 'yy34': yy34, 'yy35': yy35, 'yy36': yy36, 'yy37': yy37, 'yy38': yy38,
                                'prediction_number': row.prediction_number,
                                'scene_id': row.scene_id}})





def trajnet_scenes(row):
    return json.dumps(
        {'scene': {'id': row.scene, 'p': row.pedestrian, 's': row.start, 'e': row.end,
                   'fps': row.fps, 'tag': row.tag}})


def trajnet(row):
    if isinstance(row, TrackRow_unihumotion):
        return trajnet_tracks(row)
    if isinstance(row, SceneRow):
        return trajnet_scenes(row)


    raise Exception('unknown row type')
