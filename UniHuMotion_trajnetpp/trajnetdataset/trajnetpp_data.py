from collections import namedtuple
SceneRow = namedtuple('Row', ['scene', 'pedestrian', 'start', 'end', 'fps', 'tag'])
SceneRow.__new__.__defaults__ = (None, None, None, None, None, None)

## extract UniHuMotion
TrackRow_unihumotion = namedtuple('Row', ['frame', 'pedestrian', 'x', 'y', 'h', 'w', 'l', 'rot_z', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 
                                          'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x31', 'x32', 'x33', 'x34', 'x35', 'x36', 'x37', 'x38',
                                          'y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12', 'y13', 'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20', 'y21', 'y22', 'y23', 'y24', 'y25', 'y26', 'y27', 'y28', 'y29', 'y30', 'y31', 'y32', 'y33', 'y34', 'y35', 'y36', 'y37', 'y38',
                                          'z0', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12', 'z13', 'z14', 'z15', 'z16', 'z17', 'z18', 'z19', 'z20', 'z21', 'z22', 'z23', 'z24', 'z25', 'z26', 'z27', 'z28', 'z29', 'z30', 'z31', 'z32', 'z33', 'z34', 'z35', 'z36', 'z37', 'z38',
                                          'xx0', 'xx1', 'xx2', 'xx3', 'xx4', 'xx5', 'xx6', 'xx7', 'xx8', 'xx9', 'xx10', 'xx11', 'xx12', 'xx13', 'xx14', 'xx15', 'xx16', 'xx17', 'xx18', 'xx19', 'xx20', 'xx21', 'xx22', 'xx23', 'xx24', 'xx25', 'xx26', 'xx27', 'xx28', 'xx29', 'xx30', 'xx31', 'xx32', 'xx33', 'xx34', 'xx35', 'xx36', 'xx37', 'xx38',
                                          'yy0', 'yy1', 'yy2', 'yy3', 'yy4', 'yy5', 'yy6', 'yy7', 'yy8', 'yy9', 'yy10', 'yy11', 'yy12', 'yy13', 'yy14', 'yy15', 'yy16', 'yy17', 'yy18', 'yy19', 'yy20', 'yy21', 'yy22', 'yy23', 'yy24', 'yy25', 'yy26', 'yy27', 'yy28', 'yy29', 'yy30', 'yy31', 'yy32', 'yy33', 'yy34', 'yy35', 'yy36', 'yy37', 'yy38',
                                          'prediction_number', 'scene_id'])
TrackRow_unihumotion.__new__.__defaults__ = (None, None, None, None, None, None, None, None, None, None, None, None, 
                                             None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 
                                             None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                                             None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                                             None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                                             None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                                             None, None)

