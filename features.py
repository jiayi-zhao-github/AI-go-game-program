
import numpy as np
import go
from p1 import product


P = 8

def make_onehot(feature, planes):
    onehot_features = np.zeros(feature.shape + (planes,), dtype=np.uint8)
    capped = np.minimum(feature, planes)
    onehot_index_offsets = np.arange(0, product(onehot_features.shape), planes) + capped.ravel()

    nonzero_elements = (capped != 0).ravel()
    nonzero_index_offsets = onehot_index_offsets[nonzero_elements] - 1
    onehot_features.ravel()[nonzero_index_offsets] = 1
    return onehot_features

def planes(num_planes):
    def deco(f):
        f.planes = num_planes
        return f
    return deco

@planes(3)
def stone_color_feature(position):
    board = position.board
    features = np.zeros([go.N, go.N, 3], dtype=np.uint8)
    if position.to_play == go.BLACK:
        features[board == go.BLACK, 0] = 1
        features[board == go.WHITE, 1] = 1
    else:
        features[board == go.WHITE, 0] = 1
        features[board == go.BLACK, 1] = 1

    features[board == go.EMPTY, 2] = 1
    return features

@planes(1)
def ones_feature(position):
    return np.ones([go.N, go.N, 1], dtype=np.uint8)

@planes(P)
def recent_move_feature(position):
    onehot_features = np.zeros([go.N, go.N, P], dtype=np.uint8)
    for i, player_move in enumerate(reversed(position.recent[-P:])):
        _, move = player_move
        if move is not None:
            onehot_features[move[0], move[1], i] = 1
    return onehot_features

@planes(P)
def liberty_feature(position):
    position.get_liberties()
    return make_onehot(position.get_liberties(), P)

@planes(P)
def would_capture_feature(position):
    features = np.zeros([go.N, go.N], dtype=np.uint8)
    for g in position.lib_tracker.groups.values():
        if g.color == position.to_play:
            continue
        if len(g.liberties) == 1:
            last_lib = list(g.liberties)[0]

            features[last_lib] += len(g.stones)
    return make_onehot(features, P)

DEFAULT_FEATURES = [
    stone_color_feature,
    ones_feature,
    liberty_feature,
    recent_move_feature,
    would_capture_feature,
]

def extract_features(position, features=DEFAULT_FEATURES):
    return np.concatenate([feature(position) for feature in features], axis=2)

def bulk_extract_features(positions, features=DEFAULT_FEATURES):
    num_positions = len(positions)
    num_planes = sum(f.planes for f in features)
    output = np.zeros([num_positions, go.N, go.N, num_planes], dtype=np.uint8)
    for i, pos in enumerate(positions):
        output[i] = extract_features(pos, features=features)

    return output

#
# tmp = go.Position()
#
# # tmp7 = tmp.lib_tracker.groups.values()
# # print(tmp7)
#
# # tmp9 = would_capture_feature(tmp)
# # print(tmp9)
#
# tmp2=tmp.play_move((0,0),go.WHITE)
#
#
# # tmp7 = tmp2.lib_tracker.groups.values()
# # print(tmp7)
# # tmp9 = would_capture_feature(tmp2)
# # print(tmp9)
# tmp3=tmp2.play_move((1,0),go.BLACK)
#
# # tmp7 = tmp3.lib_tracker.groups.values()
# # print(tmp7)
# # tmp9 = would_capture_feature(tmp3)
# # print(tmp9)
# tmp4 = tmp3.flip_playerturn()
# tmp10 =would_capture_feature(tmp3)
#
# print(tmp4.lib_tracker.groups.values())
# print(tmp10)
#
# tmp4=tmp3.play_move((0,1),go.BLACK)
#
#
# # tmp7 = tmp4.lib_tracker.groups.values()
# # print(tmp7)
# # tmp9 = would_capture_feature(tmp4)
# # print(tmp9)
#
#
#
#
#
#
# tmp00 = 1
#
#
# tmp5=tmp4.play_move((1,0),go.BLACK)
#
#
#
# tmp6=tmp5.play_move((1,1),go.BLACK)
#
#
# tmp8 = 1
#
#

