#!/usr/bin/python3

'''
This file contains the functions needed to update the tile array.
'''

import numpy as np
import itertools as it

vectorized_collapsed = np.vectorize(lambda tile: tile.collapsed)
vectorized_len = np.vectorize(lambda tile: len(tile))

def vectorized_entropy(tile_info):
    # for some reason, this function does not like being np.vectorize()'d !
    entropy = np.zeros_like(tile_info)
    for iy, ix in np.ndindex(entropy.shape):
        entropy[iy,ix] = tile_info[iy,ix].get_entropy()

    return entropy

def out_of_bounds(neighbor, tile_array):
    return len(neighbor[neighbor < 0]) or len(neighbor[neighbor >= len(tile_array)])

def get_allowed_set(relation_set, relative_neighbor, current_tile):
    relative_x, relative_y = list(relative_neighbor)
    return relation_set[current_tile.result][relative_x+1, relative_y+1]

def get_overlapping_probabilities(allowed_sets, overlap_function=np.product):
    probabilities = {k: overlap_function([i[k] for i in allowed_sets]) for k in allowed_sets[0]}
    probabilities_sum = sum([p for k,p in probabilities.items()])
    return {k: p/probabilities_sum for k,p in probabilities.items()}

def calculate_entropy(tile_array, position, naive_state_list, relation_set):
    xx, yy = position
    # skip all blocks that are only none
    if np.all(tile_array[max(xx-1,0):min(xx+2, tile_array.shape[0]),
                       max(yy-1,0):min(yy+2, tile_array.shape[1])] == None):
        return naive_state_list

    # get neighbors
    allowed_sets = [naive_state_list]
    for relative_neighbor in it.product([-1, 0, 1], repeat=2):
        relative_neighbor = np.array(relative_neighbor)
        absolute_neighbor = relative_neighbor + position
        if out_of_bounds(absolute_neighbor, tile_array): continue

        neighbor_x, neighbor_y = list(absolute_neighbor)
        if not tile_array[neighbor_x, neighbor_y].collapsed: continue
        current_tile = tile_array[neighbor_x, neighbor_y]
        allowed_sets.append(get_allowed_set(relation_set, relative_neighbor, current_tile))

    allowed_sets = [i for i in allowed_sets if i is not None]
    return get_overlapping_probabilities(allowed_sets)

def update_tile_array(tile_array_old, naive_state_list, relation_set, last_update=None):
    tile_array = tile_array_old.copy() # just to be sure that the original array remains unchanged

    if last_update is not None:
        updatable_tiles = [np.array([last_update[0]]), np.array([last_update[1]])]
    else:
        updatable_tiles = np.where(vectorized_collapsed(tile_array))

    for x in zip(*updatable_tiles):
        for relative_neighbor_tile in it.product([-1, 0, 1], repeat=2):
            absolute_neighbor = x + np.array(relative_neighbor_tile)
            if out_of_bounds(absolute_neighbor, tile_array): continue
            current_tile = tile_array[absolute_neighbor[0],absolute_neighbor[1]]
            if current_tile.collapsed: continue
            tile_array[absolute_neighbor[0],absolute_neighbor[1]].update(calculate_entropy(tile_array, absolute_neighbor, naive_state_list, relation_set))

    # find tiles with least entropy not zero
    minimum_entropy = tile_array[np.where(~vectorized_collapsed(tile_array))].min().get_entropy()
    minimum_entropy_tiles = np.where(vectorized_entropy(tile_array) == minimum_entropy)

    # choose one of those tiles at random to collapse
    target_tile = np.random.randint(len(minimum_entropy_tiles[0]))
    target_position = [minimum_entropy_tiles[0][target_tile], minimum_entropy_tiles[1][target_tile]]
    target_x, target_y = target_position
    tile_array[target_x, target_y].collapse()

    return tile_array, target_position

def propagate_tile_array(tile_array, naive_state_list, relation_set, last_update=None):
    while np.count_nonzero(~vectorized_collapsed(tile_array)):
        tile_array, last_update = update_tile_array(tile_array.copy(), naive_state_list, relation_set, last_update=last_update)

    return tile_array

