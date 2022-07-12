#!/usr/bin/python3

'''
World building: Create map containing Beach, Forest, Ocean, etc.

To Do: Find probabilities that make for nice distributions
'''

import tile
import propagate as prop
import visualize as vis

import time
import numpy as np


probabilities = {
        'ocean': {'ocean': 0.5, 'beach': 0.5, 'forest': 0.0001},
        'beach': {'ocean': 0.5, 'beach': 0.5, 'forest': 0.1},
        'forest': {'ocean': 0.0001, 'beach': 0.2, 'forest': 0.8},
        }

relation_set = {k: np.full((3,3), probabilities[k]) for k in probabilities}

naive_state_list = {k: 1.0/len(probabilities) for k in probabilities}

colors = {
        'ocean': (6,66,115,1), 
        'beach': (255,238,173,1), 
        'forest': (77,76,28,1), 
        'desert': (255,204,92,1),
        None: (0,0,0,1),
        }

N = 10
tile_array = np.array([[tile.Tile(naive_state_list) for i in np.arange(N)] for y in np.arange(N)])
tile_array[4,4].collapse(state='beach')
tile_array[4,5].collapse(state='beach')
tile_array[5,4].collapse(state='beach')
tile_array[5,5].collapse(state='beach')
#last_update = np.array([5,5])

last_update = None
tile_array, last_update = prop.update_tile_array(tile_array.copy(), naive_state_list, relation_set, last_update=last_update)
vis.plot_annotated_entropy(tile_array)

#exit()

start_time = time.time()

tile_array = prop.propagate_tile_array(tile_array, naive_state_list, relation_set, last_update=last_update)

end_time = time.time()
print(f'Time needed: {end_time-start_time:.1f} seconds')

vis.plot_single_grid(tile_array, colors)

