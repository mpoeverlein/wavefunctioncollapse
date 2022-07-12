#!/usr/bin/python3

'''
This file contains the Tile class.
An object of this class has the following attributes and methods.
 - initialize by supplying whether tile is collapsed and which states the tile can assume
'''

import numpy as np

class Tile():
    def __init__(self, states, collapsed=False):
        self.states = states
        self.normalize_states()

        self.collapsed = collapsed
        if self.collapsed:
            self.collapse()

    def __repr__(self):
        collapsed = 'NOT' if not self.collapsed else ''
        return f'Tile {collapsed} collapsed., Possible States: {self.states}'

    def __len__(self): return len(self.states)
    def __lt__(self, othertile): return self.get_entropy() < othertile.get_entropy()
    def __le__(self, othertile): return self.get_entropy() <= othertile.get_entropy()
   
    def collapse(self, state=None):
        self.collapsed = True
        self.result = state or np.random.choice(tuple(self.states), p=tuple(self.states.values()))
        self.states = {k: 1 if k == self.result else 0 for k in self.states}

    def update(self, states):
        print(f'UPDATE: {states}')
        self.states = states

    def get_state(self):
        return self.result if self.collapsed else None
   
    def get_entropy(self):
        if self.collapsed: return 0
        return -1 * np.sum([p*np.log2(p) if p > 0 else 0 for k,p in self.states.items()])

    def normalize_states(self):
        total = sum([p for k,p in self.states.items()])
        self.states = {k: p/total for k,p in self.states.items()}


if __name__ == '__main__':
    tile = Tile({'beach': 0.9, 'ocean': 0.1, 'forest': 1.0})
    assert len(tile) == 3
    print(tile)
    print(f'Entropy of the tile: {tile.get_entropy():.2f}')
    assert tile.get_entropy() == -1 * (0.45*np.log2(0.45) + 0.05*np.log2(0.05) + 0.5*np.log2(0.5))

