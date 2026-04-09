import numpy as np

class TileCoder:
    def __init__(self, n_tilings, n_tiles, state_low, state_high):
        self.n_tilings = n_tilings
        self.n_tiles = n_tiles
        self.state_low = state_low
        self.state_high = state_high
        self.n_features = n_tilings * n_tiles * n_tiles

        self.offsets = np.array([
            [i/ n_tilings, i/ n_tilings] 
            for i in range(n_tilings)
        ])
    def encode(self, state):
        features = np.zeros(self.n_features)
        for i, offset in enumerate(self.offsets):
            scaled = (state - self.state_low) / (self.state_high - self.state_low)
            scaled = (scaled + offset / self.n_tiles)* self.n_tilings

            tile_idx = np.floor(scaled).astype(int)
            tile_idx = np.clip(tile_idx, 0, self.n_tiles - 1)

            flat_idx = i * (self.n_tiles ** 2) + tile_idx[0] * self.n_tiles + tile_idx[1]
            features[flat_idx] = 1
        return features