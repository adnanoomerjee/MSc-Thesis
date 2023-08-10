import jax.numpy as jp
from brax import base

maze_cell_values = -jp.array([
      [0., 19., 18., 17., 18.],
      [1.,  2., 15., 16., 19.],
      [4.,  3., 14., 13., 20.],
      [5.,  8.,  9., 12., 21.],
      [6.,  7., 10., 11., 22.],
    ])

next_cell = jp.array([
    [[0, 0], [0, 2], [0, 3], [1, 3], [0, 3]],
    [[0, 0], [1, 0], [2, 2], [1, 2], [0, 4]],
    [[2, 1], [1, 1], [2, 3], [3, 3], [1, 4]],
    [[2, 0], [4, 1], [3, 1], [4, 3], [2, 4]],
    [[3, 0], [4, 0], [3, 2], [4, 2], [3, 4]],
], dtype=jp.int32)

cell_centroid = jp.array([
    [[-8,  8], [-4,  8], [0,  8], [4,  8], [8,  8]],
    [[-8,  4], [-4,  4], [0,  4], [4,  4], [8,  4]],
    [[-8,  0], [-4,  0], [0,  0], [4,  0], [8,  0]],
    [[-8, -4], [-4, -4], [0, -4], [4, -4], [8, -4]],
    [[-8, -8], [-4, -8], [0, -8], [4, -8], [8, -8]],
], dtype=jp.int32)

def get_maze_coordinates(state: base.State):
    xy = jp.mean(state.x.pos[:, :2], axis=0)
    xpos, ypos = xy
    column = (1 + (xpos + 10) // 4 + (abs(xpos) == 2))
    row = (5 - (ypos + 10) // 4 + (abs(ypos) == 2))
    coords =  (row, column)
    value = maze_cell_values[row.astype(int)-1, column.astype(int)-1]
    next_cell  = next_cell[row.astype(int)-1, column.astype(int)-1, :]
    return coords, value, xy, next_cell