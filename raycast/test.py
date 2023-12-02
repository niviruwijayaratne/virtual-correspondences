import numpy as np
import mesh_raycast
from pprint import pprint


if __name__ == '__main__':
    triangles = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],

        [0.0, 0.0, -3.0],
        [1.0, 0.0, -3.0],
        [0.0, 1.0, -3.0],

        [5.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [5.0, 1.0, 0.0],

    ], dtype='f4')


    res = mesh_raycast.raycast(
        source=(0.0, 0.0, 2.0),
        # direction=mesh_raycast.normalize((0.1, 0.2, -1.0)),
        direction=mesh_raycast.normalize((0.0, 0.0, -1.0)),
        mesh=triangles,
    )

    pprint(res)

# res2 = mesh_raycast.iraycast((0.0, 0.0, 2.0), mesh_raycast.normalize((0.1, 0.2, -1.0)), triangles, index)
# print(res2)
