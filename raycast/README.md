## Install

```
cd raycast
python setup.py develop
```

Usage

Code in `test.py`
```py
import mesh_raycast

triangles = np.array([
    # first face
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],

    # second face
    [0.0, 0.0, -3.0],
    [1.0, 0.0, -3.0],
    [0.0, 1.0, -3.0],

    # third face that is not present
    [5.0, 0.0, 0.0],
    [6.0, 0.0, 0.0],
    [5.0, 1.0, 0.0],

], dtype='f4')

res = mesh_raycast.raycast(
    source=(0.0, 0.0, 2.0),
    direction=mesh_raycast.normalize((0.0, 0.0, -1.0)),
    mesh=triangles,
)
# sort by  distance
res.sort(key=lambda x: x['distance'])

```

The `result` is a list of objects with the following keys:

```py
[
    {'coeff': (0.0, 0.0),
      'distance': 2.0,
      'dot': 1.0,
      'face': 0,
      'normal': (0.0, 0.0, 1.0),
      'point': (0.0, 0.0, 0.0)},
    
     {'coeff': (0.0, 0.0),
      'distance': 5.0,
      'dot': 1.0,
      'face': 1,
      'normal': (0.0, 0.0, 1.0),
      'point': (0.0, 0.0, -3.0)}
]
```

- `face` is the index of the triangle from mesh
- `point` is the point in world coordinates where the ray and the triangle intersects
- `normal` is the normal of the triangle
- `coeff` is a pair of coefficients from the internal calculations
- `distance` is the distance between point and source
- `dot` is the dot product of -direction and normal

### sorting the result

```py
sorted(result, key=lambda x: x['distance'])
```

### filtering the result

```py
first_matching_face = min(result, key=lambda x: x['distance'])['face']
```

[//]: # ()
[//]: # (```py)

[//]: # (non_backfacing = filter&#40;lambda x: x['dot'] > 0.0, result&#41;)

[//]: # (```)
