import numpy as np

from cavesim.flyable import closest_on_triangles


def test_closest_point_on_triangle_matches_brute_force():
    rng = np.random.default_rng(1)
    a, b, c = (rng.normal(size=(300, 3)) for _ in range(3))
    p = 2 * rng.normal(size=(300, 3))
    exact = np.linalg.norm(closest_on_triangles(p, a, b, c) - p, axis=1)
    u = rng.random((5000, 2))
    u = np.where(u.sum(1, keepdims=True) > 1, 1 - u, u)          # uniform on the triangle
    brute = np.array([np.linalg.norm(a[i] + u[:, :1] * (b[i] - a[i]) + u[:, 1:] * (c[i] - a[i]) - p[i], axis=1).min()
                      for i in range(300)])
    assert np.all(exact <= brute + 1e-9)                         # never farther than any point on it
    assert np.all(brute - exact < 0.1)                           # and the samples come close to it

