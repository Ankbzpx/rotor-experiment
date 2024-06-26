# Reference: https://marctenbosch.com/quaternions/
from attr import define

import numpy as np
import jax
from jax.tree_util import register_pytree_node_class, tree_map
from typing_extensions import Self
from icecream import ic


@register_pytree_node_class
@define
class G3:
    c: float
    c_0: float
    c_1: float
    c_2: float
    c_01: float
    c_12: float
    c_20: float
    c_012: float

    def tree_flatten(self):
        children = (self.c, self.c_0, self.c_1, self.c_2, self.c_01, self.c_12,
                    self.c_20, self.c_012)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def from_pairs(cls, a: np.ndarray, b: np.ndarray):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        c = np.dot(a, b)
        c_01 = a[0] * b[1] - a[1] * b[0]
        c_12 = a[1] * b[2] - a[2] * b[1]
        c_20 = a[2] * b[0] - a[0] * b[2]

        return cls(c, 0, 0, 0, c_01, c_12, c_20, 0)

    @staticmethod
    def geometric_product(u, v):
        C = np.outer(jax.tree.leaves(v), jax.tree.leaves(u))

        c = C[0, 0] + C[1, 1] + C[2, 2] + C[3, 3] - C[4, 4] - C[5, 5] - C[
            6, 6] - C[7, 7]
        c_0 = C[0, 1] + C[1, 0] + C[2, 4] - C[4, 2] - C[3, 6] + C[6, 3] - C[
            5, 7] - C[7, 5]
        c_1 = C[0, 2] + C[2, 0] - C[1, 4] + C[4, 1] + C[3, 5] - C[5, 3] - C[
            6, 7] - C[7, 6]
        c_2 = C[0, 3] + C[3, 0] + C[1, 6] - C[6, 1] - C[2, 5] + C[5, 2] - C[
            4, 7] - C[7, 4]
        c_01 = C[0, 4] + C[4, 0] - C[1, 2] + C[2, 1] + C[3, 7] + C[7, 3] + C[
            5, 6] - C[6, 5]
        c_12 = C[0, 5] + C[5, 0] + C[1, 7] + C[7, 1] - C[2, 3] + C[3, 2] - C[
            4, 6] + C[6, 4]
        c_20 = C[0, 6] + C[6, 0] - C[1, 3] + C[3, 1] + C[2, 7] + C[7, 2] + C[
            4, 5] - C[5, 4]
        c_012 = C[0, 7] + C[7, 0] + C[1, 5] + C[5, 1] + C[2, 6] + C[6, 2] + C[
            3, 4] + C[4, 3]

        return G3(c, c_0, c_1, c_2, c_01, c_12, c_20, c_012)

    @property
    def dag(self) -> Self:
        return self.conjugate()

    def conjugate(self) -> Self:
        return G3(self.c, 0, 0, 0, -self.c_01, -self.c_12, -self.c_20, 0)

    def rotate(self, a):
        # R a R^\dag
        a = G3(0, a[0], a[1], a[2], 0, 0, 0, 0)
        a_rot = R * a * R.dag
        ic(a_rot)
        assert np.isclose(a_rot.c_012, 0)
        return np.array([a_rot.c_0, a_rot.c_1, a_rot.c_2])

    def __add__(self, other) -> Self:
        if isinstance(other, G3):
            return tree_map(lambda x, y: x + y, self, other)
        else:
            return NotImplemented

    def __mul__(self, other) -> Self:
        if isinstance(other, (float, int)):
            return tree_map(lambda x: x * other, self)
        elif isinstance(other, G3):
            return G3.geometric_product(self, other)
        else:
            return NotImplemented

    def __rmul__(self, other) -> Self:
        if isinstance(other, (float, int)):
            return self * other
        elif isinstance(other, G3):
            return G3.geometric_product(other, self)
        else:
            return NotImplemented

    def __truediv__(self, other):
        return (1 / other) * self

    def __neg__(self):
        return -1 * self


def angle_between(a, b):
    theta = np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
    return np.rad2deg(theta)


if __name__ == '__main__':
    # The rotation occurs at plane a wedge b
    v = np.array([0, 2, -1.5])
    a = np.array([0, 0, -1])
    b = np.array([0, 1, 0])

    ic(angle_between(a, b))

    R = G3.from_pairs(a, b)
    v2 = R.rotate(v)

    ic(angle_between(v, v2))
