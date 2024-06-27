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

    def __sub__(self, other) -> Self:
        return self + (-1 * other)


@register_pytree_node_class
@define
class Bivector3:
    c_01: float
    c_12: float
    c_20: float

    def tree_flatten(self):
        children = (self.c_01, self.c_12, self.c_20)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def from_wedge(cls, a: np.ndarray, b: np.ndarray):
        c_01 = a[0] * b[1] - a[1] * b[0]
        c_12 = a[1] * b[2] - a[2] * b[1]
        c_20 = a[2] * b[0] - a[0] * b[2]

        return cls(c_01, c_12, c_20)

    @staticmethod
    def bivector_bivector_product(B_a, B_b) -> Self:
        B_out = B_a.to_G3() * B_b.to_G3()
        return Rotor3(B_out.c, Bivector3(B_out.c_01, B_out.c_12, B_out.c_20))

    @property
    def squared_norm(self) -> float:
        return np.square(jax.tree.leaves(self)).sum()

    @property
    def norm(self) -> float:
        return np.sqrt(self.squared_norm)

    def normalize(self) -> Self:
        norm = self.norm + 1e-8
        return tree_map(lambda x: x / norm, self)

    def to_G3(self):
        return G3(0, 0, 0, 0, self.c_01, self.c_12, self.c_20, 0)

    def inverse(self) -> Self:
        squared_norm = self.squared_norm + 1e-8
        return tree_map(lambda x: x / squared_norm, self)

    def project_vector(self, a):
        B_g3 = self.to_G3()
        a_g3 = G3(0, a[0], a[1], a[2], 0, 0, 0, 0)
        dot_g3 = 0.5 * (a_g3 * B_g3 - B_g3 * a_g3)
        a_proj_g3 = dot_g3 * self.inverse().to_G3()
        return np.array([a_proj_g3.c_0, a_proj_g3.c_1, a_proj_g3.c_2])

    def __add__(self, other) -> Self:
        if isinstance(other, Bivector3):
            return tree_map(lambda x, y: x + y, self, other)
        else:
            return NotImplemented

    def __mul__(self, other) -> Self:
        if isinstance(other, (float, int)):
            return tree_map(lambda x: x * other, self)
        elif isinstance(other, Bivector3):
            return Bivector3.bivector_bivector_product(self, other)
        else:
            return NotImplemented

    def __rmul__(self, other) -> Self:
        if isinstance(other, (float, int)):
            return self * other
        elif isinstance(other, Bivector3):
            return Bivector3.bivector_bivector_product(other, self)
        else:
            return NotImplemented

    def __truediv__(self, other):
        return (1 / other) * self

    def __neg__(self):
        return -1 * self


@register_pytree_node_class
@define
class Rotor3:
    c: float
    B: Bivector3

    def tree_flatten(self):
        B_flatten, _ = self.B.tree_flatten()
        children = (self.c, B_flatten)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        c, B_flatten = children
        return cls(c, Bivector3.tree_unflatten(None, B_flatten))

    @classmethod
    def from_pairs(cls, a: np.ndarray, b: np.ndarray, normalize=True):
        c = np.dot(a, b)
        B = Bivector3.from_wedge(a, b)
        if normalize:
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            c /= norm
            B /= norm

        return cls(c, B)

    @property
    def dag(self) -> Self:
        return self.conjugate()

    @staticmethod
    def rotor_rotor_product(R_a, R_b) -> Self:
        R_out = R_a.to_G3() * R_b.to_G3()
        return Rotor3(R_out.c, Bivector3(R_out.c_01, R_out.c_12, R_out.c_20))

    def to_G3(self):
        return G3(self.c, 0, 0, 0, self.B.c_01, self.B.c_12, self.B.c_20, 0)

    def conjugate(self) -> Self:
        return Rotor3(self.c, -1 * self.B)

    def __add__(self, other) -> Self:
        if isinstance(other, Rotor3):
            return tree_map(lambda x, y: x + y, self, other)
        else:
            return NotImplemented

    def __mul__(self, other) -> Self:
        if isinstance(other, Rotor3):
            return Rotor3.rotor_rotor_product(self, other)
        elif isinstance(other, (float, int)):
            return tree_map(lambda x: x * other, self)
        else:
            return NotImplemented

    def __rmul__(self, other) -> Self:
        if isinstance(other, Rotor3):
            return Rotor3.rotor_rotor_product(other, self)
        elif isinstance(other, (float, int)):
            return tree_map(lambda x: x * other, self)
        else:
            return NotImplemented

    def __truediv__(self, other):
        return (1 / other) * self

    def __neg__(self):
        return -1 * self

    def rotate(self, a):
        R_g3 = G3(R.c, 0, 0, 0, R.B.c_01, R.B.c_12, R.B.c_20, 0)
        a_g3 = G3(0, a[0], a[1], a[2], 0, 0, 0, 0)
        R_dag_g3 = G3(R.dag.c, 0, 0, 0, R.dag.B.c_01, R.dag.B.c_12,
                      R.dag.B.c_20, 0)
        b_g3 = R_g3 * a_g3 * R_dag_g3

        assert np.isclose(b_g3.c_012, 0)

        return np.array([b_g3.c_0, b_g3.c_1, b_g3.c_2])


def angle_between(a, b):
    theta = np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
    return np.rad2deg(theta)


if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.randn(3)
    b = np.random.randn(3)
    v = np.random.randn(3)

    half_theta = angle_between(a, b)

    R = Rotor3.from_pairs(a, b)

    v_rot = R.rotate(v)

    # The rotation occurs at plane a wedge b
    theta = angle_between(R.B.project_vector(v), R.B.project_vector(v_rot))

    assert np.isclose(2 * half_theta, theta)

    ic(v, v_rot)
