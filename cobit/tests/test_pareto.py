import numpy as np

from cobit.pareto import (
    coverage,
    hypervolume_2d,
    net_coverage,
    non_dominated,
    roi_filter,
    spacing,
)


def test_non_dominated_basic():
    pts = np.array([[1, 5], [2, 3], [3, 3], [2, 6], [4, 1]])
    mask = non_dominated(pts)
    assert mask.tolist() == [True, True, False, False, True]


def test_non_dominated_duplicates_kept():
    pts = np.array([[1.0, 2.0], [1.0, 2.0], [0.5, 3.0]])
    mask = non_dominated(pts)
    assert mask.tolist() == [True, True, True]


def test_non_dominated_empty():
    assert non_dominated(np.zeros((0, 2))).size == 0


def test_hypervolume_known_values():
    ref = (4.0, 4.0)
    assert hypervolume_2d(np.array([[2, 2]]), ref) == 0.25
    # a point on/outside the reference box contributes nothing
    assert hypervolume_2d(np.array([[4, 1], [5, 5]]), ref) == 0.0
    # two staircase points
    hv = hypervolume_2d(np.array([[1, 3], [3, 1]]), ref)
    # normalized: (1-.25)*(1-.75) + (1-.75)*(.75-.25) = 0.1875 + 0.125
    assert abs(hv - 0.3125) < 1e-12
    # dominated point must not change the volume
    hv2 = hypervolume_2d(np.array([[1, 3], [3, 1], [3, 3]]), ref)
    assert hv2 == hv


def test_coverage_and_net_coverage():
    A = np.array([[1.0, 1.0]])
    B = np.array([[2.0, 2.0], [0.5, 3.0]])
    assert coverage(A, B) == 0.5
    assert coverage(B, A) == 0.0
    nc = net_coverage([A, B])
    assert nc.tolist() == [0.5, -0.5]


def test_spacing():
    assert spacing(np.array([[0, 0], [1, 1], [2, 2]])) == 0.0  # uniform
    assert spacing(np.array([[0, 0]])) == 0.0  # degenerate
    assert spacing(np.array([[0, 0], [1, 0], [5, 0]])) > 0.0


def test_roi_filter():
    pts = np.array([[4.0, 500], [6.0, 500], [4.0, 1500]])
    assert roi_filter(pts, 5.0, 1000).tolist() == [True, False, False]
