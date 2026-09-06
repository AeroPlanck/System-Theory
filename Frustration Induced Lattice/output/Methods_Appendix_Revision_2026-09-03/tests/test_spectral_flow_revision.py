"""Small deterministic tests; no particle trajectories or publication files."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import numpy as np
import pytest
import SpectralFlow as sf


def example(left=(0.9, 0.9), right=(0.01, 0.01), values=(-1j, 1j)):
    return sf.FlowData(ky=np.arange(len(values), dtype=float), eigvals=np.array(values)[:, None],
                       left_weight=np.array(left)[:, None], right_weight=np.array(right)[:, None])


def test_particle_density_uses_radius_half_of_L():
    v, omega, lam, alpha, rho0, d0 = sf.DEFAULT_PARAMS
    assert rho0 == pytest.approx(2000/(np.pi*3.5**2))
    assert lam == pytest.approx(0.12709375)
    assert lam*rho0*np.pi*d0**2 == pytest.approx(20.75)
    assert (v, omega, d0) == (3.0, 0.0, 1.0)
    assert sf.DEFAULT_ALPHA_OVER_PI == (0.2, 0.4, 0.5, 0.6, 0.8)


@pytest.mark.parametrize("alpha", [0.0, np.pi])
def test_singular_endpoints_rejected(alpha):
    with pytest.raises(ValueError, match="D0"):
        sf.M_matrix_standalone(0.2, 0.3, sf.matched_particle_params(alpha))


def test_one_sided_edge_does_not_pass_average_threshold():
    result = sf.count_horizontal_crossings(example(left=(0.9, 0.01)), 0, return_diagnostics=True)
    assert result[:2] == (0, 0)
    assert not result[4]["excluded_crossings"][0]["persistent_edge_label"]


def test_persistent_left_and_right_labels():
    assert sf.count_horizontal_crossings(example(), 0)[:2] == (1, 0)
    assert sf.count_horizontal_crossings(example(left=(0.01, 0.01), right=(0.9, 0.9)), 0)[:2] == (0, 1)


def test_exact_vertex_crossing_counted_once():
    data = example(left=(0.9, 0.9, 0.9), right=(0.01, 0.01, 0.01), values=(-1j, 0j, 1j))
    result = sf.count_horizontal_crossings(data, 0)
    assert result[:2] == (1, 0)
    assert len(result[2]) == 1
    assert result[2][0][1] == pytest.approx(1.0)


def test_ambiguous_crossing_is_reported_not_counted():
    data = example()
    data.assignment_ambiguous = np.array([[True]])
    result = sf.count_horizontal_crossings(data, 0, return_diagnostics=True)
    assert result[:2] == (0, 0)
    assert result[4]["excluded_crossings"][0]["ambiguous_assignment"]


def test_complex_crossing_uses_same_interpolation_fraction():
    result = sf.count_horizontal_crossings(example(values=(2-2j, 8+1j)), 0)
    assert result[2][0][2] == pytest.approx(6+0j)


def test_left_right_pairing_and_small_strip_shapes():
    matrix = sf.M_matrix_standalone(0.2, 0.3, sf.DEFAULT_PARAMS)
    vals, right, left = sf.eig_left_right(matrix)
    assert np.allclose(matrix@right, right*vals, atol=1e-11)
    assert np.allclose(matrix.conj().T@left, left*vals.conj(), atol=1e-11)
    assert np.allclose(left.conj().T@right, np.eye(3), atol=1e-11)
    data = sf.compute_strip_data(sf.DEFAULT_PARAMS, ky_max=2, n_ky=5, n_cells=6, kx_cut=4, n_kx=32, hop_cut=5, edge_width=1)
    assert data.eigvals.shape == (5, 18)
    assert data.ambiguous.shape == (5, 18)
    assert data.assignment_ambiguous.shape == (4, 18)
    assert data.params == sf.DEFAULT_PARAMS
    assert "no subspace continuation" in data.diagnostics["tracking_method"]
