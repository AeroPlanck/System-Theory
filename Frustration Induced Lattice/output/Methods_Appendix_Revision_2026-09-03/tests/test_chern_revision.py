"""Regression checks for singular closures, target gaps and UV attachment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import numpy as np
import pytest
from ChernNumberCompute import compute_topology, check_spectral_separation, _riesz_frame
from Dispersion import M_matrix_vectorized
import SpectralFlow as sf


def params(fraction=.5, omega=0., coupling=20.75):
    return (3., omega, coupling/np.pi, fraction*np.pi, 1., 1.)


@pytest.mark.parametrize("fraction", [0., 1.])
def test_endpoint_is_unavailable_not_zero(fraction):
    raw, integer, diag = compute_topology(params(fraction), [0], gap_samples=101)
    assert np.isnan(raw) and integer is None and not diag["valid"]
    assert "singular" in diag["reason"]
    with pytest.raises(ValueError, match="singular"):
        M_matrix_vectorized(1., 0., *params(fraction))


def test_original_small_k_EP_cannot_hide_between_coarse_mesh_rows():
    raw, integer, diag = compute_topology(params(.99), [0], Q=60,
                                         N_theta=11, N_phi=15, gap_samples=501)
    assert np.isnan(raw) and integer is None
    assert diag["status"] == "invalid_target_complement_gap"
    assert np.allclose(diag["discriminant_roots"], [.413266553435, .472216673973], atol=2e-9)


def test_internal_EP_does_not_invalidate_an_isolated_cluster():
    diag = check_spectral_separation(params(.99), [0,2], gap_samples=501)
    assert diag["valid"] and diag["pole_chern"] == 0
    raw, integer, diag = compute_topology(params(.99), [0,2], N_phi=31, gap_samples=501)
    assert diag["valid"] and integer == 0 and abs(raw)<1e-10
    assert not diag["globally_proven"]


@pytest.mark.parametrize("omega,coupling,expected", [(0.,20.75,-2),(30.,20.,2)])
def test_automatic_cap_follows_the_tracked_UV_sector(omega, coupling, expected):
    raw, integer, diag = compute_topology(params(omega=omega,coupling=coupling),
                                         [0], N_phi=31, gap_samples=501)
    assert diag["valid"] and integer == expected
    assert raw == pytest.approx(expected, abs=1e-10)
    assert diag["cap_selection"] == "tracked_asymptotic_spin_sectors"
    assert diag["cap_basis_mismatch"] < 1e-12


def test_wrong_explicit_cap_rejected_but_same_subspace_gauge_accepted():
    p = params(omega=30., coupling=20.)
    wrong = np.array([[0.],[1.],[1j]])
    raw, integer, diag = compute_topology(p, [0], infty_basis=wrong, gap_samples=101)
    assert np.isnan(raw) and integer is None and diag["status"] == "invalid_cap_basis"
    correct = (2+3j)*np.array([[0.],[1.],[-1j]])
    raw, integer, diag = compute_topology(p, [0], infty_basis=correct, N_phi=31, gap_samples=501)
    assert diag["valid"] and integer == 2 and abs(raw-2)<1e-10


def test_schur_projector_is_regular_for_internal_jordan_block():
    matrix = np.array([[1.,1.,.3],[0.,1.,.7],[0.,0.,3.]], complex)
    right, left, projector = _riesz_frame(matrix, np.array([1.,1.]), np.array([3.]))
    assert np.allclose(left.conj().T@right, np.eye(2))
    assert np.allclose(projector@projector, projector)
    assert np.allclose(matrix@projector, projector@matrix)
    assert np.linalg.matrix_rank(projector) == 2


@pytest.mark.parametrize("cap_tol", [0.,-1.,np.nan,np.inf,1.])
def test_bad_cap_tolerance_cannot_start_an_unbounded_refinement(cap_tol):
    with pytest.raises(ValueError, match="cap_tol"):
        compute_topology(params(), [0], cap_tol=cap_tol)


@pytest.mark.parametrize("bands", [[.5],[0,0],[3],[]])
def test_invalid_band_indices_rejected(bands):
    with pytest.raises(ValueError):
        compute_topology(params(), bands)


@pytest.mark.parametrize("kwargs", [{"gap_rtol":np.nan}, {"gap_atol":np.nan},
                                    {"gap_rtol":np.inf}, {"gap_samples":np.inf},
                                    {"gap_samples":101.5}])
def test_invalid_gap_tolerances_cannot_bypass_EP_screening(kwargs):
    with pytest.raises(ValueError):
        check_spectral_separation(params(.99), [0], **kwargs)


def test_full_space_is_zero_without_imposing_internal_gaps():
    raw, integer, diag = compute_topology(params(.99), [0,1,2])
    assert raw == 0 and integer == 0 and diag["valid"]
    assert diag["status"] == "valid_full_space"


def test_independent_bulk_implementations_match_on_physical_sweep():
    for fraction in sf.DEFAULT_ALPHA_OVER_PI:
        p = sf.matched_particle_params(fraction*np.pi)
        for kx, ky in ((0.,0.),(1e-5,-2e-5),(.7,1.3),(35.,-20.)):
            assert np.allclose(M_matrix_vectorized(kx,ky,*p),
                               sf.M_matrix_standalone(kx,ky,p), rtol=1e-11,atol=1e-11)


def test_threshold_reference_gap_lower_bound_on_wide_radial_grid():
    p = sf.DEFAULT_PARAMS
    radii = np.r_[0., np.geomspace(1e-6,1e5,2001)]
    values = np.linalg.eigvals(M_matrix_vectorized(radii,0.,*p))
    positive_frequency = np.max(values.imag, axis=1)
    assert positive_frequency.min() == pytest.approx(20.75/2, abs=1e-10)
    assert np.min(np.abs(values.imag-10.)) > .37499999


def test_scan_exports_invalid_endpoints_as_NA():
    import copy
    import csv
    import ChernParamScan as scan
    tmp_path = Path(__file__).resolve().parents[1] / "scan_smoke_test"
    tmp_path.mkdir(exist_ok=True)
    config = copy.deepcopy(scan.USER_CONFIG)
    config["alpha_scan"] = {"alpha_min": 0., "alpha_max": np.pi, "n_alpha": 3}
    config["param_scan"].update(v=(3.,3.,1), omega=(0.,0.,1), d0=(1.,1.,1),
                                lam=(None,None,1), lam_auto_scale=(1.,1.))
    config["compute"].update(N_theta=11, N_phi=15, gap_samples=101)
    scan.main(config=config, output_dir=tmp_path)
    with (tmp_path/"status_v_3_omega_0.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows)==21
    endpoint_rows = [r for r in rows if float(r["alpha_over_pi"]) in (0.,1.)]
    assert len(endpoint_rows)==14
    assert all(r["C_raw"]=="NA" and r["C_integer"]=="NA" and r["valid"]=="False" for r in endpoint_rows)
    assert all(r["valid"]=="True" for r in rows if float(r["alpha_over_pi"])==.5)
