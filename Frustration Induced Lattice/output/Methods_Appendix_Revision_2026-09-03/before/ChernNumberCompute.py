import numpy as np
from itertools import combinations
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from Dispersion import M_matrix_vectorized

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 16,
})


def _eig_lr(matrix):
    values_r, vectors_r = np.linalg.eig(matrix)
    values_l, vectors_l = np.linalg.eig(matrix.conj().T)
    cost = np.abs(values_l.conj()[:, None] - values_r[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)
    left_order = np.zeros_like(col_ind)
    for r, c in zip(row_ind, col_ind):
        left_order[c] = r
    vectors_l = vectors_l[:, left_order]
    return values_r, vectors_r, vectors_l


def _sort_initial(values_r, vectors_r, vectors_l, sort_by):
    if sort_by == "abs":
        order = np.argsort(np.abs(values_r))
    elif sort_by == "real":
        order = np.argsort(values_r.real)
    else:
        order = np.argsort(values_r.imag)
    return values_r[order], vectors_r[:, order], vectors_l[:, order]


def _overlap_match(prev_left, prev_values, new_right, new_left, new_values, overlap_weight, eig_weight):
    overlap = np.abs(prev_left.conj().T @ new_right)
    overlap_scale = np.max(overlap)
    if overlap_scale <= 0.0:
        overlap_scale = 1.0
    overlap_cost = 1.0 - np.clip(overlap / overlap_scale, 0.0, 1.0)
    eig_diff = np.abs(prev_values[:, None] - new_values[None, :])
    eig_scale = np.max(eig_diff)
    if eig_scale <= 0.0:
        eig_scale = 1.0
    eig_cost = eig_diff / eig_scale
    cost = overlap_weight * overlap_cost + eig_weight * eig_cost
    row_ind, col_ind = linear_sum_assignment(cost)
    new_order = np.zeros(new_right.shape[1], dtype=int)
    for r, c in zip(row_ind, col_ind):
        new_order[r] = c
    return new_right[:, new_order], new_left[:, new_order], new_values[new_order]


def _q_from_theta(theta, scale):
    return scale * np.tan(0.5 * theta)


def _point_from_angles(theta, phi, scale):
    q = _q_from_theta(theta, scale)
    return q * np.cos(phi), q * np.sin(phi)


def _is_hole(q1, q2, holes):
    for hx, hy, radius in holes:
        if (q1 - hx) ** 2 + (q2 - hy) ** 2 <= radius ** 2:
            return True
    return False


def _compute_local_flux(
    theta_left,
    theta_right,
    phi_left,
    phi_right,
    n_subdiv,
    params,
    target_bands,
    Q,
    det_tol,
    svd_tol,
    overlap_weight,
    eig_weight,
    holes,
    seed_right,
    seed_left,
    seed_values,
):
    v, omega, lam, alpha, rho0, d0 = params
    m = len(target_bands)
    theta_nodes = np.linspace(theta_left, theta_right, n_subdiv + 1)
    phi_nodes = np.linspace(phi_left, phi_right, n_subdiv + 1)
    R_loc = np.zeros((n_subdiv + 1, n_subdiv + 1, 3, 3), dtype=np.complex128)
    L_loc = np.zeros_like(R_loc)
    E_loc = np.zeros((n_subdiv + 1, n_subdiv + 1, 3), dtype=np.complex128)
    R_loc[0, 0] = seed_right
    L_loc[0, 0] = seed_left
    E_loc[0, 0] = seed_values

    for i in range(n_subdiv + 1):
        for j in range(n_subdiv + 1):
            if i == 0 and j == 0:
                continue
            q1, q2 = _point_from_angles(theta_nodes[i], phi_nodes[j], Q)
            matrix = M_matrix_vectorized(q1, q2, v, omega, lam, alpha, rho0, d0)
            values_r, vectors_r, vectors_l = _eig_lr(matrix)
            if j > 0:
                prev_left = L_loc[i, j - 1]
                prev_values = E_loc[i, j - 1]
            else:
                prev_left = L_loc[i - 1, j]
                prev_values = E_loc[i - 1, j]
            vectors_r, vectors_l, values_r = _overlap_match(
                prev_left,
                prev_values,
                vectors_r,
                vectors_l,
                values_r,
                overlap_weight=overlap_weight,
                eig_weight=eig_weight,
            )
            R_loc[i, j] = vectors_r
            L_loc[i, j] = vectors_l
            E_loc[i, j] = values_r

    R_sub = np.zeros((n_subdiv + 1, n_subdiv + 1, 3, m), dtype=np.complex128)
    L_sub = np.zeros_like(R_sub)
    min_sigma = np.inf
    for i in range(n_subdiv + 1):
        for j in range(n_subdiv + 1):
            R = R_loc[i, j][:, target_bands]
            L_tilde = L_loc[i, j][:, target_bands]
            S = L_tilde.conj().T @ R
            svals = np.linalg.svd(S, compute_uv=False)
            sigma_min = np.min(svals)
            min_sigma = min(min_sigma, sigma_min)
            if sigma_min < svd_tol:
                S_inv = np.linalg.pinv(S)
            else:
                S_inv = np.linalg.inv(S)
            L = L_tilde @ S_inv.conj().T
            R_sub[i, j] = R
            L_sub[i, j] = L

    U_theta = np.ones((n_subdiv, n_subdiv + 1), dtype=np.complex128)
    U_phi = np.ones((n_subdiv + 1, n_subdiv), dtype=np.complex128)
    bad_theta = np.zeros((n_subdiv, n_subdiv + 1), dtype=bool)
    bad_phi = np.zeros((n_subdiv + 1, n_subdiv), dtype=bool)
    min_det = np.inf

    for i in range(n_subdiv):
        for j in range(n_subdiv + 1):
            W = L_sub[i, j].conj().T @ R_sub[i + 1, j]
            detW = np.linalg.det(W)
            abs_det = np.abs(detW)
            min_det = min(min_det, abs_det)
            if abs_det < det_tol:
                U_theta[i, j] = 1.0 + 0.0j
                bad_theta[i, j] = True
            else:
                U_theta[i, j] = detW / abs_det

    for i in range(n_subdiv + 1):
        for j in range(n_subdiv):
            W = L_sub[i, j].conj().T @ R_sub[i, j + 1]
            detW = np.linalg.det(W)
            abs_det = np.abs(detW)
            min_det = min(min_det, abs_det)
            if abs_det < det_tol:
                U_phi[i, j] = 1.0 + 0.0j
                bad_phi[i, j] = True
            else:
                U_phi[i, j] = detW / abs_det

    flux_sum = 0.0
    bad_plaquette_count = 0
    for i in range(n_subdiv):
        theta_mid = 0.5 * (theta_nodes[i] + theta_nodes[i + 1])
        for j in range(n_subdiv):
            phi_mid = 0.5 * (phi_nodes[j] + phi_nodes[j + 1])
            q1_mid, q2_mid = _point_from_angles(theta_mid, phi_mid, Q)
            if _is_hole(q1_mid, q2_mid, holes):
                continue
            if bad_theta[i, j] or bad_theta[i, j + 1] or bad_phi[i, j] or bad_phi[i + 1, j]:
                bad_plaquette_count += 1
                continue
            plaquette = (
                U_theta[i, j]
                * U_phi[i + 1, j]
                * np.conj(U_theta[i, j + 1])
                * np.conj(U_phi[i, j])
            )
            flux_sum += np.angle(plaquette)
    return flux_sum, bad_plaquette_count, min_det, min_sigma


def _refine_bad_plaquette(
    theta_left,
    theta_right,
    phi_left,
    phi_right,
    params,
    target_bands,
    Q,
    det_tol,
    svd_tol,
    overlap_weight,
    eig_weight,
    holes,
    seed_right,
    seed_left,
    seed_values,
    refine_max_level,
):
    for level in range(1, refine_max_level + 1):
        n_subdiv = 2**level
        flux_sum, bad_count, _, _ = _compute_local_flux(
            theta_left,
            theta_right,
            phi_left,
            phi_right,
            n_subdiv,
            params,
            target_bands,
            Q,
            det_tol,
            svd_tol,
            overlap_weight,
            eig_weight,
            holes,
            seed_right,
            seed_left,
            seed_values,
        )
        if bad_count == 0:
            return True, flux_sum, level
    return False, 0.0, refine_max_level if refine_max_level > 0 else 0


def compute_topology(
    params,
    target_bands,
    Q=10.0,
    N_theta=41,
    N_phi=61,
    delta=1e-3,
    holes=None,
    infty_basis=None,
    sort_by="imag",
    det_tol=1e-7,
    svd_tol=1e-6,
    overlap_weight=0.4,
    eig_weight=0.6,
    phase_branch_tol=0.1,
    phase_jump_tol=3.0,
    refine_bad=True,
    refine_max_level=3,
):
    v, omega, lam, alpha, rho0, d0 = params
    holes = holes or []
    target_bands = np.array(target_bands, dtype=int)
    m = len(target_bands)

    theta = np.linspace(0.0, np.pi - delta, N_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, N_phi, endpoint=False)

    R_all = np.zeros((N_theta, N_phi, 3, 3), dtype=np.complex128)
    L_all = np.zeros_like(R_all)
    E_all = np.zeros((N_theta, N_phi, 3), dtype=np.complex128)

    q1, q2 = _point_from_angles(theta[0], phi[0], Q)
    matrix0 = M_matrix_vectorized(q1, q2, v, omega, lam, alpha, rho0, d0)
    values_r, vectors_r, vectors_l = _eig_lr(matrix0)
    values_r, vectors_r, vectors_l = _sort_initial(values_r, vectors_r, vectors_l, sort_by)
    for j in range(N_phi):
        R_all[0, j] = vectors_r
        L_all[0, j] = vectors_l
        E_all[0, j] = values_r

    for i in range(1, N_theta):
        for j in range(N_phi):
            q1, q2 = _point_from_angles(theta[i], phi[j], Q)
            matrix = M_matrix_vectorized(q1, q2, v, omega, lam, alpha, rho0, d0)
            values_r, vectors_r, vectors_l = _eig_lr(matrix)
            if j == 0:
                prev_left = L_all[i - 1, 0]
                prev_values = E_all[i - 1, 0]
            else:
                prev_left = L_all[i, j - 1]
                prev_values = E_all[i, j - 1]
            vectors_r, vectors_l, values_r = _overlap_match(
                prev_left,
                prev_values,
                vectors_r,
                vectors_l,
                values_r,
                overlap_weight=overlap_weight,
                eig_weight=eig_weight,
            )
            R_all[i, j] = vectors_r
            L_all[i, j] = vectors_l
            E_all[i, j] = values_r

    R_sub = np.zeros((N_theta, N_phi, 3, m), dtype=np.complex128)
    L_sub = np.zeros_like(R_sub)
    min_sigma = np.inf

    for i in range(N_theta):
        for j in range(N_phi):
            R = R_all[i, j][:, target_bands]
            L_tilde = L_all[i, j][:, target_bands]
            S = L_tilde.conj().T @ R
            svals = np.linalg.svd(S, compute_uv=False)
            sigma_min = np.min(svals)
            min_sigma = min(min_sigma, sigma_min)
            if sigma_min < svd_tol:
                S_inv = np.linalg.pinv(S)
            else:
                S_inv = np.linalg.inv(S)
            L = L_tilde @ S_inv.conj().T
            R_sub[i, j] = R
            L_sub[i, j] = L

    if infty_basis is None:
        tb_tuple = tuple(target_bands)
        if tb_tuple == (0,):
            R_inf = np.array([[0], [1], [1j]], dtype=np.complex128)
        elif tb_tuple == (1,):
            R_inf = np.array([[1], [0], [0]], dtype=np.complex128)
        elif tb_tuple == (2,):
            R_inf = np.array([[0], [1], [-1j]], dtype=np.complex128)
        elif tb_tuple == (0, 1):
            R_inf = np.array([[1, 0], [0, 1], [0, 1j]], dtype=np.complex128)
        elif tb_tuple == (0, 2):
            R_inf = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.complex128)
        elif tb_tuple == (1, 2):
            R_inf = np.array([[1, 0], [0, 1], [0, -1j]], dtype=np.complex128)
        elif tb_tuple == (0, 1, 2):
            R_inf = np.eye(3, dtype=np.complex128)
        else:
            R_inf = np.eye(3, dtype=np.complex128)[:, target_bands]
    else:
        R_inf = np.array(infty_basis, dtype=np.complex128)
        if R_inf.shape != (3, m):
            raise ValueError("infty_basis must have shape (3, m)")
    S_inf = R_inf.conj().T @ R_inf
    L_inf = R_inf @ np.linalg.inv(S_inf).conj().T
    for j in range(N_phi):
        R_sub[-1, j] = R_inf
        L_sub[-1, j] = L_inf

    U_theta = np.ones((N_theta - 1, N_phi), dtype=np.complex128)
    U_phi = np.ones((N_theta, N_phi), dtype=np.complex128)
    bad_theta = np.zeros((N_theta - 1, N_phi), dtype=bool)
    bad_phi = np.zeros((N_theta, N_phi), dtype=bool)
    min_det = np.inf
    bad_edge_count = 0

    for i in range(N_theta - 1):
        for j in range(N_phi):
            W = L_sub[i, j].conj().T @ R_sub[i + 1, j]
            detW = np.linalg.det(W)
            abs_det = np.abs(detW)
            min_det = min(min_det, abs_det)
            if abs_det < det_tol:
                U_theta[i, j] = 1.0 + 0.0j
                bad_theta[i, j] = True
                bad_edge_count += 1
            else:
                U_theta[i, j] = detW / abs_det

    for i in range(N_theta):
        for j in range(N_phi):
            jp = (j + 1) % N_phi
            W = L_sub[i, j].conj().T @ R_sub[i, jp]
            detW = np.linalg.det(W)
            abs_det = np.abs(detW)
            min_det = min(min_det, abs_det)
            if abs_det < det_tol:
                U_phi[i, j] = 1.0 + 0.0j
                bad_phi[i, j] = True
                bad_edge_count += 1
            else:
                U_phi[i, j] = detW / abs_det

    flux_sum = 0.0
    bad_plaquette_count = 0
    refined_plaquette_count = 0
    unresolved_bad_plaquette_count = 0
    max_refine_level_used = 0
    plaquette_phase = np.full((N_theta - 1, N_phi), np.nan, dtype=float)
    valid_plaquette = np.zeros((N_theta - 1, N_phi), dtype=bool)
    for i in range(N_theta - 1):
        theta_mid = 0.5 * (theta[i] + theta[i + 1])
        for j in range(N_phi):
            phi_mid = phi[j] + 0.5 * (phi[(j + 1) % N_phi] - phi[j])
            q1_mid, q2_mid = _point_from_angles(theta_mid, phi_mid, Q)
            if _is_hole(q1_mid, q2_mid, holes):
                continue
            jp = (j + 1) % N_phi
            if bad_theta[i, j] or bad_theta[i, jp] or bad_phi[i, j] or bad_phi[i + 1, j]:
                bad_plaquette_count += 1
                if refine_bad and refine_max_level > 0:
                    phi_left = phi[j]
                    if jp == 0:
                        phi_right = phi[0] + 2.0 * np.pi
                    else:
                        phi_right = phi[jp]
                    refined_ok, refined_flux, level_used = _refine_bad_plaquette(
                        theta[i],
                        theta[i + 1],
                        phi_left,
                        phi_right,
                        params,
                        target_bands,
                        Q,
                        det_tol,
                        svd_tol,
                        overlap_weight,
                        eig_weight,
                        holes,
                        R_all[i, j],
                        L_all[i, j],
                        E_all[i, j],
                        refine_max_level,
                    )
                    if refined_ok:
                        flux_sum += refined_flux
                        refined_plaquette_count += 1
                        max_refine_level_used = max(max_refine_level_used, level_used)
                        continue
                unresolved_bad_plaquette_count += 1
                continue
            plaquette = (
                U_theta[i, j]
                * U_phi[i + 1, j]
                * np.conj(U_theta[i, jp])
                * np.conj(U_phi[i, j])
            )
            phase_val = np.angle(plaquette)
            plaquette_phase[i, j] = phase_val
            valid_plaquette[i, j] = True
            flux_sum += phase_val

    near_branch_mask = np.abs(np.abs(plaquette_phase) - np.pi) <= phase_branch_tol
    near_branch_count = int(np.sum(valid_plaquette & near_branch_mask))
    flip_mask = np.zeros((N_theta - 1, N_phi), dtype=bool)
    flip_pair_count = 0
    for i in range(N_theta - 1):
        for j in range(N_phi):
            if not valid_plaquette[i, j]:
                continue
            for ni, nj in ((i + 1, j), (i, (j + 1) % N_phi)):
                if ni >= N_theta - 1:
                    continue
                if not valid_plaquette[ni, nj]:
                    continue
                p1 = plaquette_phase[i, j]
                p2 = plaquette_phase[ni, nj]
                if (np.abs(np.abs(p1) - np.pi) <= phase_branch_tol
                        and np.abs(np.abs(p2) - np.pi) <= phase_branch_tol
                        and p1 * p2 < 0.0
                        and np.abs(p1 - p2) >= phase_jump_tol):
                    flip_pair_count += 1
                    flip_mask[i, j] = True
                    flip_mask[ni, nj] = True
    flip_plaquette_count = int(np.sum(flip_mask))
    flip_samples = []
    idx_i, idx_j = np.where(flip_mask)
    for i, j in zip(idx_i[:20], idx_j[:20]):
        theta_mid = 0.5 * (theta[i] + theta[i + 1])
        phi_mid = phi[j] + 0.5 * (phi[(j + 1) % N_phi] - phi[j])
        flip_samples.append((int(i), int(j), float(theta_mid / np.pi), float(phi_mid / np.pi)))

    C_val = flux_sum / (2.0 * np.pi)
    C_int = int(np.rint(C_val))
    diagnostics = {
        "min_sigma": float(min_sigma),
        "min_det": float(min_det),
        "bad_edge_count": int(bad_edge_count),
        "bad_plaquette_count": int(bad_plaquette_count),
        "refined_plaquette_count": int(refined_plaquette_count),
        "unresolved_bad_plaquette_count": int(unresolved_bad_plaquette_count),
        "max_refine_level_used": int(max_refine_level_used),
        "near_branch_count": int(near_branch_count),
        "phase_flip_pair_count": int(flip_pair_count),
        "phase_flip_plaquette_count": int(flip_plaquette_count),
        "phase_flip_samples": flip_samples,
        "det_tol": float(det_tol),
        "svd_tol": float(svd_tol),
        "phase_branch_tol": float(phase_branch_tol),
        "phase_jump_tol": float(phase_jump_tol),
        "refine_bad": bool(refine_bad),
        "refine_max_level": int(refine_max_level),
    }
    return C_val, C_int, diagnostics


if __name__ == "__main__":
    v = 3.0
    omega = 0
    rho0 = 0.0204
    d0 = 1
    lam = 20 / (rho0 * np.pi * d0**2)
    holes = [(0.0, 0.0, 0.00)]
    bands = [0, 1, 2]
    alpha_min = 0.00001 * np.pi
    alpha_max = 0.999 * np.pi
    N_alpha = 50
    alphas = np.linspace(alpha_min, alpha_max, N_alpha)
    alpha_focus = 1.0 * np.pi
    focus_bands = [0, 1, 2]
    combos = []
    for r in range(1, len(bands) + 1):
        for combo in combinations(bands, r):
            combos.append(list(combo))

    results = {tuple(combo): [] for combo in combos}
    bad_counts = {tuple(combo): [] for combo in combos}
    min_det_series = {tuple(combo): [] for combo in combos}
    min_sigma_series = {tuple(combo): [] for combo in combos}
    phase_flip_series = {tuple(combo): [] for combo in combos}
    unresolved_bad_series = {tuple(combo): [] for combo in combos}
    full_combo = tuple(bands)
    for alpha in alphas:
        params = (v, omega, lam, alpha, rho0, d0)
        for combo in combos:
            C_val, C_int, diagnostics = compute_topology(
                params,
                target_bands=combo,
                Q=60.0,
                N_theta=71,
                N_phi=91,
                delta=1e-3,
                holes=holes,
            )
            results[tuple(combo)].append(C_val)
            bad_counts[tuple(combo)].append(diagnostics["bad_plaquette_count"])
            min_det_series[tuple(combo)].append(diagnostics["min_det"])
            min_sigma_series[tuple(combo)].append(diagnostics["min_sigma"])
            phase_flip_series[tuple(combo)].append(diagnostics["phase_flip_plaquette_count"])
            unresolved_bad_series[tuple(combo)].append(diagnostics["unresolved_bad_plaquette_count"])
            print("alpha:", alpha, "bands:", combo, "C_val:", C_val, "C_int:", C_int, "diagnostics:", diagnostics)

    params_focus = (v, omega, lam, alpha_focus, rho0, d0)
    _, _, diagnostics_focus = compute_topology(
        params_focus,
        target_bands=focus_bands,
        Q=20.0,
        N_theta=51,
        N_phi=71,
        delta=1e-3,
        holes=holes,
        det_tol=1e-7,
        svd_tol=1e-6,
    )
    print("alpha_focus:", alpha_focus, "bands:", focus_bands, "bad_plaquettes:", diagnostics_focus["bad_plaquette_count"])

    x = alphas / np.pi
    single_combos = [c for c in combos if len(c) == 1]
    double_combos = [c for c in combos if len(c) == 2]
    triple_combos = [c for c in combos if len(c) == 3]
    mixed_combos = [c for c in combos if len(c) >= 2]
    single_sum = np.zeros_like(alphas, dtype=float)
    for combo in single_combos:
        single_sum += np.array(results[tuple(combo)])
    residual = single_sum - np.array(results[full_combo])

    fig1, ax1 = plt.subplots(1, 1, figsize=(9, 6))
    bad_label_used = False
    for combo in single_combos:
        label = "bands=" + ",".join(str(b) for b in combo)
        ax1.plot(x, results[tuple(combo)], marker="o", lw=1.4, ms=3.5, label=label)
        bad_idx = [i for i, c in enumerate(bad_counts[tuple(combo)]) if c > 0]
        if bad_idx:
            label_bad = "bad plaquette" if not bad_label_used else None
            ax1.scatter(x[bad_idx], np.array(results[tuple(combo)])[bad_idx], color="red", marker="x", s=30, label=label_bad)
            bad_label_used = True
    ax1.set_xlabel(r"$\alpha / \pi$")
    ax1.set_ylabel("Chern number")
    ax1.set_title("Single-band Chern vs alpha")
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=1, fontsize=9)
    plt.tight_layout()
    output_path_1 = "d:\\PrivatePythonProject\\Math\\chern_vs_alpha_single.png"
    plt.savefig(output_path_1, dpi=150)

    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 6))
    bad_label_used = False
    for combo in double_combos:
        label = "bands=" + ",".join(str(b) for b in combo)
        ax2.plot(x, results[tuple(combo)], marker="o", lw=1.4, ms=3.5, label=label)
        bad_idx = [i for i, c in enumerate(bad_counts[tuple(combo)]) if c > 0]
        if bad_idx:
            label_bad = "bad plaquette" if not bad_label_used else None
            ax2.scatter(x[bad_idx], np.array(results[tuple(combo)])[bad_idx], color="red", marker="x", s=30, label=label_bad)
            bad_label_used = True
    ax2.set_xlabel(r"$\alpha / \pi$")
    ax2.set_ylabel("Chern number")
    ax2.set_title("Two-band Chern vs alpha")
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=1, fontsize=9)
    plt.tight_layout()
    output_path_2 = "d:\\PrivatePythonProject\\Math\\chern_vs_alpha_double.png"
    plt.savefig(output_path_2, dpi=150)

    fig3, ax3 = plt.subplots(1, 1, figsize=(9, 6))
    bad_label_used = False
    for combo in triple_combos:
        label = "bands=" + ",".join(str(b) for b in combo)
        ax3.plot(x, results[tuple(combo)], marker="o", lw=1.4, ms=3.5, label=label)
        bad_idx = [i for i, c in enumerate(bad_counts[tuple(combo)]) if c > 0]
        if bad_idx:
            label_bad = "bad plaquette" if not bad_label_used else None
            ax3.scatter(x[bad_idx], np.array(results[tuple(combo)])[bad_idx], color="red", marker="x", s=30, label=label_bad)
            bad_label_used = True
    ax3.set_xlabel(r"$\alpha / \pi$")
    ax3.set_ylabel("Chern number")
    ax3.set_title("Three-band Chern vs alpha")
    ax3.grid(True, alpha=0.3)
    ax3.legend(ncol=1, fontsize=9)
    plt.tight_layout()
    output_path_3 = "d:\\PrivatePythonProject\\Math\\chern_vs_alpha_triple.png"
    plt.savefig(output_path_3, dpi=150)

    fig4, axes = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
    axes[0].plot(x, residual, color="black", marker="o", lw=1.4, ms=3.5)
    axes[0].axhline(0.0, color="gray", lw=1.0, linestyle="--")
    axes[0].set_ylabel(r"$\Sigma C_{\mathrm{single}}-C_{\mathrm{all}}$")
    axes[0].set_title("Residual and diagnostics vs alpha")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(x, np.array(min_det_series[full_combo]), color="tab:blue", marker="o", lw=1.4, ms=3.5)
    axes[1].set_ylabel("min_det (all)")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(x, np.array(min_sigma_series[full_combo]), color="tab:green", marker="o", lw=1.4, ms=3.5)
    axes[2].set_ylabel("min_sigma (all)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(x, np.array(bad_counts[full_combo]), color="tab:red", marker="o", lw=1.4, ms=3.5)
    axes[3].set_ylabel("bad_plaquette (all)")
    axes[3].grid(True, alpha=0.3)
    axes[4].plot(x, np.array(unresolved_bad_series[full_combo]), color="tab:orange", marker="o", lw=1.4, ms=3.5)
    axes[4].set_ylabel("unresolved_bad (all)")
    axes[4].grid(True, alpha=0.3)
    axes[5].plot(x, np.array(phase_flip_series[full_combo]), color="tab:purple", marker="o", lw=1.4, ms=3.5)
    axes[5].set_ylabel("phase_flip (all)")
    axes[5].set_xlabel(r"$\alpha / \pi$")
    axes[5].grid(True, alpha=0.3)
    plt.tight_layout()
    output_path_4 = "d:\\PrivatePythonProject\\Math\\chern_diagnostics_linked.png"
    plt.savefig(output_path_4, dpi=150)

    fig5, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
    for combo in single_combos:
        label = "bands=" + ",".join(str(b) for b in combo)
        axes[0].semilogy(x, np.array(min_det_series[tuple(combo)]), marker="o", lw=1.2, ms=3.0, label=label)
        axes[1].semilogy(x, np.array(min_sigma_series[tuple(combo)]), marker="o", lw=1.2, ms=3.0, label=label)
        axes[2].plot(x, np.array(bad_counts[tuple(combo)]), marker="o", lw=1.2, ms=3.0, label=label)
        axes[3].plot(x, np.array(unresolved_bad_series[tuple(combo)]), marker="o", lw=1.2, ms=3.0, label=label)
        axes[4].plot(x, np.array(phase_flip_series[tuple(combo)]), marker="o", lw=1.2, ms=3.0, label=label)
    axes[0].set_ylabel("min_det")
    axes[0].set_title("Single-band diagnostics vs alpha")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=1, fontsize=9)
    axes[1].set_ylabel("min_sigma")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=1, fontsize=9)
    axes[2].set_ylabel("bad_plaquette")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=1, fontsize=9)
    axes[3].set_ylabel("unresolved_bad")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(ncol=1, fontsize=9)
    axes[4].set_ylabel("phase_flip")
    axes[4].set_xlabel(r"$\alpha / \pi$")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(ncol=1, fontsize=9)
    plt.tight_layout()
    output_path_5 = "d:\\PrivatePythonProject\\Math\\chern_diagnostics_single.png"
    plt.savefig(output_path_5, dpi=150)

    fig6, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
    for combo in mixed_combos:
        label = "bands=" + ",".join(str(b) for b in combo)
        axes[0].semilogy(x, np.array(min_det_series[tuple(combo)]), marker="o", lw=1.2, ms=3.0, label=label)
        axes[1].semilogy(x, np.array(min_sigma_series[tuple(combo)]), marker="o", lw=1.2, ms=3.0, label=label)
        axes[2].plot(x, np.array(bad_counts[tuple(combo)]), marker="o", lw=1.2, ms=3.0, label=label)
        axes[3].plot(x, np.array(unresolved_bad_series[tuple(combo)]), marker="o", lw=1.2, ms=3.0, label=label)
        axes[4].plot(x, np.array(phase_flip_series[tuple(combo)]), marker="o", lw=1.2, ms=3.0, label=label)
    axes[0].set_ylabel("min_det")
    axes[0].set_title("Mixed-band diagnostics vs alpha")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2, fontsize=9)
    axes[1].set_ylabel("min_sigma")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=2, fontsize=9)
    axes[2].set_ylabel("bad_plaquette")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=2, fontsize=9)
    axes[3].set_ylabel("unresolved_bad")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(ncol=2, fontsize=9)
    axes[4].set_ylabel("phase_flip")
    axes[4].set_xlabel(r"$\alpha / \pi$")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(ncol=2, fontsize=9)
    plt.tight_layout()
    output_path_6 = "d:\\PrivatePythonProject\\Math\\chern_diagnostics_mixed.png"
    plt.savefig(output_path_6, dpi=150)
    plt.show()
