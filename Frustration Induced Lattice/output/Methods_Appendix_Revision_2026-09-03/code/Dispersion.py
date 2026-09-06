import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.special import j1

def closure_denominator(v, omega, lam, alpha, rho0, d0, rtol=1e-12):
    """Return D0, rejecting singular or numerically unresolved closures.

    In particular, floating-point sin(pi) must not turn an excluded endpoint
    into a huge but apparently finite matrix. This tolerance is numerical,
    not an assertion that every nearby nonzero D0 is mathematically zero.
    """
    parameters = np.asarray([v, omega, lam, alpha, rho0, d0], dtype=float)
    if not np.all(np.isfinite(parameters)) or d0 <= 0:
        raise ValueError("invalid_parameters: finite scalars and d0 > 0 required")
    coupling = lam * rho0 * np.pi * d0**2
    denominator = 2.0 * (omega - coupling * np.sin(alpha))
    scale = max(2.0 * abs(omega), 2.0 * abs(coupling), np.finfo(float).tiny)
    if abs(denominator) <= rtol * scale:
        raise ValueError("singular_or_unresolved_closure: D0 is zero within relative tolerance")
    return denominator


def radial_coefficients(k, params):
    """The same a(k), b(k), D0 as M, including a stable disk-kernel limit."""
    v, omega, lam, alpha, rho0, d0 = params
    denominator = closure_denominator(*params)
    x = np.asarray(k, dtype=float) * d0
    with np.errstate(divide="ignore", invalid="ignore"):
        disk = 2.0 * j1(x) / x
    disk = np.where(np.abs(x) < 1e-4,
                    1.0 - x*x/8.0 + x**4/192.0 - x**6/9216.0, disk)
    coupling = lam * rho0 * np.pi * d0**2
    a = 0.5 * coupling * disk * np.cos(alpha)
    b = (-omega + coupling * np.sin(alpha)
         - 0.5 * coupling * disk * np.sin(alpha)
         + v*v * np.asarray(k)**2 / (4.0 * denominator))
    return a, b, denominator


def M_matrix_vectorized(kx, ky, v, omega, lam, alpha, rho0, d0):
    """
    Vectorized Matrix M(q).
    kx, ky can be 2D meshgrids.
    Returns M of shape (Ny, Nx, 3, 3) suitable for np.linalg.eigvals.
    """
    q1, q2 = np.broadcast_arrays(np.asarray(kx), np.asarray(ky))
    q_sq = q1*q1 + q2*q2
    q = np.sqrt(q_sq)

    # G_hat(0) = pi * R^2
    G0 = np.pi * d0**2

    # G_hat(q) = 2*pi*(R/q)*J1(qR)
    # Handle q -> 0 limit using np.where to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        G_q = 2 * np.pi * (d0 / q) * j1(q * d0)
    
    # Fix q=0 entries (where q is close to 0)
    G_q = np.where(q < 1e-9, G0, G_q)

    # Fail explicitly instead of dividing by floating-point sin(pi).
    D0_val = closure_denominator(v, omega, lam, alpha, rho0, d0)

    # Prepare components
    zeros = np.zeros_like(q1, dtype=np.complex128)
    
    m00 = zeros
    # m01 = -v * q1
    # m02 = -v * q2
    m01 = -1j * v * q1
    m02 = -1j * v * q2
    
    # m10 = -(v / 2.0) * q1
    # m20 = -(v / 2.0) * q2
    m10 = -1j * (v / 2.0) * q1
    m20 = -1j * (v / 2.0) * q2
    
    diag, term12, _ = radial_coefficients(q, (v, omega, lam, alpha, rho0, d0))
    m11 = diag
    m22 = diag
    
    m12 = term12
    m21 = -term12

    # Construct matrix with shape (3, 3, ...)
    M_stack = np.array([
        [m00, m01, m02],
        [m10, m11, m12],
        [m20, m21, m22]
    ])
    
    # Move the (3, 3) axes to the end: (..., 3, 3)
    if M_stack.ndim > 2:
        return np.moveaxis(M_stack, [0, 1], [-2, -1])
    else:
        return M_stack

def eigs_at_k(kx, ky, params):
    # Wrapper that works for both scalar and vector
    # But for backward compatibility with existing scalar calls in 1d plot,
    # we can just use the vectorized version generally if input is scalar.
    v, omega, lam, alpha, rho0, d0 = params
    M = M_matrix_vectorized(kx, ky, v, omega, lam, alpha, rho0, d0)
    return np.linalg.eigvals(M)

def sort_eigs_continuous(sig_array):
    """
    Sort eigenvalues at each k to make 3 branches visually continuous.
    Greedy matching by minimal distance to previous step.
    sig_array: shape (Nk, 3) complex, unsorted.
    Returns sorted array same shape.
    """
    Nk = sig_array.shape[0]
    out = np.zeros_like(sig_array)
    out[0] = sig_array[0]

    for i in range(1, Nk):
        prev = out[i-1]
        cur = list(sig_array[i])

        # build distance matrix
        dist = np.abs(prev.reshape(3,1) - np.array(cur).reshape(1,3))

        # greedy matching
        used = set()
        for j in range(3):
            kmin = None
            dmin = None
            for k in range(3):
                if k in used:
                    continue
                d = dist[j, k]
                if (dmin is None) or (d < dmin):
                    dmin = d
                    kmin = k
            out[i, j] = cur[kmin]
            used.add(kmin)
    return out

def reorder_branches_across_alpha(sig_all):
    Nalpha, Nk, _ = sig_all.shape
    out = np.zeros_like(sig_all)
    out[0] = sig_all[0]
    for i in range(1, Nalpha):
        prev = out[i-1]
        cur = sig_all[i]
        reordered = np.zeros_like(cur)
        for k in range(Nk):
            p = prev[k]
            c = list(cur[k])
            dist = np.abs(p.reshape(3, 1) - np.array(c).reshape(1, 3))
            used = set()
            for j in range(3):
                kmin = None
                dmin = None
                for m in range(3):
                    if m in used:
                        continue
                    d = dist[j, m]
                    if (dmin is None) or (d < dmin):
                        dmin = d
                        kmin = m
                reordered[k, j] = c[kmin]
                used.add(kmin)
        out[i] = reordered
    return out

def plot_1d_dispersion(params, kmax=10.0, Nk=400, direction=(1.0, 0.0)):
    """
    Plot Re/Im sigma_j(k) along k*(direction).
    """
    ux, uy = direction
    nrm = (ux*ux + uy*uy)**0.5
    ux, uy = ux/nrm, uy/nrm

    ks = np.linspace(-kmax, kmax, Nk)
    sig = np.zeros((Nk, 3), dtype=np.complex128)

    for i, k in enumerate(ks):
        kx, ky = k*ux, k*uy
        sig[i] = eigs_at_k(kx, ky, params)

    sig = sort_eigs_continuous(sig)

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    for j in range(3):
        axes[0].plot(ks, np.real(sig[:, j]), label=fr'branch {j+1}')
    axes[0].axhline(0.0, color='k', lw=0.8)
    axes[0].set_ylabel(r'$\mathrm{Re}\,\sigma(k)$')
    axes[0].legend()

    for j in range(3):
        axes[1].plot(ks, np.imag(sig[:, j]), label=fr'branch {j+1}')
    axes[1].axhline(0.0, color='k', lw=0.8)
    axes[1].set_xlabel(r'$k$')
    axes[1].set_ylabel(r'$\mathrm{Im}\,\sigma(k)$')

    plt.tight_layout()
    plt.show()

def plot_eigenvector_count_vs_q(params, kmax=10.0, Nk=400, direction=(1.0, 0.0), tol=1e-8):
    """
    Plot the number of linearly independent eigenvectors versus q.
    """
    ux, uy = direction
    nrm = (ux*ux + uy*uy)**0.5
    ux, uy = ux/nrm, uy/nrm

    ks = np.linspace(-kmax, kmax, Nk)
    counts = np.zeros(Nk, dtype=int)

    v, omega, lam, alpha, rho0, d0 = params
    for i, k in enumerate(ks):
        kx, ky = k*ux, k*uy
        M = M_matrix_vectorized(kx, ky, v, omega, lam, alpha, rho0, d0)
        _, evecs = np.linalg.eig(M)
        counts[i] = np.linalg.matrix_rank(evecs, tol=tol)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(ks, counts, '-', lw=1.8)
    ax.set_xlabel(r'$q$')
    ax.set_ylabel('Number of independent eigenvectors')
    ax.set_title('Eigenvector Count vs q')
    ax.set_yticks(np.arange(0, 4, 1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_2d_dispersion(params, kmax=6.0, Nk=201):
    """
    Plot Re and Im sigma for all 3 branches on 2D grids.
    Sorted by Real part descending.
    Vectorized implementation.
    """
    v, omega, lam, alpha, rho0, d0 = params

    kxs = np.linspace(-kmax, kmax, Nk)
    kys = np.linspace(-kmax, kmax, Nk)
    
    # Create meshgrid
    KX, KY = np.meshgrid(kxs, kys)
    
    # Calculate eigenvalues vectorized
    # sig shape: (Nk, Nk, 3)
    sig = eigs_at_k(KX, KY, params)
    
    # Sort by Real part descending
    re_sig = np.real(sig)
    sort_idx = np.argsort(-re_sig, axis=-1)
    
    # Reorder sig based on sort_idx
    sig_sorted = np.take_along_axis(sig, sort_idx, axis=-1)

    fig, axes = plt.subplots(3, 2, figsize=(11, 12), sharex=True, sharey=True)
    
    extent = [kxs[0], kxs[-1], kys[0], kys[-1]]
    
    for i in range(3):
        # Real part
        ax_re = axes[i, 0]
        im_re = ax_re.imshow(np.real(sig_sorted[:, :, i]), origin='lower',
                             extent=extent, aspect='equal', cmap='viridis')
        cb_re = plt.colorbar(im_re, ax=ax_re)
        cb_re.set_label(fr'$\mathrm{{Re}}\,\sigma_{{{i+1}}}$')
        ax_re.set_ylabel(r'$k_y$')
        
        # Add titles only to the top row
        if i == 0:
            ax_re.set_title('Growth Rate (Real)')
        
        # Imaginary part
        ax_im = axes[i, 1]
        im_im = ax_im.imshow(np.abs(np.imag(sig_sorted[:, :, i])), origin='lower',
                             extent=extent, aspect='equal', cmap='coolwarm')
        cb_im = plt.colorbar(im_im, ax=ax_im)
        cb_im.set_label(fr'$|\mathrm{{Im}}\,\sigma_{{{i+1}}}|$')
        
        if i == 0:
            ax_im.set_title('Frequency (Abs Imag)')

    # Set x labels only for the bottom row
    axes[2, 0].set_xlabel(r'$k_x$')
    axes[2, 1].set_xlabel(r'$k_x$')

    plt.tight_layout()
    plt.show()

def plot_3d_dispersion(params, kmax=10.0, Nk=400, direction=(1.0, 0.0)):
    """
    Plot 3D trajectory of eigenvalues: (k, Re[sigma], Im[sigma]).
    """
    ux, uy = direction
    nrm = (ux*ux + uy*uy)**0.5
    ux, uy = ux/nrm, uy/nrm

    ks = np.linspace(-kmax, kmax, Nk)
    sig = np.zeros((Nk, 3), dtype=np.complex128)

    for i, k in enumerate(ks):
        kx, ky = k*ux, k*uy
        sig[i] = eigs_at_k(kx, ky, params)

    sig = sort_eigs_continuous(sig)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for j in range(3):
        # x: k, y: Re, z: Im
        ax.plot(ks, np.real(sig[:, j]), np.imag(sig[:, j]), label=fr'branch {j+1}')

    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$\mathrm{Re}\,\sigma(k)$')
    ax.set_zlabel(r'$\mathrm{Im}\,\sigma(k)$')
    ax.set_title('3D Dispersion: k vs Re vs Im')
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_complex_plane_dispersion(params, kmax=10.0, Nk=400, direction=(1.0, 0.0)):
    """
    Plot eigenvalue trajectories on the complex plane for each branch.
    """
    ux, uy = direction
    nrm = (ux*ux + uy*uy)**0.5
    ux, uy = ux/nrm, uy/nrm

    ks = np.linspace(-kmax, kmax, Nk)
    sig = np.zeros((Nk, 3), dtype=np.complex128)

    for i, k in enumerate(ks):
        kx, ky = k*ux, k*uy
        sig[i] = eigs_at_k(kx, ky, params)

    sig = sort_eigs_continuous(sig)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    for j in range(3):
        ax.plot(np.real(sig[:, j]), np.imag(sig[:, j]), label=fr'branch {j+1}')

    ax.axhline(0.0, color='k', lw=0.8)
    ax.axvline(0.0, color='k', lw=0.8)
    ax.set_xlabel(r'$\mathrm{Re}\,\sigma$')
    ax.set_ylabel(r'$\mathrm{Im}\,\sigma$')
    ax.set_title('Complex Plane Trajectories')
    ax.set_aspect('equal', adjustable='datalim')
    ax.autoscale(enable=True, axis='both', tight=True)
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_complex_plane_evolution(params, alpha_min=0.0001*np.pi, alpha_max=1.0*np.pi, Nalpha=60, kmax=10.0, Nk=400, direction=(1.0, 0.0)):
    v, omega, lam, _, rho0, d0 = params

    ux, uy = direction
    nrm = (ux*ux + uy*uy)**0.5
    ux, uy = ux/nrm, uy/nrm

    ks = np.linspace(-kmax, kmax, Nk)
    alphas = np.linspace(alpha_min, alpha_max, Nalpha)
    sig_all = np.zeros((Nalpha, Nk, 3), dtype=np.complex128)

    for i, a in enumerate(alphas):
        p = (v, omega, lam, a, rho0, d0)
        sig = np.zeros((Nk, 3), dtype=np.complex128)
        for j, k in enumerate(ks):
            kx, ky = k*ux, k*uy
            sig[j] = eigs_at_k(kx, ky, p)
        sig_all[i] = sort_eigs_continuous(sig)
    sig_all = reorder_branches_across_alpha(sig_all)

    re_min = np.min(np.real(sig_all))
    re_max = np.max(np.real(sig_all))
    im_min = np.min(np.imag(sig_all))
    im_max = np.max(np.imag(sig_all))
    span = max(re_max - re_min, im_max - im_min)
    cx = 0.5 * (re_max + re_min)
    cy = 0.5 * (im_max + im_min)
    half = 0.5 * span if span > 0 else 1.0

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    lines = []
    for j in range(3):
        line, = ax.plot([], [], label=fr'branch {j+1}', lw=2.2)
        lines.append(line)

    ax.axhline(0.0, color='k', lw=0.8)
    ax.axvline(0.0, color='k', lw=0.8)
    ax.set_xlabel(r'$\mathrm{Re}\,\sigma$')
    ax.set_ylabel(r'$\mathrm{Im}\,\sigma$')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.legend()

    def init():
        for line in lines:
            line.set_data([], [])
        ax.set_title(fr'Complex Plane Trajectories (alpha={alphas[0]/np.pi:.4f}\pi)')
        return lines

    def update(i):
        sig = sig_all[i]
        for j in range(3):
            lines[j].set_data(np.real(sig[:, j]), np.imag(sig[:, j]))
        re_min = np.min(np.real(sig))
        re_max = np.max(np.real(sig))
        im_min = np.min(np.imag(sig))
        im_max = np.max(np.imag(sig))
        span = max(re_max - re_min, im_max - im_min)
        if span <= 0:
            span = 1.0
        pad = 0.05 * span
        cx = 0.5 * (re_max + re_min)
        cy = 0.5 * (im_max + im_min)
        half = 0.5 * span + pad
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_title(fr'Complex Plane Trajectories (alpha={alphas[i]/np.pi:.4f}\pi)')
        return lines

    anim = FuncAnimation(fig, update, frames=Nalpha, init_func=init, interval=80, blit=False)
    plt.tight_layout()
    plt.show()
    return anim

if __name__ == "__main__":
    # -------- parameters (edit here) --------
    v = 3.0
    omega = 0
    alpha = 0.5*np.pi
    rho0 = 0.0204
    d0 = 2
    lam = 20 / (rho0 * np.pi * d0**2)
    params = (v, omega, lam, alpha, rho0, d0)

    # Print eigenvalues at k=0
    print("Eigenvalues at k=(0,0):", eigs_at_k(0.0, 0.0, params))

    # 1D dispersion along x direction
    plot_1d_dispersion(params, kmax=10.0, Nk=1000, direction=(1.0, 0.0))

    plot_eigenvector_count_vs_q(params, kmax=100.0, Nk=1000, direction=(1.0, 0.0), tol=1e-8)

    # 3D dispersion plot
    plot_3d_dispersion(params, kmax=1000.0, Nk=1000, direction=(1.0, 0.0))

    anim = plot_complex_plane_evolution(params, alpha_min=0.0001*np.pi, alpha_max=1.0*np.pi, Nalpha=800, kmax=10.0, Nk=1000, direction=(1.0, 0.0))

    # 2D map of max growth and frequency
    plot_2d_dispersion(params, kmax=100.0, Nk=1000)
