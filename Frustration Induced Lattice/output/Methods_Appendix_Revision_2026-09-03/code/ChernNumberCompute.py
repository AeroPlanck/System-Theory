"""Target/complement-screened Chern calculations.

Finite radial sampling/root refinement is numerical evidence, NOT a proof
on the whole real axis. A norm bound controls the ultraviolet tail.
Return API: (C_raw, C_integer, diagnostics); invalid: (nan, None, diagnostics).
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import schur, solve_sylvester
from scipy.optimize import brentq, linear_sum_assignment, minimize_scalar
from Dispersion import M_matrix_vectorized, closure_denominator, radial_coefficients

_J = np.array([[0,0,0], [0,0,-1j], [0,1j,0]], complex)
_SPINS = np.array([-1,0,1])
_POLES = np.array([[0,1,0], [1,0,1], [-1j,0,1j]], complex)
_POLES[:, [0,2]] /= np.sqrt(2.0)


def _empty_diagnostics(status, reason):
    result = dict(valid=False, status=status, reason=reason, globally_proven=False,
                  gap_valid=False, validation_scope="not_checked", pole_chern=None,
                  min_target_complement_gap=np.nan, min_relative_gap=np.nan,
                  min_sigma=np.nan, min_det=np.nan, phase_flip_samples=[])
    for name in ("bad_edge_count", "bad_plaquette_count", "refined_plaquette_count",
                 "unresolved_bad_plaquette_count", "max_refine_level_used",
                 "near_branch_count", "phase_flip_pair_count", "phase_flip_plaquette_count"):
        result[name] = 0
    return result


def _bands(target_bands):
    raw = np.asarray(target_bands)
    if not np.issubdtype(raw.dtype, np.number) or not np.all(np.isfinite(raw)):
        raise ValueError("target_bands must contain finite integer indices")
    if np.iscomplexobj(raw) or np.any(raw != np.rint(raw)):
        raise ValueError("target_bands must contain integer indices, not fractions")
    bands = raw.astype(int)
    if bands.ndim != 1 or not 1 <= len(bands) <= 3:
        raise ValueError("target_bands must be a nonempty 1D subset of [0,1,2]")
    if np.any((bands < 0) | (bands > 2)) or len(set(bands)) != len(bands):
        raise ValueError("target_bands must contain distinct indices in [0,1,2]")
    return np.sort(bands)


def _initial_order(values, sort_by):
    if sort_by == "imag":
        return np.lexsort((values.real, values.imag))
    if sort_by == "real":
        return np.lexsort((values.imag, values.real))
    if sort_by == "abs":
        return np.lexsort((values.imag, np.abs(values)))
    raise ValueError("sort_by must be 'imag', 'real', or 'abs'")


def _track_values(matrices, sort_by):
    scale = np.maximum(np.linalg.norm(matrices, axis=(-2,-1)), 1e-300)
    raw = np.linalg.eigvals(matrices/scale[:,None,None])*scale[:,None]
    tracked = np.empty_like(raw)
    tracked[0] = raw[0, _initial_order(raw[0], sort_by)]
    for i in range(1,len(raw)):
        rows, cols = linear_sum_assignment(np.abs(tracked[i-1,:,None]-raw[i,None,:]))
        tracked[i,rows] = raw[i,cols]
    return tracked


def _discriminant(k, params):
    # Scale and long-double arithmetic reduce double-root cancellation;
    # this remains floating-point localization, not exact EP certification.
    a,b,_ = radial_coefficients(k,params)
    c = params[0]*float(k)/np.sqrt(2)
    scale = max(abs(float(a)), abs(float(b)), abs(c), 1e-300)
    a,b,c = np.asarray([a/scale,b/scale,c/scale],dtype=np.longdouble)
    A,B,C = -2*a, a*a+b*b+c*c, -a*c*c
    return float(A*A*B*B-4*B**3-4*A**3*C-27*C*C+18*A*B*C)


def _tail_error(k, params, beta):
    # |Ghat/G0|<=1. Triangle/Frobenius bound for ||B/k^2-i beta J||_2,
    # uniform for all larger k. Below |beta|/4, the normal-limit spectral
    # disks are disjoint. This certifies the tail, not the finite search.
    v,omega,lam,_,rho0,d0 = params
    coupling = abs(lam*rho0*np.pi*d0*d0)
    return (np.sqrt(2.5)*abs(v)/k
            + np.sqrt(2)*(abs(omega)+2*coupling)/k**2)/abs(beta)


@lru_cache(maxsize=64)
def _radial_search(params, sort_by, gap_samples):
    v,omega,lam,alpha,rho0,d0 = params
    D0 = closure_denominator(*params)
    if v == 0:
        raise ValueError("unresolved_uv_poles: compactification requires v != 0")
    beta = v*v/(4*D0)
    coupling = lam*rho0*np.pi*d0*d0
    b0 = -omega+0.5*coupling*np.sin(alpha)
    tail_k = max(1/d0, abs(D0/v), np.sqrt(abs(b0/beta)))
    while _tail_error(tail_k,params,beta) >= 0.20:
        tail_k *= 2
        if not np.isfinite(tail_k):
            raise ValueError("unresolved_uv_tail")
    scales = [1/d0, abs(D0/v), np.sqrt(abs(b0/beta))]
    k_min = max(min(x for x in scales if x>0)*1e-7, np.finfo(float).tiny)
    nodes = np.unique(np.r_[0., np.geomspace(k_min,tail_k,gap_samples),
                           np.linspace(0,min(tail_k,40/d0),gap_samples)])
    b = radial_coefficients(nodes,params)[1]
    b_roots = []
    # b=0 seeds resolve narrow real-root intervals near alpha endpoints.
    for i in np.flatnonzero(b[:-1]*b[1:]<0):
        b_roots.append(brentq(lambda k: float(radial_coefficients(k,params)[1]),
                             nodes[i],nodes[i+1],xtol=1e-14,rtol=1e-13))
    nodes = np.unique(np.r_[nodes,b_roots])
    disc = np.array([_discriminant(k,params) for k in nodes])
    maxima = np.flatnonzero((disc[1:-1]>=disc[:-2]) & (disc[1:-1]>=disc[2:])
                            & (disc[1:-1]>-1e-3))+1
    extra = []
    for i in maxima:
        opt = minimize_scalar(lambda k:-_discriminant(k,params),
                              bounds=(nodes[i-1],nodes[i+1]),method="bounded",
                              options={"xatol":max(1e-14,nodes[i]*1e-12)})
        extra.append(float(opt.x))
    nodes = np.unique(np.r_[nodes,extra])
    disc = np.array([_discriminant(k,params) for k in nodes])
    ep_roots = []
    for i in np.flatnonzero(disc[:-1]*disc[1:]<0):
        ep_roots.append(brentq(lambda k:_discriminant(k,params),
                              nodes[i],nodes[i+1],xtol=1e-14,rtol=1e-13))
    near_zero = [nodes[i] for i in range(1,len(nodes)-1)
                 if abs(disc[i])<1e-12 and disc[i]>=disc[i-1] and disc[i]>=disc[i+1]]
    nodes = np.unique(np.r_[nodes,ep_roots,near_zero])
    matrices = M_matrix_vectorized(nodes,np.zeros_like(nodes),*params)
    values = _track_values(matrices,sort_by)
    rows,cols = linear_sum_assignment(
        np.abs(values[-1,:,None]/tail_k**2-1j*beta*_SPINS[None,:]))
    spins = np.zeros(3,dtype=int)
    spins[rows] = _SPINS[cols]
    return dict(k=nodes, values=values, matrices=matrices, spins=spins, D0=D0,
                beta=beta, tail_k=tail_k, tail_error_bound=_tail_error(tail_k,params,beta),
                discriminant_roots=np.unique(ep_roots).tolist(),
                b_roots=np.unique(b_roots).tolist())


def _riesz_frame(matrix, selected, complement):
    """Ordered Schur/Sylvester frames remain regular at internal EPs."""
    m = len(selected)
    if m == 3:
        I = np.eye(3,dtype=complex)
        return I,I,I
    def choose(z):
        return np.min(np.abs(z-selected)) < np.min(np.abs(z-complement))
    T,Z,dimension = schur(matrix,output="complex",sort=choose)
    if dimension != m:
        raise ValueError("unresolved_schur_cluster")
    # In Schur coordinates P=[[I,X],[0,0]], with A X-X D=B.
    X = solve_sylvester(T[:m,:m],-T[m:,m:],T[:m,m:])
    R = Z[:,:m]
    L = R+Z[:,m:]@X.conj().T
    P = R@L.conj().T
    if not (np.all(np.isfinite(P))
            and np.linalg.norm(P@P-P)<=1e-7*max(1.,np.linalg.norm(P))):
        raise ValueError("unresolved_riesz_frame")
    return R,L,P


def check_spectral_separation(params,target_bands,sort_by="imag",
                              gap_rtol=1e-6,gap_atol=1e-9,gap_samples=1201):
    """Cached radial screening and conditional pole Chern formula.

    valid=True means tested finite radii and a bounded UV tail passed.
    globally_proven remains False for every nontrivial spectral cluster.
    Internal degeneracies are allowed; target/complement contacts are not.
    """
    params = tuple(float(x) for x in params)
    if len(params)!=6:
        raise ValueError("params must be (v,omega,lambda,alpha,rho0,d0)")
    bands = _bands(target_bands)
    other = np.setdiff1d(np.arange(3),bands)
    if not np.all(np.isfinite([gap_samples,gap_rtol,gap_atol])):
        raise ValueError("gap_samples and gap tolerances must be finite")
    if int(gap_samples)!=gap_samples or gap_samples<51 or gap_rtol<=0 or gap_atol<0:
        raise ValueError("integer gap_samples>=51, gap_rtol>0, gap_atol>=0 required")
    diag = _empty_diagnostics("invalid","")
    diag.update(gap_rtol=float(gap_rtol),gap_atol=float(gap_atol),
                gap_samples=int(gap_samples),target_bands=bands.tolist(),
                sort_by=sort_by,internal_degeneracies_allowed=True)
    try:
        D0 = closure_denominator(*params)
        diag["D0"] = float(D0)
        if len(bands)==3:
            diag.update(valid=True,gap_valid=True,status="valid_full_space",
                        reason="P=I3; no complementary spectrum",pole_chern=0,
                        globally_proven=True,infinity_spins=[-1,0,1],
                        validation_scope="identity bundle of a finite closed matrix")
            return diag
        data = _radial_search(params,sort_by,int(gap_samples))
    except ValueError as exc:
        diag.update(status="invalid",reason=str(exc))
        return diag
    values,nodes = data["values"],data["k"]
    gaps = np.abs(values[:,bands,None]-values[:,None,other]).min(axis=(1,2))
    frequency = max(abs(params[1]),abs(params[2]*params[4]*np.pi*params[5]**2),
                    abs(params[0]/params[5]),1e-300)
    scales = np.maximum(np.max(np.abs(values),axis=1),frequency)
    relative = gaps/scales
    index = int(np.argmin(relative))
    diag.update(min_target_complement_gap=float(gaps.min()),
                min_relative_gap=float(relative[index]),worst_k=float(nodes[index]),
                finite_k_max=float(data["tail_k"]),radial_sample_count=len(nodes),
                discriminant_roots=data["discriminant_roots"],b_zero_roots=data["b_roots"],
                uv_tail_relative_error_bound=float(data["tail_error_bound"]),
                uv_tail_separation_certified=True,infinity_spins=data["spins"][bands].tolist(),
                validation_scope="finite radial sampling/root refinement; bounded UV tail",
                limitation="No interval/global finite-k proof; refine gap_samples.")
    if np.any(gaps<=gap_atol+gap_rtol*scales):
        diag.update(status="invalid_target_complement_gap",
                    reason="target_complement_gap_below_tolerance")
        return diag
    try:
        _,_,P0 = _riesz_frame(data["matrices"][0],values[0,bands],values[0,other])
        zero_spin = np.trace(_J@P0)
    except ValueError as exc:
        diag.update(status="invalid_origin_projector",reason=str(exc))
        return diag
    pole = -(sum(data["spins"][bands])-zero_spin)
    if abs(pole.imag)>1e-7 or abs(pole.real-np.rint(pole.real))>1e-7:
        diag.update(status="invalid_poles",reason="unresolved_pole_projector")
        return diag
    diag.update(valid=True,gap_valid=True,status="valid_numerical",
                reason="target/complement separated on tested radial set and UV tail",
                zero_spin=float(zero_spin.real),pole_chern=int(np.rint(pole.real)))
    return diag


def _basis_projector(R):
    if not np.all(np.isfinite(R)):
        raise ValueError("infty_basis must be finite")
    if np.linalg.matrix_rank(R)!=R.shape[1]:
        raise ValueError("infty_basis must have full column rank")
    return R@np.linalg.solve(R.conj().T@R,R.conj().T)


def compute_topology(params,target_bands,Q=10.,N_theta=41,N_phi=61,delta=1e-3,
                     holes=None,infty_basis=None,sort_by="imag",det_tol=1e-7,
                     svd_tol=1e-6,overlap_weight=0.4,eig_weight=0.6,
                     phase_branch_tol=0.1,phase_jump_tol=3.,refine_bad=True,
                     refine_max_level=3,gap_rtol=1e-6,gap_atol=1e-9,
                     gap_samples=1201,cap_tol=5e-3):
    """Determinant-link flux after explicit target/complement screening.

    Historical matching/refinement keywords remain accepted. Ordered Schur
    frames replace unstable eigenvector matching; unresolved links are
    invalid, never omitted and then rounded into a misleading integer.
    """
    params = tuple(float(x) for x in params)
    bands = _bands(target_bands)
    m = len(bands)
    if not np.all(np.isfinite([Q,N_theta,N_phi,delta])):
        raise ValueError("compactification and mesh parameters must be finite")
    if Q<=0 or N_theta<3 or N_phi<5 or not 0<delta<np.pi:
        raise ValueError("Q>0, N_theta>=3, N_phi>=5, 0<delta<pi required")
    if int(N_theta)!=N_theta or int(N_phi)!=N_phi:
        raise ValueError("mesh counts must be integers")
    N_theta,N_phi = int(N_theta),int(N_phi)
    if not np.isfinite(cap_tol) or not 0<cap_tol<0.25:
        raise ValueError("cap_tol must be finite and in (0,0.25)")
    if not np.all(np.isfinite([det_tol,svd_tol,phase_branch_tol])):
        raise ValueError("link tolerances must be finite")
    if det_tol<=0 or svd_tol<0 or not 0<phase_branch_tol<np.pi:
        raise ValueError("det_tol>0, svd_tol>=0, 0<phase_branch_tol<pi required")
    diag = check_spectral_separation(params,bands,sort_by,gap_rtol,gap_atol,gap_samples)
    diag.update(det_tol=float(det_tol),svd_tol=float(svd_tol),
                phase_branch_tol=float(phase_branch_tol),phase_jump_tol=float(phase_jump_tol),
                refine_bad=bool(refine_bad),refine_max_level=int(refine_max_level),
                frame_method="ordered Schur/Sylvester Riesz frames",
                historical_matching_weights=[float(overlap_weight),float(eig_weight)],
                compatibility_only_arguments=["svd_tol","overlap_weight","eig_weight",
                                              "phase_jump_tol","refine_bad","refine_max_level"])
    if not diag["valid"]:
        return np.nan,None,diag
    if any(radius>0 for _,_,radius in (holes or [])):
        diag.update(valid=False,status="invalid_punctured_base",
                    reason="positive-radius holes require boundary transition data")
        return np.nan,None,diag
    if m==3:
        diag.update(min_sigma=1.,min_det=1.,cap_projector_error=0.,
                    flux_raw=0.,integer_residual=0.)
        return 0.,0,diag
    data = _radial_search(params,sort_by,int(gap_samples))
    other = np.setdiff1d(np.arange(3),bands)
    expected = _POLES[:,[int(np.flatnonzero(_SPINS==s)[0])
                        for s in data["spins"][bands]]]
    # Continue the requested origin-labelled cluster to its actual UV spin
    # sectors; a band index is not a parameter-independent circular polarization.
    R_inf = expected.copy() if infty_basis is None else np.asarray(infty_basis,complex)
    diag["cap_selection"] = "tracked_asymptotic_spin_sectors" if infty_basis is None else "validated_user_basis"
    if R_inf.shape!=(3,m):
        raise ValueError("infty_basis must have shape (3,m)")
    mismatch = np.linalg.norm(_basis_projector(R_inf)-_basis_projector(expected))
    diag["cap_basis_mismatch"] = float(mismatch)
    if mismatch>cap_tol:
        diag.update(valid=False,status="invalid_cap_basis",
                    reason="default/supplied infinity basis disagrees with tracked UV sectors")
        return np.nan,None,diag
    cap_k = max(data["tail_k"],Q*np.tan(0.5*(np.pi-delta)))
    while _tail_error(cap_k,params,data["beta"])>cap_tol:
        cap_k *= 2
    nominal = Q*np.tan(0.5*np.linspace(0,np.pi-delta,N_theta))
    nodes = np.unique(np.r_[data["k"],nominal,
                           np.geomspace(data["tail_k"],cap_k,65)])
    matrices = M_matrix_vectorized(nodes,np.zeros_like(nodes),*params)
    values = _track_values(matrices,sort_by)
    gaps = np.abs(values[:,bands,None]-values[:,None,other]).min(axis=(1,2))
    scale = np.maximum(np.max(np.abs(values),axis=1),1e-300)
    if np.any(gaps<=gap_atol+gap_rtol*scale):
        diag.update(valid=False,status="invalid_refined_gap",
                    reason="target/complement contact on augmented flux grid")
        return np.nan,None,diag
    right,left = [],[]
    residual = 0.
    try:
        for matrix,vals in zip(matrices,values):
            R,L,P = _riesz_frame(matrix,vals[bands],vals[other])
            residual = max(residual,float(np.linalg.norm(matrix@P-P@matrix)
                           /max(1.,np.linalg.norm(matrix)*np.linalg.norm(P))))
            right.append(R)
            left.append(L)
    except (ValueError,np.linalg.LinAlgError) as exc:
        diag.update(valid=False,status="invalid_frames",reason=str(exc))
        return np.nan,None,diag
    right,left = np.asarray(right),np.asarray(left)
    cap_error = np.linalg.norm(right[-1]@left[-1].conj().T-_basis_projector(expected))
    diag.update(cap_projector_error=float(cap_error),cap_k=float(cap_k),
                max_projector_commutator=residual,flux_radial_sample_count=len(nodes))
    if cap_error>cap_tol or residual>1e-7:
        diag.update(valid=False,status="invalid_cap_convergence",
                    reason="cap or projector residual exceeds tolerance")
        return np.nan,None,diag
    phi = np.linspace(0,2*np.pi,N_phi,endpoint=False)
    S = np.zeros((N_phi,3,3),complex)
    S[:,0,0] = 1
    S[:,1,1] = S[:,2,2] = np.cos(phi)
    S[:,1,2],S[:,2,1] = -np.sin(phi),np.sin(phi)
    Rg = np.einsum("pij,njm->npim",S,right)
    Lg = np.einsum("pij,njm->npim",S,left)
    Rg[0],Lg[0] = right[0],left[0]
    L_inf = R_inf@np.linalg.inv(R_inf.conj().T@R_inf)
    Rg = np.concatenate([Rg,np.broadcast_to(R_inf,(1,N_phi,3,m))])
    Lg = np.concatenate([Lg,np.broadcast_to(L_inf,(1,N_phi,3,m))])
    Wr = np.einsum("npim,npij->npmj",Lg[:-1].conj(),Rg[1:])
    Wp = np.einsum("npim,npij->npmj",Lg.conj(),np.roll(Rg,-1,axis=1))
    dr,dp = np.linalg.det(Wr),np.linalg.det(Wp)
    ar,ap = np.abs(dr),np.abs(dp)
    br,bp = ar<det_tol,ap<det_tol
    bad = br|np.roll(br,-1,axis=1)|bp[:-1]|bp[1:]
    diag.update(min_sigma=1.,min_det=float(min(ar.min(),ap.min())),
                bad_edge_count=int(br.sum()+bp.sum()),bad_plaquette_count=int(bad.sum()),
                unresolved_bad_plaquette_count=int(bad.sum()),
                min_sigma_meaning="L^dagger R=I by construction; NOT a spectral gap")
    if np.any(bad):
        diag.update(valid=False,status="invalid_links",
                    reason="unresolved determinant link; no plaquette omitted")
        return np.nan,None,diag
    Ur,Up = dr/ar,dp/ap
    phases = np.angle(Ur*Up[1:]*np.conj(np.roll(Ur,-1,axis=1))*np.conj(Up[:-1]))
    near = int(np.sum(np.abs(phases)>=np.pi-phase_branch_tol))
    raw = float(phases.sum()/(2*np.pi))
    rounded = int(np.rint(raw))
    diag.update(near_branch_count=near,flux_raw=raw,integer_residual=abs(raw-rounded),
                max_plaquette_phase=float(np.max(np.abs(phases))),
                refinement_status="independent radial refinement; unresolved angular flux rejected")
    if near or abs(raw-rounded)>1e-6 or rounded!=diag["pole_chern"]:
        diag.update(valid=False,status="invalid_flux_resolution",
                    reason="branch-cut, integer or independent pole check failed")
        return np.nan,None,diag
    diag.update(status="valid_numerical",
                reason="radial screening, UV cap, determinant links and pole cross-check passed")
    return raw,rounded,diag


if __name__=="__main__":
    for fraction in (0.,0.01,0.2,0.5,0.8,0.99,1.):
        p = (3.,0.,20.75/np.pi,fraction*np.pi,1.,1.)
        for bands in ((0,),(0,2),(0,1,2)):
            raw,integer,diag = compute_topology(p,bands,N_phi=31)
            print(fraction,bands,raw,integer,diag["status"])
