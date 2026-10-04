"""Moment-matched mixtures on real features: build a bulk law whose mean embedding matches a tail law, mix both into a base sampler, and compare tail ratios and Frechet distances."""
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import analysis_common as E                                        # noqa: E402

t0 = time.time()
SMOKE = os.environ.get("E7_SMOKE", "0") == "1"
P = E.TAU_P
CS = [2.0, 5.0]
BASES = ["EDM", "StyleGAN-XL"]
N_POOL = int(os.environ.get("E7_NPOOL", "150000")) if not SMOKE else 6_000
HULL_SWEEP = [int(x) for x in
              os.environ.get("E7_HULL_SWEEP", "").split(",") if x]
D_SUB = 128 if not SMOKE else 16
N_REP = 10 if not SMOKE else 2
M_EVAL = 100_000 if not SMOKE else 4_000
os.makedirs(E.P3, exist_ok=True)

tau = E.tau3()
R50 = E.real_r50()
ref_idx, pool_idx = E.ref_pool_split()
Ereal = E.real_emb()
print(f"tau(1e-3) = {tau:.6f}   REF n={len(ref_idx)}  POOL n={len(pool_idx)}",
      flush=True)

# ---------------- real reference moments (shared with ) ------------------
REF_CACHE = f"{E.P3}/e7_ref_moments.npz"
if os.path.exists(REF_CACHE) and not SMOKE:
    z = np.load(REF_CACHE)
    mu_ref, S_ref = z["mu"], z["S"]
    assert np.array_equal(z["ref_idx"], ref_idx), "REF cache split mismatch"
    print("loaded cached REF moments", flush=True)
else:
    mu_ref, S_ref = E.moments(Ereal[ref_idx if not SMOKE else ref_idx[:40000]])
    if not SMOKE:
        np.savez(REF_CACHE, mu=mu_ref, S=S_ref, ref_idx=ref_idx)
C_ref = E.cov_of(mu_ref, S_ref)
print(f"REF tr(C)={np.trace(C_ref):.4f}  ({time.time()-t0:.0f}s)", flush=True)

# whitened top-D_SUB principal coordinates of the REAL feature law
evals, evecs = np.linalg.eigh(C_ref)
lam_d = evals[::-1][:D_SUB].copy()
U_d = evecs[:, ::-1][:, :D_SUB].copy()
evr = lam_d.sum() / np.trace(C_ref)
Wd = U_d / np.sqrt(lam_d)
print(f"top-{D_SUB} real-feature subspace carries {evr:.4f} of the variance",
      flush=True)


def whiten(X):
    return (np.asarray(X, np.float64) - mu_ref) @ Wd


# ---------------- tail law Q_T and the bulk candidate pool ------------------
Rp = R50[pool_idx]
tail_idx = pool_idx[Rp >= tau]
bulk_all = pool_idx[Rp < tau]
rng = np.random.default_rng(70701)
bulk_idx = np.sort(rng.choice(bulk_all, size=min(N_POOL, len(bulk_all)),
                              replace=False))
print(f"Q_T atoms = {len(tail_idx)}   bulk pool = {len(bulk_idx)} "
      f"(of {len(bulk_all)} eligible)", flush=True)

Phi_T = np.asarray(Ereal[tail_idx], np.float64)
mu_T, S_T = E.moments(Phi_T)
Phi_pool = np.asarray(Ereal[bulk_idx], np.float64)
Z_pool = whiten(Phi_pool)
Z_T = whiten(Phi_T)
bS_T = (Z_T.T @ Z_T) / len(Z_T)
print(f"pool loaded {Phi_pool.shape} ({time.time()-t0:.0f}s)", flush=True)

# scale references for the residuals
disp_T = float(np.linalg.norm(mu_T - mu_ref))          # tail law's own offset
sub = rng.choice(len(Phi_pool), size=min(4000, len(Phi_pool)), replace=False)
dd = Phi_pool[sub[:2000]] - Phi_pool[sub[2000:4000]]
med_pair = float(np.median(np.linalg.norm(dd, axis=1)))
print(f"||mu_T - mu_ref|| = {disp_T:.5f}   median real-real feature distance "
      f"= {med_pair:.5f}", flush=True)
zn_pool = (Z_pool ** 2).sum(1)
zn_tail = (Z_T ** 2).sum(1)
print(f"whitened top-{D_SUB} energies: tail mean={zn_tail.mean():.2f} "
      f"max={zn_tail.max():.2f} | bulk pool mean={zn_pool.mean():.2f} "
      f"max={zn_pool.max():.2f} | frac of bulk above the tail mean = "
      f"{(zn_pool >= zn_tail.mean()).mean():.4f}", flush=True)
print(f"||E_QT[zz^T]||_F = {np.linalg.norm(bS_T):.4f}  "
      f"tr = {np.trace(bS_T):.4f}  (bulk-pool tr = {zn_pool.mean():.4f})",
      flush=True)

# ---------------- build the bulk laws ---------------------------------------
laws = {}
constr = []

print("\n[QB sparse] fully-corrective Frank-Wolfe (Caratheodory witness)",
      flush=True)
s_idx, s_w, s_res = E.fcfw_simplex_ls(
    Phi_pool, mu_T, max_atoms=769 if not SMOKE else 60,
    inner=1200, tol=1e-9 * max(disp_T, 1e-12))
laws["sparse"] = dict(idx=bulk_idx[s_idx], w=s_w, local=s_idx)
print(f"  support={len(s_idx)} atoms (Caratheodory bound k+1 = 769), "
      f"resid={s_res:.3e}  ({time.time()-t0:.0f}s)", flush=True)

print("\n[QB dense] accelerated projected gradient on the full pool "
      "(warm-started from the Frank-Wolfe witness)", flush=True)
w_warm = np.zeros(len(Phi_pool))
w_warm[s_idx] = s_w
d_w, d_res = E.simplex_ls(Phi_pool, mu_T, iters=3000 if not SMOKE else 300,
                          w0=w_warm, verbose=500, tag="dense")
if d_res > s_res:                    # never ship a worse witness than FCFW's
    d_w, d_res = w_warm, s_res
laws["dense"] = dict(idx=bulk_idx, w=d_w, local=np.arange(len(bulk_idx)))
print(f"  resid={d_res:.3e}  nnz(>1e-12)={int((d_w > 1e-12).sum())} "
      f"({time.time()-t0:.0f}s)", flush=True)

print("\n[QB mom2] mean in R^768 + second moment in the top-D subspace",
      flush=True)
m_w, m_rmu, m_rS = E.simplex_ls_mom2(
    Phi_pool, Z_pool, mu_T, bS_T,
    iters=int(os.environ.get("E7_MOM2_ITERS", "1500")) if not SMOKE else 200,
    lam=1.0, verbose=250, w0=d_w)
laws["mom2"] = dict(idx=bulk_idx, w=m_w, local=np.arange(len(bulk_idx)))
print(f"  |dmu|={m_rmu:.3e}  |dS_d|_F={m_rS:.3e}  "
      f"nnz={int((m_w > 1e-12).sum())}  ({time.time()-t0:.0f}s)", flush=True)

r_sel = rng.choice(len(bulk_idx), size=min(5000, len(bulk_idx)), replace=False)
# noqa -- QB_random is the unmatched control

# ---- is E_{Q_T} psi in the convex hull of psi(B_tau)? residual vs pool size
hull_rows = []
if HULL_SWEEP:
    print("\n[hull sweep] min ||sum w psi(x_i) - E_{Q_T}psi|| over the simplex,"
          " as the bulk candidate pool grows", flush=True)
    hperm = np.random.default_rng(4711).permutation(len(Phi_pool))
    for npool in HULL_SWEEP:
        npool = min(npool, len(Phi_pool))
        sidx, sw, sres = E.fcfw_simplex_ls(
            np.ascontiguousarray(Phi_pool[hperm[:npool]]), mu_T,
            max_atoms=769, inner=900, tol=1e-10, verbose=False)
        hull_rows.append(dict(n_pool=npool, support=len(sidx),
                              resid_abs=sres,
                              resid_rel_tail_offset=sres / disp_T,
                              resid_rel_median_pair=sres / med_pair))
        print(f"  N={npool:7d}: resid={sres:.4e}  "
              f"({sres/disp_T:.3e} x tail offset)  support={len(sidx)} "
              f"({time.time()-t0:.0f}s)", flush=True)
    E.writecsv(f"{E.DATA}/prop1_hull_sweep.csv", hull_rows)
laws["random"] = dict(idx=bulk_idx[np.sort(r_sel)],
                      w=np.full(len(r_sel), 1.0 / len(r_sel)),
                      local=np.sort(r_sel))
print(f"[QB random] unmatched control on {len(r_sel)} random bulk images",
      flush=True)

# ---------------- moments + residual diagnostics of every law ---------------
mom = {"tail": (mu_T, S_T)}
for k, L in laws.items():
    sub_pool = (Phi_pool if len(L["local"]) == len(Phi_pool)
                else Phi_pool[L["local"]])
    mu_b, S_b = E.moments(sub_pool, w=L["w"])
    mom[k] = (mu_b, S_b)
    Zb = Z_pool[L["local"]]
    Sd_b = (Zb * L["w"][:, None]).T @ Zb
    dmu = float(np.linalg.norm(mu_b - mu_T))
    dS = float(np.linalg.norm(S_b - S_T))
    dSd = float(np.linalg.norm(Sd_b - bS_T))
    rel_mu = dmu / float(np.linalg.norm(mu_T))
    rel_S = dS / float(np.linalg.norm(S_T))
    constr.append(dict(
        bulk_law=k, n_support=int((L["w"] > 1e-12).sum()),
        n_pool_candidates=len(L["w"]),
        resid_mean_abs=dmu,
        resid_mean_rel_to_muT=rel_mu,
        resid_mean_rel_to_tail_offset=dmu / disp_T,
        resid_mean_rel_to_median_pair_dist=dmu / med_pair,
        resid_2ndmoment_full_fro=dS,
        resid_2ndmoment_full_rel=rel_S,
        resid_2ndmoment_topD_fro=dSd,
        resid_2ndmoment_topD_rel=dSd / float(np.linalg.norm(bS_T)),
        # selection rule fixed in advance: the canonical Q_B is the law with
        # the smallest COMBINED relative residual of the FID moment map
        # psi = (phi, phi phi^T).
        resid_fid_momentmap_combined=float(np.hypot(rel_mu, rel_S)),
        caratheodory_bound=769, D_sub=D_SUB, evr_topD=float(evr)))
    print(f"  {k:7s}: |dmu|={dmu:.3e} ({dmu/disp_T:.2e} x tail offset)  "
          f"|dS|_F={dS:.4f}  |dS_d|_F={dSd:.4f}", flush=True)
E.writecsv(f"{E.DATA}/prop1_construction.csv", constr)

# persist the constructed laws so re-uses the SAME Q_B / Q_T (no re-solve)
np.savez(f"{E.P3}/e7a_laws.npz",
         tail_idx=tail_idx, bulk_idx=bulk_idx, ref_idx=ref_idx,
         tau=tau, p=P, D_sub=D_SUB,
         **{f"w_{k}": laws[k]["w"] for k in laws},
         **{f"loc_{k}": laws[k]["local"] for k in laws})
print(f"wrote {E.P3}/e7a_laws.npz", flush=True)

# ---------------- law-level FMD, rho, and moment-vector gaps ----------------
rows = []
fig = {}
for base in BASES:
    Eb = E.load_model_emb(base)
    Rb = E.load_model_r50(base)
    if SMOKE:
        Eb, Rb = Eb[:20000], Rb[:20000]
    assert len(Eb) == len(Rb)
    mu0, S0 = E.moments(Eb)
    frac0 = float((Rb >= tau).mean())
    fmd0 = E.frechet(mu0, E.cov_of(mu0, S0), mu_ref, C_ref)
    # MED: the moment metric whose moment map is exactly the one we match,
    # psi = phi. M(P,Q) = ||E_Q phi - E_P phi||^2 (a function of E_Q psi).
    def med(mu):
        d = mu - mu_ref
        return float(d @ d)
    med0 = med(mu0)
    rho0 = frac0 / P
    print(f"\n=== base {base}: M={len(Eb)} rho_0={rho0:.4f} FMD_0={fmd0:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    for c in CS:
        eps = c * P
        # Q_2 (tail arm)
        mu2, S2 = E.mix_moments(mu0, S0, mu_T, S_T, eps)
        fmd2 = E.frechet(mu2, E.cov_of(mu2, S2), mu_ref, C_ref)
        rho2 = ((1 - eps) * frac0 + eps) / P
        med2 = med(mu2)
        rows.append(dict(base=base, c=c, eps=eps, p=P, tau=tau, M=len(Eb),
                         arm="Q2_tail", bulk_law="", rho_law=rho2,
                         fmd_law=fmd2, delta_fmd_law=0.0, delta_rho_law=0.0,
                         med_law=med2, delta_med_law=0.0,
                         moment_gap_mean=0.0, moment_gap_2nd_fro=0.0,
                         rho_sample_mean="", rho_sample_sd="",
                         fmd_sample_mean="", fmd_sample_sd="",
                         delta_fmd_sample_mean="", delta_fmd_sample_sd=""))
        rho1 = (1 - eps) * frac0 / P
        for k in ["sparse", "dense", "mom2", "random"]:
            mu_b, S_b = mom[k]
            mu1, S1 = E.mix_moments(mu0, S0, mu_b, S_b, eps)
            fmd1 = E.frechet(mu1, E.cov_of(mu1, S1), mu_ref, C_ref)
            rows.append(dict(
                base=base, c=c, eps=eps, p=P, tau=tau, M=len(Eb),
                arm="Q1_bulk", bulk_law=k, rho_law=rho1, fmd_law=fmd1,
                delta_fmd_law=fmd1 - fmd2, delta_rho_law=rho1 - rho2,
                med_law=med(mu1), delta_med_law=med(mu1) - med2,
                moment_gap_mean=float(np.linalg.norm(mu1 - mu2)),
                moment_gap_2nd_fro=float(np.linalg.norm(S1 - S2)),
                rho_sample_mean="", rho_sample_sd="",
                fmd_sample_mean="", fmd_sample_sd="",
                delta_fmd_sample_mean="", delta_fmd_sample_sd=""))
            print(f"  c={c:g} {k:7s}: rho1={rho1:.4f} rho2={rho2:.4f} "
                  f"(diff {rho2-rho1:.4f})  FMD1={fmd1:.5f} FMD2={fmd2:.5f} "
                  f"(diff {fmd1-fmd2:+.6f})  dMED={med(mu1)-med2:+.3e}",
                  flush=True)
        rows.append(dict(base=base, c=c, eps=eps, p=P, tau=tau, M=len(Eb),
                         arm="Q0_base", bulk_law="", rho_law=rho0,
                         fmd_law=fmd0, delta_fmd_law=fmd0 - fmd2,
                         delta_rho_law=rho0 - rho2,
                         med_law=med0, delta_med_law=med0 - med2,
                         moment_gap_mean="",
                         moment_gap_2nd_fro="", rho_sample_mean="",
                         rho_sample_sd="", fmd_sample_mean="",
                         fmd_sample_sd="", delta_fmd_sample_mean="",
                         delta_fmd_sample_sd=""))

    # ------------ finite-M realisations, paired on the base draws ----------
    for c in CS:
        eps = c * P
        n_mix = int(round(eps * M_EVAL))
        acc = {k: dict(fmd=[], rho=[]) for k in
               ["Q2_tail", "sparse", "dense", "mom2", "random"]}
        dlt = {k: [] for k in ["sparse", "dense", "mom2", "random"]}
        rr = np.random.default_rng(90210 + int(c * 10))
        for rep in range(N_REP):
            b_sel = rr.choice(len(Eb), size=M_EVAL - n_mix, replace=True)
            Xb = Eb[b_sel]
            mu_b0, S_b0 = E.moments(Xb)
            kb = int((Rb[b_sel] >= tau).sum())
            t_sel = rr.choice(len(tail_idx), size=n_mix, replace=True)
            arms = {"Q2_tail": (Phi_T[t_sel], n_mix)}
            for k in ["sparse", "dense", "mom2", "random"]:
                L = laws[k]
                sel = rr.choice(L["local"], size=n_mix, replace=True,
                                p=L["w"] / L["w"].sum())
                arms[k] = (Phi_pool[sel], 0)
            base_w = (M_EVAL - n_mix) / M_EVAL
            for k, (Xm, k_extra) in arms.items():
                mu_m, S_m = E.moments(Xm)
                mu = base_w * mu_b0 + (1 - base_w) * mu_m
                S = base_w * S_b0 + (1 - base_w) * S_m
                f = E.frechet(mu, E.cov_of(mu, S), mu_ref, C_ref)
                acc[k]["fmd"].append(f)
                acc[k]["rho"].append((kb + k_extra) / M_EVAL / P)
            for k in dlt:
                dlt[k].append(acc[k]["fmd"][-1] - acc["Q2_tail"]["fmd"][-1])
        for r in rows:
            if r["base"] != base or r["c"] != c:
                continue
            key = "Q2_tail" if r["arm"] == "Q2_tail" else r["bulk_law"]
            if key not in acc:
                continue
            r["rho_sample_mean"] = float(np.mean(acc[key]["rho"]))
            r["rho_sample_sd"] = float(np.std(acc[key]["rho"], ddof=1))
            r["fmd_sample_mean"] = float(np.mean(acc[key]["fmd"]))
            r["fmd_sample_sd"] = float(np.std(acc[key]["fmd"], ddof=1))
            if key in dlt:
                r["delta_fmd_sample_mean"] = float(np.mean(dlt[key]))
                r["delta_fmd_sample_sd"] = float(np.std(dlt[key], ddof=1))
        print(f"  [finite M={M_EVAL}, {N_REP} reps, c={c:g}] "
              f"rho: Q2={np.mean(acc['Q2_tail']['rho']):.3f} "
              f"mom2={np.mean(acc['mom2']['rho']):.3f} | "
              f"dFMD(mom2-Q2)={np.mean(dlt['mom2']):+.5f} "
              f"+/-{np.std(dlt['mom2'], ddof=1):.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    fig[base] = dict(fmd0=fmd0, rho0=rho0, M=len(Eb))
    del Eb, Rb

# ---------------- audited yardsticks -------------------------------------------
print("\n=== yardsticks ===", flush=True)
configs_mu, configs_S, configs_fmd = {}, {}, {}
for name in E.CONFIG_NAMES:
    Em = E.load_model_emb(name)
    if SMOKE:
        Em = Em[:5000]
    mu, S = E.moments(Em)
    configs_mu[name], configs_S[name] = mu, S
    configs_fmd[name] = E.frechet(mu, E.cov_of(mu, S), mu_ref, C_ref)
    del Em
names = list(configs_mu)
pair_mu = [(a, b, float(np.linalg.norm(configs_mu[a] - configs_mu[b])))
           for i, a in enumerate(names) for b in names[i + 1:]]
pair_S = [(a, b, float(np.linalg.norm(configs_S[a] - configs_S[b])))
          for i, a in enumerate(names) for b in names[i + 1:]]
fv = sorted(configs_fmd.values())
adj = [fv[i + 1] - fv[i] for i in range(len(fv) - 1)]
min_mu_gap = min(g for _, _, g in pair_mu)
min_S_gap = min(g for _, _, g in pair_S)
arg_mu = min(pair_mu, key=lambda t: t[2])
print(f"closest audited pair in mean-embedding: {arg_mu[0]}/{arg_mu[1]} "
      f"= {min_mu_gap:.5f}", flush=True)
print(f"closest audited pair in 2nd moment (Fro): {min_S_gap:.5f}", flush=True)
print(f"adjacent audited FMD gaps (audited M): min={min(adj):.5f} "
      f"median={np.median(adj):.5f}", flush=True)

for r in rows:
    r["yard_min_adjacent_zoo_fmd_gap"] = float(min(adj))
    r["yard_median_adjacent_zoo_fmd_gap"] = float(np.median(adj))
    r["yard_min_zoo_pair_mean_gap"] = min_mu_gap
    r["yard_min_zoo_pair_2nd_gap"] = min_S_gap
E.writecsv(f"{E.DATA}/prop1_empirical.csv", rows)

np.savez(f"{E.P3}/e7a_results.npz",
         configs_names=np.array(names),
         configs_fmd=np.array([configs_fmd[n] for n in names]),
         adj_gaps=np.array(adj), min_mu_gap=min_mu_gap, min_S_gap=min_S_gap,
         tau=tau, p=P, disp_T=disp_T, med_pair=med_pair,
         n_tail=len(tail_idx), n_pool=len(bulk_idx),
         sparse_support=len(s_idx), sparse_resid=s_res,
         dense_resid=d_res, mom2_resid_mu=m_rmu, mom2_resid_S=m_rS,
         evr_topD=float(evr), D_sub=D_SUB)
print(f"\nE7a DONE ({time.time()-t0:.0f}s)", flush=True)
