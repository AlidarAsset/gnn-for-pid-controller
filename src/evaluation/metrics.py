import torch


def compute_metrics(trajectories, dt, edge_index, t_step=None, tolerance=0.02,
                    absolute_tolerance=1e-3):
    """Discrete integrals over all nodes; settling relative to last reference step.

    None means not observed within the simulation (or no reference step).
    Overshoot is direction-aware and averaged over nodes with nonzero steps.
    """
    x, r, e, u, t = (trajectories[k] for k in ('x','r','e','u','t'))
    i, j = edge_index
    out = {'ISE': float(e.square().sum()*dt), 'IAE': float(e.abs().sum()*dt),
           'U2': float(u.square().sum()*dt),
           'Sync': float((x[:,i]-x[:,j]).square().sum()*dt),
           'final_abs_error_mean': float(e[-1].abs().mean()),
           'final_abs_error_max': float(e[-1].abs().max()),
           'peak_abs_u': float(u.abs().max())}
    changes = torch.where((r[1:] != r[:-1]).any(dim=1))[0] + 1
    if t_step is not None:
        start = int(torch.searchsorted(t, t.new_tensor(t_step)))
    elif len(changes):
        start = int(changes[-1])
    else:
        start = len(t)
    out.update(settling_time=None, overshoot=None, pre_step_max_abs_x=None)
    if start >= len(t):
        return out
    if start:
        out['pre_step_max_abs_x'] = float(x[:start].abs().max())
    initial = r[start-1] if start else torch.zeros_like(r[0])
    delta = r[-1] - initial
    band = torch.maximum(tolerance*delta.abs(), torch.full_like(delta, absolute_tolerance))
    inside = ((x[start:] - r[-1]).abs() <= band).all(dim=1)
    bad = torch.where(~inside)[0]
    first = int(bad[-1])+1 if len(bad) else 0
    # Require at least one full sample interval after entry into the band.
    if first < len(inside)-1:
        out['settling_time'] = float(t[start+first]-t[start])
    active = delta.abs() > 1e-8
    if active.any():
        excess = ((x[start:,active]-r[-1,active])*delta[active].sign()).clamp_min(0)
        out['overshoot'] = float((excess.max(dim=0).values/delta[active].abs()).mean())
    return out
