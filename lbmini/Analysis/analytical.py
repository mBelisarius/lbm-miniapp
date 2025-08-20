import numpy as np

from scipy import optimize


def sound_speed(gamma, pressure, density, dust_frac=0.0):
    scale = np.sqrt(1.0 - dust_frac)
    return np.sqrt(gamma * pressure / density) * scale


def shock_tube_function(p4, p1, p5, rho1, rho5, gamma, dust_frac=0.0):
    z = (p4 / p5 - 1.0)
    c1 = sound_speed(gamma, p1, rho1, dust_frac)
    c5 = sound_speed(gamma, p5, rho5, dust_frac)

    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    g2 = 2.0 * gamma

    fact = gm1 / g2 * (c5 / c1) * z / np.sqrt(1.0 + gp1 / g2 * z)
    fact = (1.0 - fact) ** (g2 / gm1)

    return p1 * fact - p4


def calculate_regions(pl, rhol, ul, pr, rhor, ur, gamma=1.4, dust_frac=0.0):
    rho1 = rhol
    p1 = pl
    u1 = ul
    rho5 = rhor
    p5 = pr
    u5 = ur

    if pl < pr:
        rho1 = rhor
        p1 = pr
        u1 = ur
        rho5 = rhol
        p5 = pl
        u5 = ul

    try:
        p4 = optimize.fsolve(
            shock_tube_function, p1,
            (p1, p5, rho1, rho5, gamma, dust_frac),
        )[0]
    except Exception:
        a = min(p1, p5) * 1e-8
        b = max(p1, p5) * 1e6

        def ftarget(p):
            return shock_tube_function(p, p1, p5, rho1, rho5, gamma, dust_frac)

        p4 = optimize.brentq(ftarget, a, b)

    z = (p4 / p5 - 1.0)
    c5 = sound_speed(gamma, p5, rho5, dust_frac)

    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    gmfac1 = 0.5 * gm1 / gamma
    gmfac2 = 0.5 * gp1 / gamma

    fact = np.sqrt(1.0 + gmfac2 * z)

    u4 = c5 * z / (gamma * fact)
    rho4 = rho5 * (1.0 + gmfac2 * z) / (1.0 + gmfac1 * z)

    w = c5 * fact

    p3 = p4
    u3 = u4
    rho3 = rho1 * (p3 / p1) ** (1.0 / gamma)

    return (p1, rho1, u1), (p3, rho3, u3), (p4, rho4, u4), (p5, rho5, u5), w


def calc_positions(pl, pr, region1, region3, w, xi, t, gamma, dust_frac=0.0):
    p1, rho1 = region1[:2]
    p3, rho3, u3 = region3
    c1 = sound_speed(gamma, p1, rho1, dust_frac)
    c3 = sound_speed(gamma, p3, rho3, dust_frac)

    if pl > pr:
        xsh = xi + w * t
        xcd = xi + u3 * t
        xft = xi + (u3 - c3) * t
        xhd = xi - c1 * t
    else:
        xsh = xi - w * t
        xcd = xi - u3 * t
        xft = xi - (u3 - c3) * t
        xhd = xi + c1 * t

    return xhd, xft, xcd, xsh


def region_states(pl, pr, region1, region3, region4, region5):
    if pl > pr:
        return {'Region 1': region1,
                'Region 2': 'RAREFACTION',
                'Region 3': region3,
                'Region 4': region4,
                'Region 5': region5}
    else:
        return {'Region 1': region5,
                'Region 2': region4,
                'Region 3': region3,
                'Region 4': 'RAREFACTION',
                'Region 5': region1}


def create_arrays(pl, pr, xl, xr, positions, state1, state3, state4, state5, npts, gamma, t, xi, dust_frac=0.0):
    xhd, xft, xcd, xsh = positions
    p1, rho1, u1 = state1
    p3, rho3, u3 = state3
    p4, rho4, u4 = state4
    p5, rho5, u5 = state5
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    x_arr = np.linspace(xl, xr, npts)
    rho = np.zeros(npts, dtype=float)
    p = np.zeros(npts, dtype=float)
    u = np.zeros(npts, dtype=float)
    c1 = sound_speed(gamma, p1, rho1, dust_frac)

    if t == 0.0:
        for i, x in enumerate(x_arr):
            if x < xi:
                rho[i] = rho1
                p[i] = p1
                u[i] = u1
            else:
                rho[i] = rho5
                p[i] = p5
                u[i] = u5
        return x_arr, p, rho, u

    if pl > pr:
        for i, x in enumerate(x_arr):
            if x < xhd:
                rho[i] = rho1
                p[i] = p1
                u[i] = u1
            elif x < xft:
                u_i = 2.0 / gp1 * (c1 + (x - xi) / t)
                fact = 1.0 - 0.5 * gm1 * u_i / c1
                rho[i] = rho1 * fact ** (2.0 / gm1)
                p[i] = p1 * fact ** (2.0 * gamma / gm1)
                u[i] = u_i
            elif x < xcd:
                rho[i] = rho3
                p[i] = p3
                u[i] = u3
            elif x < xsh:
                rho[i] = rho4
                p[i] = p4
                u[i] = u4
            else:
                rho[i] = rho5
                p[i] = p5
                u[i] = u5
    else:
        for i, x in enumerate(x_arr):
            if x < xsh:
                rho[i] = rho5
                p[i] = p5
                u[i] = -u1
            elif x < xcd:
                rho[i] = rho4
                p[i] = p4
                u[i] = -u4
            elif x < xft:
                rho[i] = rho3
                p[i] = p3
                u[i] = -u3
            elif x < xhd:
                u_i = -2.0 / gp1 * (c1 + (xi - x) / t)
                fact = 1.0 + 0.5 * gm1 * u_i / c1
                rho[i] = rho1 * fact ** (2.0 / gm1)
                p[i] = p1 * fact ** (2.0 * gamma / gm1)
                u[i] = u_i
            else:
                rho[i] = rho1
                p[i] = p1
                u[i] = -u1

    return x_arr, p, rho, u


def solve(left_state, right_state, geometry, t, gamma=1.4, npts=500, dust_frac=0.0):
    pl, rhol, ul = left_state
    pr, rhor, ur = right_state
    xl, xr, xi = geometry

    if xl >= xr:
        raise ValueError("xl must be < xr")
    if not (xl < xi < xr):
        raise ValueError("xi must be between xl and xr")

    region1, region3, region4, region5, w = calculate_regions(pl, rhol, ul, pr, rhor, ur, gamma, dust_frac)
    regions = region_states(pl, pr, region1, region3, region4, region5)

    x_positions = calc_positions(pl, pr, region1, region3, w, xi, t, gamma, dust_frac)
    pos_description = ('Head of Rarefaction', 'Foot of Rarefaction', 'Contact Discontinuity', 'Shock')
    positions = dict(zip(pos_description, x_positions))

    x, p, rho, u = create_arrays(pl, pr, xl, xr, x_positions, region1, region3, region4, region5, npts, gamma, t, xi, dust_frac)
    energy = p / (rho * (gamma - 1.0))
    rho_total = rho / (1.0 - dust_frac)

    val_dict = {'x': x, 'p': p, 'rho': rho, 'u': u, 'energy': energy, 'rho_total': rho_total}
    return positions, regions, val_dict
