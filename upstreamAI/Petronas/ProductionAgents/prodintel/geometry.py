"""Proxy wellbore trajectories.

These are geometric proxies built from surface location, landing TVD, heel and
toe: vertical to kick-off, a constant-radius build to horizontal, then the
lateral. They are not measured directional surveys and are labelled as such
everywhere they are shown.
"""
from __future__ import annotations

import numpy as np

BUILD_RADIUS_FT = 650.0


def proxy_trajectory(surf_x, surf_y, heel_x, heel_y, toe_x, toe_y, tvd, n_lat=25):
    """Return arrays (x, y, z) with z as TVD (positive down), in local field feet."""
    kop = tvd - BUILD_RADIUS_FT
    # Direction from surface towards heel, then the lateral direction
    lat_vec = np.array([toe_x - heel_x, toe_y - heel_y], dtype=float)
    lat_len = np.linalg.norm(lat_vec) or 1.0
    u = lat_vec / lat_len
    # Point where the build starts, set back from the heel by the build radius
    build_start = np.array([heel_x, heel_y]) - u * BUILD_RADIUS_FT
    xs = [surf_x, surf_x, build_start[0]]
    ys = [surf_y, surf_y, build_start[1]]
    zs = [0.0, kop * 0.6, kop]
    for a in np.linspace(0, np.pi / 2, 10)[1:]:
        horiz = BUILD_RADIUS_FT * np.sin(a)
        xs.append(build_start[0] + u[0] * horiz)
        ys.append(build_start[1] + u[1] * horiz)
        zs.append(kop + BUILD_RADIUS_FT * (1 - np.cos(a)))
    for f in np.linspace(0, 1, n_lat)[1:]:
        xs.append(heel_x + (toe_x - heel_x) * f)
        ys.append(heel_y + (toe_y - heel_y) * f)
        zs.append(tvd + 15 * np.sin(f * 6))   # gentle undulation in the lateral
    return np.array(xs), np.array(ys), np.array(zs)


def well_trajectory(row):
    return proxy_trajectory(row.surf_x, row.surf_y, row.heel_x, row.heel_y,
                            row.toe_x, row.toe_y, row.tvd)
