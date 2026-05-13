"""
Debug camera world alignment by visualising the saved camera.npy.

Shows:
  - 3-D view: camera path, per-frame view-direction arrows, ground plane (Y=0)
  - Side view (Z forward / Y up): clearest indicator of pitch
  - Pitch & roll time-series

Usage:
    python scripts/debug_camera_plane.py --video Ravens_trimmed/clip.mp4
    python scripts/debug_camera_plane.py --video Ravens_trimmed/clip.mp4 --stride 5
    python scripts/debug_camera_plane.py --video Ravens_trimmed/clip.mp4 --yard_line_align
"""

import argparse
import os
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ── helpers ──────────────────────────────────────────────────────────────────

def load_camera(seq_folder):
    path = os.path.join(seq_folder, 'camera.npy')
    if not os.path.exists(path):
        sys.exit(f'ERROR: camera.npy not found at {path}')
    return np.load(path, allow_pickle=True).item()


def camera_stats(world_cam_R, world_cam_T):
    """
    world_cam_R : (T,3,3)  camera-to-world rotation
    world_cam_T : (T,3)    camera position in world

    Camera conventions
      +Z cam  = forward (look direction)
      +Y cam  = down  (so world-up = camera -Y)
    """
    # Camera position = world_cam_T (it IS the translation for cam→world)
    pos = world_cam_T                               # (T, 3)

    # Forward direction in world = third column of R
    fwd = world_cam_R[:, :, 2]                     # (T, 3)

    # World-up direction as seen by camera = negative camera-Y = -R[:,1,:]
    cam_up = -world_cam_R[:, :, 1]                 # (T, 3)

    # Pitch: angle of forward below the horizontal plane
    # positive pitch = looking down
    pitch_rad = np.arcsin(np.clip(-fwd[:, 1], -1, 1))
    pitch_deg = np.degrees(pitch_rad)

    # Roll: true optical roll — angle between cam_up and the level reference.
    # The level reference is world-Y projected onto the plane perpendicular to
    # the forward direction: it is the "up" a perfectly level camera at the
    # same pitch and heading would have.  This reads 0° for any level camera
    # regardless of pitch, and non-zero only for genuine optical roll.
    world_y = np.array([0., 1., 0.])
    rolls = []
    for i in range(len(world_cam_R)):
        f = fwd[i]
        u = cam_up[i]

        # Level reference: world_y with forward component removed
        ref = world_y - np.dot(world_y, f) * f
        n_ref = np.linalg.norm(ref)
        if n_ref < 1e-6:
            rolls.append(0.)
            continue
        ref /= n_ref

        # Cam-up with forward component removed
        u_perp = u - np.dot(u, f) * f
        n_u = np.linalg.norm(u_perp)
        if n_u < 1e-6:
            rolls.append(0.)
            continue
        u_perp /= n_u

        cos_a = float(np.clip(np.dot(u_perp, ref), -1, 1))
        sign  = np.sign(np.dot(np.cross(u_perp, ref), f))
        rolls.append(np.degrees(np.arccos(cos_a)) * sign)

    return pos, fwd, cam_up, pitch_deg, np.array(rolls)


def ground_patch_verts(cx, cz, size, ground_y=0.0):
    """
    Corners of the ground patch in *plot* coordinates.
    ax3d uses (X, Z, Y) mapping so the patch has constant plot-z = ground_y.
    """
    h = size / 2
    return np.array([
        [cx - h, cz - h, ground_y],
        [cx + h, cz - h, ground_y],
        [cx + h, cz + h, ground_y],
        [cx - h, cz + h, ground_y],
    ])


# ── yard-line debug ───────────────────────────────────────────────────────────

def collect_yard_line_debug(imgfile, calib, R_wc_np, cam_pos, ground_y):
    """
    Re-run yard-line detection on *imgfile* and return everything needed to
    visualise the pitch calculation.

    Returns a dict (or None on detection failure) with keys:
      positions       – 1-D sorted (near→far) perp-axis projections of detected lines
      perp_dir        – (2,) unit vector perpendicular to yard lines in image
      dominant_angle  – dominant line angle in image (radians)
      vp              – vanishing-point coordinate on perp axis (may be None)
      c_perp          – principal-point coordinate on perp axis
      f_perp          – effective focal length along perp direction
      pitch_deg       – estimated pitch (°), positive = looking down
      yard_world_pts  – list of (3,) world-space ground intersections, one per line
      yard_rays       – (N,3) normalised world-space rays to each yard line centre
      vp_ray          – (3,) normalised world-space ray toward vanishing point (or None)
      horizon_ray     – (3,) horizontal world-space ray in same heading as fwd
    """
    from lib.pipeline.field_detection import (
        detect_field_mask, detect_yard_lines, cluster_yard_lines,
        filter_long_yard_lines,
    )

    fx, fy, cx, cy = calib[:4]
    img = cv2.imread(imgfile)
    if img is None:
        print(f'  [yard-line debug] cannot read {imgfile}')
        return None

    field_mask, found = detect_field_mask(img)
    if not found:
        print('  [yard-line debug] field mask not found')
        return None

    lines, dominant_angle = detect_yard_lines(img, field_mask)
    if lines is None or len(lines) == 0:
        print('  [yard-line debug] no lines detected')
        return None

    positions, perp_dir = cluster_yard_lines(lines, dominant_angle)
    if perp_dir is None or len(positions) < 3:
        print(f'  [yard-line debug] too few clusters ({len(positions) if perp_dir is not None else 0})')
        return None

    positions_f = filter_long_yard_lines(lines, dominant_angle, positions, perp_dir,
                                         img.shape[1])
    if len(positions_f) < 3:
        print(f'  [yard-line debug] too few lines after filter ({len(positions_f)})')
        return None

    positions_sorted = np.sort(positions_f)[::-1].copy()   # near → far (descending)
    N = len(positions_sorted)
    ks = np.arange(N, dtype=float)

    A = np.column_stack([ks, -positions_sorted * ks, np.ones(N)])
    result, _, _, _ = np.linalg.lstsq(A, positions_sorted, rcond=None)
    a_coeff, c_coeff, _ = result

    vp = float(a_coeff / c_coeff) if abs(c_coeff) > 1e-9 else None

    px, py = float(perp_dir[0]), float(perp_dir[1])
    along_dir = np.array([np.cos(dominant_angle), np.sin(dominant_angle)])
    c_along = float(cx * along_dir[0] + cy * along_dir[1])

    f_perp = float(np.sqrt((fx * px)**2 + (fy * py)**2))
    if f_perp < 1.0:
        f_perp = float(fy)
    c_perp = float(cx * px + cy * py)

    pitch_rad = float(np.arctan2(c_perp - vp, f_perp)) if vp is not None else None
    pitch_deg = float(np.degrees(pitch_rad)) if pitch_rad is not None else None

    def perp_pos_to_pixel(p):
        """Convert a perp-axis position to an image pixel at the yard-line centre."""
        u = p * px + c_along * along_dir[0]
        v = p * py + c_along * along_dir[1]
        return float(u), float(v)

    def pixel_to_world_ray(u, v):
        d = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        d /= np.linalg.norm(d)
        return R_wc_np @ d

    def ray_ground_intersection(d_world):
        """
        Ray from cam_pos along d_world; intersect with y = ground_y.
        Returns None if the ray is too horizontal (misses ground) or behind.
        """
        if abs(d_world[1]) < 1e-4:
            return None
        t = (ground_y - cam_pos[1]) / d_world[1]
        return cam_pos + t * d_world

    # Per-yard-line world rays and ground intersections
    yard_rays = []
    yard_world_pts = []
    for p in positions_sorted:
        u, v = perp_pos_to_pixel(p)
        d_world = pixel_to_world_ray(u, v)
        yard_rays.append(d_world.copy())
        wp = ray_ground_intersection(d_world)
        yard_world_pts.append(wp)       # may be None

    yard_rays = np.array(yard_rays)

    # Vanishing-point ray
    vp_ray = None
    vp_world = None
    if vp is not None:
        u_vp, v_vp = perp_pos_to_pixel(vp)
        d_vp = pixel_to_world_ray(u_vp, v_vp)
        vp_ray = d_vp
        vp_world = ray_ground_intersection(d_vp)

    # Horizon ray: forward vector with Y zeroed (level in world XZ plane)
    # Use first frame forward direction
    fwd0 = R_wc_np[:, 2]
    h = np.array([fwd0[0], 0.0, fwd0[2]])
    hn = np.linalg.norm(h)
    horizon_ray = h / hn if hn > 1e-6 else np.array([0., 0., 1.])

    print(f'  [yard-line debug] N={N}  positions={positions_sorted.tolist()}')
    print(f'  [yard-line debug] perp_dir={perp_dir}  dominant_angle={np.degrees(dominant_angle):.1f}°')
    print(f'  [yard-line debug] c_perp={c_perp:.1f}  vp={vp}  f_perp={f_perp:.1f}')
    print(f'  [yard-line debug] pitch={pitch_deg}°')

    return {
        'positions':      positions_sorted,
        'perp_dir':       perp_dir,
        'dominant_angle': dominant_angle,
        'vp':             vp,
        'c_perp':         c_perp,
        'f_perp':         f_perp,
        'pitch_deg':      pitch_deg,
        'yard_world_pts': yard_world_pts,
        'yard_rays':      yard_rays,
        'vp_ray':         vp_ray,
        'vp_world':       vp_world,
        'horizon_ray':    horizon_ray,
        'cam_pos':        cam_pos.copy(),
        'N':              N,
    }


def _draw_yard_lines_3d(ax, yd, ground_y, alen):
    """
    Draw detected yard-line positions as horizontal stripes on the ground plane
    in the 3-D axes.  Each stripe is a short segment in the world X direction.
    """
    stripe_half = alen * 2.0
    colors = plt.cm.cool(np.linspace(0.1, 0.9, yd['N']))

    cam = yd['cam_pos']
    for i, wp in enumerate(yd['yard_world_pts']):
        if wp is None:
            continue
        x0, z0 = wp[0] - stripe_half, wp[2]
        x1, z1 = wp[0] + stripe_half, wp[2]
        ax.plot([x0, x1], [z0, z1], [ground_y, ground_y],
                color=colors[i], lw=2.5, alpha=0.85,
                label=f'yard {i} (p={yd["positions"][i]:.0f}px)' if i < 5 else None)
        # Number label at the stripe midpoint on the ground
        ax.text(wp[0], wp[2], ground_y, str(i),
                color='white', fontsize=8, fontweight='bold', ha='center', va='bottom',
                bbox=dict(facecolor=colors[i], edgecolor='none', pad=1.5, alpha=0.85))

    # Ray from camera to each yard-line ground point
    for i, wp in enumerate(yd['yard_world_pts']):
        if wp is None:
            continue
        ax.plot([cam[0], wp[0]], [cam[2], wp[2]], [cam[1], wp[1]],
                color=colors[i], lw=0.8, linestyle=':', alpha=0.6)

    # Vanishing-point world location (if on ground)
    if yd['vp_world'] is not None:
        vw = yd['vp_world']
        ax.scatter(vw[0], vw[2], vw[1], color='magenta', s=120, marker='*',
                   zorder=10, label=f'vp (pitch={yd["pitch_deg"]:.1f}°)')
        ax.plot([cam[0], vw[0]], [cam[2], vw[2]], [cam[1], vw[1]],
                color='magenta', lw=1.2, linestyle='--', alpha=0.8)

    # Horizon ray (dashed white/gray)
    hr = yd['horizon_ray'] * alen * 4
    ax.quiver(cam[0], cam[2], cam[1],
              hr[0], hr[2], hr[1],
              color='silver', lw=1.2, linestyle='--', length=1, normalize=False,
              label='horizon')

    ax.legend(fontsize=6, loc='upper left')


def _draw_yard_lines_side(ax2, yd, ground_y, alen):
    """
    Draw yard-line rays and pitch geometry on the side view (Z vs Y).
    """
    cam = yd['cam_pos']
    colors = plt.cm.cool(np.linspace(0.1, 0.9, yd['N']))

    # Ray to each detected yard line
    for i, (ray, wp) in enumerate(zip(yd['yard_rays'], yd['yard_world_pts'])):
        if wp is not None:
            ax2.annotate('', xy=(wp[2], wp[1]), xytext=(cam[2], cam[1]),
                         arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5))
            ax2.text(wp[2], wp[1], f' {i}', color='white', fontsize=8, fontweight='bold',
                     va='center', zorder=7,
                     bbox=dict(facecolor=colors[i], edgecolor='none', pad=1.5, alpha=0.85))
        else:
            # Ray doesn't hit ground — draw a long directional arrow and label at tip
            end = cam + ray * alen * 3
            ax2.annotate('', xy=(end[2], end[1]), xytext=(cam[2], cam[1]),
                         arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5,
                                         linestyle='dashed'))
            ax2.text(end[2], end[1], f' {i}', color='white', fontsize=8, fontweight='bold',
                     va='center', zorder=7,
                     bbox=dict(facecolor=colors[i], edgecolor='none', pad=1.5, alpha=0.85))

    # Vanishing-point ray
    if yd['vp_ray'] is not None:
        vr = yd['vp_ray']
        if yd['vp_world'] is not None:
            vw = yd['vp_world']
            ax2.annotate('', xy=(vw[2], vw[1]), xytext=(cam[2], cam[1]),
                         arrowprops=dict(arrowstyle='->', color='magenta', lw=2.2))
            ax2.scatter(vw[2], vw[1], color='magenta', s=120, marker='*', zorder=10)
        else:
            end_vp = cam + vr * alen * 5
            ax2.annotate('', xy=(end_vp[2], end_vp[1]), xytext=(cam[2], cam[1]),
                         arrowprops=dict(arrowstyle='->', color='magenta', lw=2.2,
                                         linestyle='dashed'))

    # Horizon ray (level, no pitch)
    hr = yd['horizon_ray'] * alen * 3
    ax2.annotate('', xy=(cam[2] + hr[2], cam[1] + hr[1]), xytext=(cam[2], cam[1]),
                 arrowprops=dict(arrowstyle='->', color='silver', lw=1.5,
                                 linestyle='dashed'))

    # Pitch angle arc between forward vector (from cam_up/forward already drawn)
    # Show the angle between horizon and the vp direction
    if yd['pitch_deg'] is not None and yd['vp_ray'] is not None:
        # Draw a small arc to show the pitch angle
        r_arc = alen * 1.2
        # Horizon angle in ZY plane: arctan2(y_component, z_component) of horizon_ray
        h = yd['horizon_ray']
        ang_horiz = np.degrees(np.arctan2(h[1], h[2]))
        vr = yd['vp_ray']
        ang_vp = np.degrees(np.arctan2(vr[1], vr[2]))

        theta1, theta2 = sorted([ang_horiz, ang_vp])
        arc = mpatches.Arc((cam[2], cam[1]), 2 * r_arc, 2 * r_arc,
                            angle=0, theta1=theta1, theta2=theta2,
                            color='magenta', lw=1.5, linestyle='-')
        ax2.add_patch(arc)

        mid_ang = np.radians((ang_horiz + ang_vp) / 2)
        lx = cam[2] + r_arc * 1.3 * np.cos(mid_ang)
        ly = cam[1] + r_arc * 1.3 * np.sin(mid_ang)
        ax2.text(lx, ly, f'{yd["pitch_deg"]:.1f}°', color='magenta', fontsize=8,
                 ha='center', va='center', fontweight='bold')

    # Legend entries (manual since arrows don't auto-legend)
    legend_handles = [
        mpatches.Patch(color=colors[i], label=f'yard {i} ({yd["positions"][i]:.0f}px)')
        for i in range(yd['N'])
    ]
    legend_handles += [
        mpatches.Patch(color='magenta', label=f'VP (pitch={yd["pitch_deg"]:.1f}°)'),
        mpatches.Patch(color='silver',  label='horizon'),
    ]
    ax2.legend(handles=legend_handles, fontsize=6, loc='upper right')


# ── main plot ─────────────────────────────────────────────────────────────────

def make_plot(cam, output_path, stride=10, imgfiles=None, show_yard_lines=False):
    R = cam['world_cam_R']   # (T,3,3)
    T = cam['world_cam_T']   # (T,3)

    pos, fwd, cam_up, pitch, roll = camera_stats(R, T)

    # Ground Y: stored explicitly when a height shift was applied (field_mode).
    # For other paths the camera sits at Y≈0 and the ground is below; estimate
    # it from the camera height above what appears to be the floor of the scene.
    raw_ground_y = cam.get('world_ground_y', float('nan'))
    if np.isfinite(raw_ground_y):
        ground_y = float(raw_ground_y)
    else:
        # Camera is near Y=0; ground is below.  Use the camera's mean Y minus
        # the vertical distance implied by its pitch and forward extent.
        mean_fwd_dist = float(np.linalg.norm(pos[-1] - pos[0])) if len(pos) > 1 else 1.0
        mean_fwd_dist = max(mean_fwd_dist, 1.0)
        pitch_rad_mean = float(np.radians(np.abs(pitch).mean()))
        estimated_height = mean_fwd_dist * np.tan(max(pitch_rad_mean, np.radians(5.0)))
        ground_y = float(pos[:, 1].mean()) - max(estimated_height, 1.0)
        print(f'  (ground_y not stored — estimated at {ground_y:.2f} m)')

    # Subsampled positions + directions for arrow drawing
    pos_s  = pos[::stride]
    fwd_s  = fwd[::stride]
    up_s   = cam_up[::stride]

    # Arrow length: ~10 % of trajectory extent, clamped to something visible
    extent = max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1]), np.ptp(pos[:, 2]), 0.5)
    alen   = max(extent * 0.12, 0.3)

    # Ground patch size
    patch_size = max(extent * 2.5, 3.0)
    cx = pos[:, 0].mean()
    cz = pos[:, 2].mean()

    # ── yard-line debug data ─────────────────────────────────────────────────
    yd = None
    if show_yard_lines and imgfiles:
        calib = np.array([cam['img_focal'], cam['img_focal'],
                          cam['img_center'][0], cam['img_center'][1]])
        # Use frame 0 rotation as R_wc (SLAM starts at identity, so R[0] ≈ R_wc)
        R_wc_np = R[0].copy()
        cam_pos0 = pos[0].copy()
        # Try a handful of frames and keep first success
        sample_idxs = np.linspace(0, len(imgfiles) - 1, min(10, len(imgfiles)), dtype=int)
        for idx in sample_idxs:
            print(f'  Trying yard-line detection on frame {idx} ({imgfiles[idx]}) …')
            # Use the actual frame's rotation for the ray directions
            R_frame = R[idx].copy()
            pos_frame = pos[idx].copy()
            yd = collect_yard_line_debug(imgfiles[idx], calib, R_frame, pos_frame, ground_y)
            if yd is not None:
                print(f'  Yard-line debug collected from frame {idx}')
                break
        if yd is None:
            print('  WARNING: yard-line detection failed on all sampled frames.')

    # ── print stats ──────────────────────────────────────────────────────────
    print('\n=== Camera World Alignment Stats ===')
    print(f'  Pitch (+ = looking down)  '
          f'mean={pitch.mean():.1f}°  min={pitch.min():.1f}°  max={pitch.max():.1f}°')
    print(f'  Roll  (0 = level)          '
          f'mean={roll.mean():.1f}°  min={roll.min():.1f}°  max={roll.max():.1f}°')
    print(f'  Camera Y (world up)        '
          f'mean={pos[:,1].mean():.3f}  min={pos[:,1].min():.3f}  max={pos[:,1].max():.3f}')
    print(f'  Ground Y                   {ground_y:.3f}  '
          f'(camera ~{pos[:,1].mean() - ground_y:.1f} m above ground)')
    look_down_frac = (fwd[:, 1] < 0).mean() * 100
    print(f'  Forward Y < 0 (looking down): {look_down_frac:.0f}% of frames')

    # ── figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f'Camera World Plane Debug  —  ground Y={ground_y:.2f} m  (green)',
                 fontsize=13, fontweight='bold')

    # ── subplot 1: 3-D view ──────────────────────────────────────────────────
    ax = fig.add_subplot(131, projection='3d')
    ax.set_title('3-D view\nred=fwd  gold=cam-up')

    # Ground plane patch
    verts = ground_patch_verts(cx, cz, patch_size, ground_y=ground_y)
    poly  = Poly3DCollection([verts], alpha=0.18,
                              facecolor='limegreen', edgecolor='darkgreen', linewidth=0.5)
    ax.add_collection3d(poly)

    # Camera path  (ax3d axes: X, Z, Y so ground is horizontal)
    ax.plot(pos[:, 0], pos[:, 2], pos[:, 1],
            color='steelblue', lw=1.5, label='cam path')
    ax.scatter(*pos[0, [0, 2, 1]],  color='lime', s=60, zorder=5, label='start')
    ax.scatter(*pos[-1, [0, 2, 1]], color='red',  s=60, zorder=5, label='end')

    # Forward & up arrows
    for i in range(len(pos_s)):
        p = pos_s[i]
        f = fwd_s[i] * alen
        u = up_s[i]  * alen
        ax.quiver(p[0], p[2], p[1], f[0], f[2], f[1],
                  color='tomato',    length=1, normalize=False, linewidth=0.9)
        ax.quiver(p[0], p[2], p[1], u[0], u[2], u[1],
                  color='goldenrod', length=1, normalize=False, linewidth=0.9)

    # Yard-line overlays in 3-D
    if yd is not None:
        _draw_yard_lines_3d(ax, yd, ground_y, alen)

    # World axis indicators
    al = alen * 1.5
    for d, col, lbl in [([al,0,0],'red','X'), ([0,al,0],'blue','Z'), ([0,0,al],'green','Y')]:
        ax.quiver(0, 0, 0, d[0], d[1], d[2], color=col, lw=2)
        ax.text(d[0], d[1], d[2], f' {lbl}', color=col, fontsize=8)

    ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y (up)')
    ax.legend(fontsize=7, loc='upper left')

    # ── subplot 2: side view Y vs Z ──────────────────────────────────────────
    ax2 = fig.add_subplot(132)
    ax2.set_title('Side view  (Z forward, Y up)\nred=fwd  gold=cam-up'
                  + ('\n[magenta=VP ray  cool=yard lines]' if yd else ''))
    ax2.axhline(ground_y, color='limegreen', lw=2, linestyle='--',
                label=f'ground Y={ground_y:.2f} m')
    ax2.plot(pos[:, 2], pos[:, 1], color='steelblue', lw=1.5, label='cam path')
    ax2.scatter(pos[0, 2],  pos[0, 1],  color='lime', s=50, zorder=5)
    ax2.scatter(pos[-1, 2], pos[-1, 1], color='red',  s=50, zorder=5)

    for i in range(len(pos_s)):
        p = pos_s[i]
        f = fwd_s[i] * alen
        u = up_s[i]  * alen
        ax2.annotate('', xy=(p[2]+f[2], p[1]+f[1]), xytext=(p[2], p[1]),
                     arrowprops=dict(arrowstyle='->', color='tomato', lw=1.2))
        ax2.annotate('', xy=(p[2]+u[2], p[1]+u[1]), xytext=(p[2], p[1]),
                     arrowprops=dict(arrowstyle='->', color='goldenrod', lw=1.2))

    # Yard-line overlays in side view
    if yd is not None:
        _draw_yard_lines_side(ax2, yd, ground_y, alen)

    ax2.set_xlabel('Z (forward)'); ax2.set_ylabel('Y (world up)')
    ax2.set_aspect('equal', adjustable='datalim')
    if yd is None:
        ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── subplot 3: pitch & roll time-series ──────────────────────────────────
    ax3 = fig.add_subplot(133)
    ax3.set_title('Pitch & Roll over time')
    frames = np.arange(len(pitch))
    ax3.plot(frames, pitch, color='tomato',    lw=1.5, label='pitch (° down)')
    ax3.plot(frames, roll,  color='royalblue', lw=1.5, label='roll (° level=0)')
    ax3.axhline(0, color='gray', lw=0.8, linestyle=':')
    ax3.axhline(pitch.mean(), color='tomato',    lw=1, linestyle='--',
                label=f'mean pitch {pitch.mean():.1f}°')
    ax3.axhline(roll.mean(),  color='royalblue', lw=1, linestyle='--',
                label=f'mean roll {roll.mean():.1f}°')

    # If yard-line debug succeeded, draw its estimated pitch as a horizontal line
    if yd is not None and yd['pitch_deg'] is not None:
        ax3.axhline(yd['pitch_deg'], color='magenta', lw=1.5, linestyle='-.',
                    label=f'yard-line pitch {yd["pitch_deg"]:.1f}°')

    ax3.set_xlabel('Frame'); ax3.set_ylabel('Degrees')
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    print(f'\nSaved: {output_path}')


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',  type=str, required=True,
                        help='Video path used with estimate_camera.py')
    parser.add_argument('--stride', type=int, default=10,
                        help='Draw arrows every N frames (default 10)')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--yard_line_align', action='store_true',
                        help='Overlay detected yard lines and vanishing-point pitch '
                             'geometry in the 3-D and side views')
    args = parser.parse_args()

    seq        = os.path.splitext(os.path.basename(args.video))[0]
    seq_folder = os.path.join('results', seq)
    out        = args.output or os.path.join(seq_folder, 'debug_camera_plane.png')

    cam = load_camera(seq_folder)

    imgfiles = None
    if args.yard_line_align:
        from glob import glob
        img_folder = os.path.join(seq_folder, 'images')
        imgfiles = sorted(glob(os.path.join(img_folder, '*.jpg')))
        if not imgfiles:
            print(f'WARNING: no images found in {img_folder}, skipping yard-line overlay')
            imgfiles = None

    make_plot(cam, out, stride=args.stride,
              imgfiles=imgfiles, show_yard_lines=args.yard_line_align)


if __name__ == '__main__':
    main()
