"""
Debug camera world alignment by visualising the saved camera.npy.

Shows:
  - 3-D view: camera path, per-frame view-direction arrows, ground plane (Y=0)
  - Side view (Z forward / Y up): clearest indicator of pitch
  - Pitch & roll time-series

Usage:
    python scripts/debug_camera_plane.py --video Ravens_trimmed/clip.mp4
    python scripts/debug_camera_plane.py --video Ravens_trimmed/clip.mp4 --stride 5
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


# ── main plot ─────────────────────────────────────────────────────────────────

def make_plot(cam, output_path, stride=10):
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

    # World axis indicators
    al = alen * 1.5
    for d, col, lbl in [([al,0,0],'red','X'), ([0,al,0],'blue','Z'), ([0,0,al],'green','Y')]:
        ax.quiver(0, 0, 0, d[0], d[1], d[2], color=col, lw=2)
        ax.text(d[0], d[1], d[2], f' {lbl}', color=col, fontsize=8)

    ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y (up)')
    ax.legend(fontsize=7, loc='upper left')

    # ── subplot 2: side view Y vs Z ──────────────────────────────────────────
    ax2 = fig.add_subplot(132)
    ax2.set_title('Side view  (Z forward, Y up)\nred=fwd  gold=cam-up')
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

    ax2.set_xlabel('Z (forward)'); ax2.set_ylabel('Y (world up)')
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

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
    args = parser.parse_args()

    seq        = os.path.splitext(os.path.basename(args.video))[0]
    seq_folder = os.path.join('results', seq)
    out        = args.output or os.path.join(seq_folder, 'debug_camera_plane.png')

    cam = load_camera(seq_folder)
    make_plot(cam, out, stride=args.stride)


if __name__ == '__main__':
    main()
