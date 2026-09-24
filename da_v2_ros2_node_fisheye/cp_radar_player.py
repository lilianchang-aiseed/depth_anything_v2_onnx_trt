#!/usr/bin/env python3
"""
Animated CollisionPrevention geometry player.

Displays, on a single polar (radar) axis whose radial coordinate is distance
in metres and whose angular coordinate is body-frame direction:

  - the 72-bin obstacle_distance_fused distances as a radar outline + bars
    (only bins that carry a real reading are drawn)
  - the min_dist_to_keep no-go disc
  - the decomposition vectors, all sharing the SAME polar coordinate frame:
        accel_in            (raw stick input direction; length = |accel_in|)
        sp_dir              (commanded direction, post-adapt)
        closest_dir         (toward nearest obstacle)
        normal / tangential (unscaled decomposition, dashed)
        scaled_normal / scaled_tangential (thick)
        vel_comp            (braking contribution)
        constr_accel        (constrained result)
        setpoint_accel      (final output)
  - the pink "toward-obstacle" half-plane
  - a legend placed on the hemisphere OPPOSITE the pink half-plane

Playback over a timestamp range with keyboard control:
    space  : play / pause
    right  : step forward one frame (when paused)
    left   : step back one frame (when paused)
    up/down: speed up / slow down
    q      : quit

The current timestamp and branch are shown in the title.

Usage:
    python cp_radar_player.py flight.ulg
    python cp_radar_player.py flight.ulg --t0 20 --t1 40
    python cp_radar_player.py flight.ulg --t0 20 --t1 40 --fps 10
    python cp_radar_player.py flight.ulg --branch main   # only MAIN frames
    python cp_radar_player.py flight.ulg --save-gif out.gif  # headless export
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

try:
    from pyulog import ULog
except ImportError:
    sys.exit("pyulog required: pip install pyulog")


BRANCH_LABELS = {0: 'idle', 1: 'MAIN', 2: 'BAIL', 3: 'PASS'}
BIN_SIZE_DEG = 5
N_BINS = 72
DIST_INVALID_CM = 65530


# ---------------------------------------------------------------------------
# Loading (state + trace + obstacle map, merged on shared timestamp)
# ---------------------------------------------------------------------------

def _get_ds(ulog, name):
    for d in ulog.data_list:
        if d.name == name:
            return d
    return None


def _shared_origin(ulog):
    origins = []
    for name in ('cp_state', 'cp_trace', 'obstacle_distance_fused'):
        d = _get_ds(ulog, name)
        if d is not None and len(d.data['timestamp']) > 0:
            origins.append(int(np.min(d.data['timestamp'])))
    return min(origins) if origins else 0


def _load_state(ulog, t0):
    d = _get_ds(ulog, 'cp_state')
    if d is None:
        return None
    raw = pd.DataFrame(d.data)
    df = pd.DataFrame()
    df['t_s']                    = (raw['timestamp'].values - t0) * 1e-6
    df['branch']                 = raw['data[0]'].astype(int).values
    df['closest_dist']           = raw['data[2]'].values
    df['min_dist_to_keep']       = raw['data[4]'].values
    df['feasible']               = raw['data[6]'].astype(int).values
    df['vehicle_yaw_deg']        = raw['data[7]'].values
    df['accel_in_x']             = raw['data[8]'].values
    df['accel_in_y']             = raw['data[9]'].values
    df['accel_in_norm']          = raw['data[10]'].values
    df['setpoint_accel_x']       = raw['data[11]'].values
    df['setpoint_accel_y']       = raw['data[12]'].values
    df['setpoint_accel_norm']    = raw['data[13]'].values
    df['constr_accel_x']         = raw['data[14]'].values
    df['constr_accel_y']         = raw['data[15]'].values
    df['vel_comp_mag']           = raw['data[16]'].values
    df['setpoint_dir_x']         = raw['data[26]'].values
    df['setpoint_dir_y']         = raw['data[27]'].values
    df['closest_dir_x']          = raw['data[30]'].values
    df['closest_dir_y']          = raw['data[31]'].values
    df['vel_comp_dir_x']         = raw['data[33]'].values
    df['vel_comp_dir_y']         = raw['data[34]'].values
    return df.sort_values('t_s').reset_index(drop=True)


def _load_trace(ulog, t0):
    d = _get_ds(ulog, 'cp_trace')
    if d is None:
        return None
    raw = pd.DataFrame(d.data)
    df = pd.DataFrame()
    df['t_s']                       = (raw['timestamp'].values - t0) * 1e-6
    df['normal_component_x']        = raw['data[5]'].values
    df['normal_component_y']        = raw['data[6]'].values
    df['tangential_component_x']    = raw['data[7]'].values
    df['tangential_component_y']    = raw['data[8]'].values
    df['normal_scale']              = raw['data[9]'].values
    df['tangential_scale']          = raw['data[10]'].values
    df['constrain_toward_obstacle'] = raw['data[13]'].astype(int).values
    return df.sort_values('t_s').reset_index(drop=True)


def _load_obstacle_map(ulog, t0):
    d = _get_ds(ulog, 'obstacle_distance_fused')
    if d is None:
        return None
    raw = pd.DataFrame(d.data)
    dist_cols = sorted([c for c in raw.columns if c.startswith('distances[')],
                       key=lambda s: int(s[len('distances['):-1]))
    df = pd.DataFrame()
    df['t_s']          = (raw['timestamp'].values - t0) * 1e-6
    df['min_distance'] = raw['min_distance'].values
    df['max_distance'] = raw['max_distance'].values
    df['angle_offset'] = (raw['angle_offset'].values
                          if 'angle_offset' in raw.columns else 0.0)
    df['distances_cm'] = list(raw[dist_cols].values.astype(int))
    return df.sort_values('t_s').reset_index(drop=True)


def load_merged(ulog):
    t0 = _shared_origin(ulog)
    state = _load_state(ulog, t0)
    if state is None:
        sys.exit('cp_state not found in log')
    trace = _load_trace(ulog, t0)
    if trace is not None:
        merged = pd.merge_asof(state, trace, on='t_s',
                               direction='nearest', tolerance=0.005)
    else:
        merged = state
        for c in ('normal_component_x', 'normal_component_y',
                  'tangential_component_x', 'tangential_component_y',
                  'normal_scale', 'tangential_scale',
                  'constrain_toward_obstacle'):
            merged[c] = np.nan

    obs = _load_obstacle_map(ulog, t0)
    if obs is not None:
        # backward: only maps the firmware already had
        merged = pd.merge_asof(merged.sort_values('t_s'),
                               obs.sort_values('t_s'),
                               on='t_s', direction='backward',
                               tolerance=0.200)
    else:
        merged['distances_cm'] = None
        merged['min_distance'] = np.nan
        merged['angle_offset'] = 0.0
    return merged


# ---------------------------------------------------------------------------
# Coordinate helpers
#   Body frame convention for the polar plot:
#     theta = 0   -> body forward (drone nose)
#     theta grows counter-clockwise in math sense, but we set the polar axis
#     so that forward is up and +body-right is to the right (clockwise).
# ---------------------------------------------------------------------------

def world_to_body(vx, vy, yaw_rad):
    """World NED xy -> body (right, forward)."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    fwd   = c * vx + s * vy
    right = -s * vx + c * vy
    return right, fwd


def body_xy_to_polar(right, fwd):
    """Body (right, forward) -> (theta_rad, r) for the polar axis.

    With set_theta_zero_location('N') and set_theta_direction(-1):
      theta measured clockwise from north (forward).
      A body vector (right, fwd) has:
        r     = hypot(right, fwd)
        theta = atan2(right, fwd)   # 0 = forward, +90deg = right
    """
    r = np.hypot(right, fwd)
    theta = np.arctan2(right, fwd)
    return theta, r


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class RadarPlayer:
    def __init__(self, merged, t0=None, t1=None, branch_filter='all',
                 fps=8):
        df = merged
        if t0 is not None:
            df = df[df['t_s'] >= t0]
        if t1 is not None:
            df = df[df['t_s'] <= t1]
        if branch_filter != 'all':
            bmap = {'main': 1, 'idle': 0, 'bail': 2, 'pass': 3}
            df = df[df['branch'] == bmap[branch_filter]]
        self.df = df.reset_index(drop=True)
        if len(self.df) == 0:
            sys.exit('no frames in requested range / branch filter')

        self.n = len(self.df)
        self.idx = 0
        self.playing = True
        self.fps = fps
        self.base_interval = 1000.0 / fps

        # Two-panel figure: polar radar (left) + vector rows (right)
        self.fig = plt.figure(figsize=(18, 10))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.0, 0.75],
                                   wspace=0.08)
        self.ax_radar = self.fig.add_subplot(gs[0], projection='polar')
        self.ax_radar.set_theta_zero_location('N')
        self.ax_radar.set_theta_direction(-1)
        self.ax_vec = self.fig.add_subplot(gs[1])

        self.fig.text(0.5, 0.02,
                      'space play/pause   ← → step (paused)   '
                      '↑ ↓ speed   q quit',
                      ha='center', va='bottom', fontsize=9, color='dimgray')
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    # ---- keyboard ----
    def _on_key(self, event):
        if event.key == ' ':
            self.playing = not self.playing
        elif event.key == 'right' and not self.playing:
            self.idx = (self.idx + 1) % self.n
            self._draw()
            self.fig.canvas.draw_idle()
        elif event.key == 'left' and not self.playing:
            self.idx = (self.idx - 1) % self.n
            self._draw()
            self.fig.canvas.draw_idle()
        elif event.key == 'up':
            self.fps = min(60, self.fps * 1.5)
            if getattr(self, 'anim', None) and self.anim.event_source:
                self.anim.event_source.interval = 1000.0 / self.fps
        elif event.key == 'down':
            self.fps = max(1, self.fps / 1.5)
            if getattr(self, 'anim', None) and self.anim.event_source:
                self.anim.event_source.interval = 1000.0 / self.fps
        elif event.key == 'q':
            plt.close(self.fig)

    # ---- animation step ----
    def _update(self, frame):
        if self.playing:
            self.idx = (self.idx + 1) % self.n
        self._draw()
        return []

    # ---- the actual drawing of one frame ----
    def _draw(self):
        row = self.df.iloc[self.idx]
        yaw = np.radians(row['vehicle_yaw_deg'])

        # ============================================================
        # Distinct color palette — chosen to be maximally separable
        # ============================================================
        C_ACCEL_IN       = '#0066FF'   # bright blue
        C_SP_DIR         = '#FFD700'   # gold
        C_NORMAL_UN      = '#FF69B4'   # hot pink (dashed)
        C_TANGENTIAL_UN  = '#00CED1'   # dark turquoise (dashed)
        C_SCALED_NORMAL  = '#8B0000'   # dark red / maroon
        C_SCALED_TANG    = '#008080'   # teal
        C_VELCOMP        = '#FF6600'   # deep orange
        C_CONSTR         = '#9400D3'   # dark violet
        C_SETPOINT       = '#00CC00'   # bright green

        # ============================================================
        # Prepare shared data
        # ============================================================
        a_in = float(row['accel_in_norm'])
        cdist = float(row['closest_dist'])
        mdk = float(row['min_dist_to_keep'])

        # Compute deviation angle between accel_in and setpoint_accel
        ai_x, ai_y = row['accel_in_x'], row['accel_in_y']
        sp_x, sp_y = row['setpoint_accel_x'], row['setpoint_accel_y']
        ai_norm = np.hypot(ai_x, ai_y)
        sp_norm = np.hypot(sp_x, sp_y)
        if ai_norm > 0.02 and sp_norm > 0.02:
            ai_deg = np.degrees(np.arctan2(ai_y, ai_x))
            sp_deg = np.degrees(np.arctan2(sp_y, sp_x))
            deviation_deg = ((sp_deg - ai_deg + 180) % 360) - 180
        else:
            deviation_deg = 0.0

        # rmax for the radar
        vec_mags = [a_in, float(row['setpoint_accel_norm']),
                    np.hypot(row['constr_accel_x'], row['constr_accel_y']),
                    abs(float(row['vel_comp_mag']))]
        near_span = max([mdk, cdist if 0 < cdist < 50 else 0] + vec_mags)
        rmax = max(3.0, near_span * 1.25)
        rmax = 8.0

        ns = float(row.get('normal_scale', np.nan))
        ts = float(row.get('tangential_scale', np.nan))
        have_trace = np.isfinite(ns) and np.isfinite(ts)

        # Body-frame vector components for the right subplot
        def to_body_vec(wx, wy):
            r, f = world_to_body(wx, wy, yaw)
            return r, f  # (right, forward)

        # ============================================================
        # LEFT PANEL: polar radar
        # ============================================================
        ax = self.ax_radar
        ax.clear()
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        # ---- obstacle map: radar polygon ----
        dists_cm = row.get('distances_cm', None)
        if dists_cm is not None and not isinstance(dists_cm, float):
            d = np.asarray(dists_cm, dtype=float)
            valid = d < DIST_INVALID_CM
            angle_offset = float(row.get('angle_offset', 0.0) or 0.0)
            theta = np.radians(np.arange(N_BINS) * BIN_SIZE_DEG + angle_offset)
            radius = np.where(valid, d * 0.01, np.nan)
            radius = np.clip(radius, 0.0, rmax)
            theta_closed = np.append(theta, theta[0])
            radius_closed = np.append(radius, radius[0])
            ax.plot(theta_closed, radius_closed,
                    color='steelblue', linewidth=2.0, zorder=2)
            ax.fill(theta_closed, radius_closed,
                    color='steelblue', alpha=0.25, zorder=1)
            ax.scatter(theta[valid], radius[valid],
                       s=18, color='steelblue', edgecolors='white',
                       linewidth=0.5, zorder=3)

        # ---- no-go disc ----
        theta_full = np.linspace(0, 2 * np.pi, 120)
        ax.fill(theta_full, np.full_like(theta_full, mdk),
                color='red', alpha=0.12, zorder=1)
        ax.plot(theta_full, np.full_like(theta_full, mdk),
                color='red', ls='--', lw=0.8, alpha=0.6, zorder=2)

        # ---- pink toward-obstacle half-plane ----
        cd_body = np.array(world_to_body(row['closest_dir_x'],
                                         row['closest_dir_y'], yaw))
        cd_theta = np.arctan2(cd_body[0], cd_body[1])
        half = np.linspace(cd_theta - np.pi / 2, cd_theta + np.pi / 2, 60)
        ax.fill_between(half, 0, rmax, color='magenta', alpha=0.10, zorder=0)

        # ---- closest_dir as a RED DOT (not an arrow) ----
        cd_r_display = min(cdist, rmax * 0.95)
        ax.plot([cd_theta], [cd_r_display], marker='o', color='red',
                markersize=10, markeredgecolor='black', markeredgewidth=1.5,
                zorder=10)

        # ---- direction vectors on the radar ----
        udir_len = 0.72 * rmax

        def draw_vec(wx, wy, color, lw, ls='-', alpha=1.0,
                     as_unit=False, unit_len=udir_len):
            r, f = world_to_body(wx, wy, yaw)
            if as_unit:
                nrm = np.hypot(r, f)
                if nrm < 1e-9:
                    return None
                r, f = r / nrm * unit_len, f / nrm * unit_len
            th, rr = body_xy_to_polar(r, f)
            if rr < 1e-6:
                return None
            ax.plot([0, th], [0, rr], color=color, lw=lw, ls=ls,
                    alpha=alpha, zorder=6, solid_capstyle='round')
            ax.plot([th], [rr], marker=(3, 0, -np.degrees(th)),
                    markersize=7 + lw, color=color, alpha=alpha, zorder=7)
            return (th, rr)

        # Draw vectors on radar
        draw_vec(ai_x, ai_y, C_ACCEL_IN, 2.6)
        draw_vec(row['setpoint_dir_x'], row['setpoint_dir_y'],
                 C_SP_DIR, 2.0, as_unit=True)

        if have_trace:
            nc = np.array([row['normal_component_x'], row['normal_component_y']])
            tc = np.array([row['tangential_component_x'], row['tangential_component_y']])
            draw_vec(nc[0]*a_in, nc[1]*a_in, C_NORMAL_UN, 1.3, ls='--', alpha=0.7)
            draw_vec(tc[0]*a_in, tc[1]*a_in, C_TANGENTIAL_UN, 1.3, ls='--', alpha=0.7)
            draw_vec(nc[0]*ns*a_in, nc[1]*ns*a_in, C_SCALED_NORMAL, 2.6)
            draw_vec(tc[0]*ts*a_in, tc[1]*ts*a_in, C_SCALED_TANG, 2.6)

        vc_mag = float(row['vel_comp_mag'])
        draw_vec(vc_mag * row['vel_comp_dir_x'], vc_mag * row['vel_comp_dir_y'],
                 C_VELCOMP, 2.2)
        draw_vec(row['constr_accel_x'], row['constr_accel_y'],
                 C_CONSTR, 2.2, ls=':')
        draw_vec(sp_x, sp_y, C_SETPOINT, 2.8)

        # ---- body-axis direction triangles ----
        def draw_direction_marker(theta_deg, color, size=18):
            r = rmax * 1.08
            t = np.deg2rad(theta_deg)
            ax.scatter(t, r, marker=(3, 0, -theta_deg),
                       s=size**2, color=color, edgecolors='black',
                       linewidths=1.0, zorder=20, clip_on=False)

        draw_direction_marker(0,   'red',       size=22)
        draw_direction_marker(90,  'limegreen', size=15)
        draw_direction_marker(180, 'royalblue', size=15)
        draw_direction_marker(270, 'darkorange', size=15)

        # ---- radar axis setup ----
        ax.set_rmax(rmax)
        ax.set_rticks(np.round(np.linspace(0, rmax, 4), 1))
        ax.set_rlabel_position(np.degrees(cd_theta) + 180)
        ax.grid(True, alpha=0.3)
        ax.set_thetagrids(range(0, 360, 30),
                          labels=[f'{a}°' for a in range(0, 360, 30)],
                          fontsize=8)

        # ---- radar legend on opposite hemisphere ----
        away_theta = cd_theta + np.pi
        ax_dx = np.sin(away_theta)
        ax_dy = np.cos(away_theta)
        loc, anchor = self._legend_anchor(ax_dx, ax_dy)

        radar_legend = [
            ('obstacle map',     'steelblue', 2, '-'),
            ('min_dist_to_keep', 'red',       1, '--'),
            ('toward half-plane','magenta',    6, '-'),
        ]
        handles = []
        for label, color, lw, ls in radar_legend:
            handles.append(Line2D([0, 1], [0, 0], color=color, lw=lw, ls=ls,
                                  label=label))
        # closest dot
        handles.append(Line2D([0], [0], marker='o', linestyle='None',
                              markersize=8, markerfacecolor='red',
                              markeredgecolor='black', label='closest bin'))
        # body triangles
        for tlabel, tcolor in (('Forward', 'red'), ('Right', 'limegreen'),
                               ('Back', 'royalblue'), ('Left', 'darkorange')):
            handles.append(Line2D([0], [0], marker='^', linestyle='None',
                                  markersize=8, markerfacecolor=tcolor,
                                  markeredgecolor='black', label=tlabel))
        ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor,
                  fontsize=7, framealpha=0.92, facecolor='white',
                  edgecolor='gray', borderpad=0.4, labelspacing=0.3,
                  handlelength=2.2)

        # ============================================================
        # RIGHT PANEL: vector rows with x,y component numbers
        # ============================================================
        axv = self.ax_vec
        axv.clear()

        # Vector specs: (label, world_x, world_y, color, lw, ls)
        vec_specs = [
            ('accel_in',          ai_x, ai_y, C_ACCEL_IN, 2.6, '-'),
            ('sp_dir',            row['setpoint_dir_x'], row['setpoint_dir_y'],
                                  C_SP_DIR, 2.0, '-'),
        ]
        if have_trace:
            nc = np.array([row['normal_component_x'], row['normal_component_y']])
            tc = np.array([row['tangential_component_x'], row['tangential_component_y']])
            vec_specs += [
                ('normal (unsc)',     nc[0]*a_in, nc[1]*a_in, C_NORMAL_UN, 1.5, '--'),
                ('tangential (unsc)', tc[0]*a_in, tc[1]*a_in, C_TANGENTIAL_UN, 1.5, '--'),
                (f'scaled_normal ×{ns:.2f}',  nc[0]*ns*a_in, nc[1]*ns*a_in,
                                      C_SCALED_NORMAL, 2.6, '-'),
                (f'scaled_tang ×{ts:.2f}',    tc[0]*ts*a_in, tc[1]*ts*a_in,
                                      C_SCALED_TANG, 2.6, '-'),
            ]
        else:
            vec_specs += [
                ('normal (unsc)',     0, 0, C_NORMAL_UN, 1.5, '--'),
                ('tangential (unsc)', 0, 0, C_TANGENTIAL_UN, 1.5, '--'),
                ('scaled_normal',     0, 0, C_SCALED_NORMAL, 2.6, '-'),
                ('scaled_tang',       0, 0, C_SCALED_TANG, 2.6, '-'),
            ]

        vec_specs += [
            ('vel_comp',
             vc_mag * row['vel_comp_dir_x'], vc_mag * row['vel_comp_dir_y'],
             C_VELCOMP, 2.2, '-'),
            ('constr_accel',
             row['constr_accel_x'], row['constr_accel_y'],
             C_CONSTR, 2.2, ':'),
            ('setpoint_accel', sp_x, sp_y, C_SETPOINT, 2.8, '-'),
        ]

        n_rows = len(vec_specs)
        row_y = np.arange(n_rows).astype(float)

        # Compute body-frame components and magnitudes
        body_vecs = []
        all_mags = []
        for _, wx, wy, *_ in vec_specs:
            bx, by = to_body_vec(wx, wy)  # (right, forward)
            body_vecs.append((bx, by))
            all_mags.append(np.hypot(bx, by))

        # Auto-scale: longest arrow fills ~40% of available width
        max_mag = max(m for m in all_mags if m > 1e-6) if any(m > 1e-6 for m in all_mags) else 1.0
        arrow_scale = max_mag / 0.40

        # Row background bands
        for j in range(n_rows):
            _, _, _, color, _, _ = vec_specs[j]
            axv.axhspan(j - 0.45, j + 0.45, color=color, alpha=0.06, zorder=0)

        # Draw arrows (body frame: x=right, y=forward; plot x=right, y=row)
        for j, (label, wx, wy, color, lw, ls) in enumerate(vec_specs):
            bx, by = body_vecs[j]
            mag = all_mags[j]

            # Arrow from row center, scaled
            u = bx / arrow_scale
            v = by / arrow_scale

            axv.annotate('',
                         xy=(u, row_y[j] + v),
                         xytext=(0, row_y[j]),
                         arrowprops=dict(arrowstyle='->', color=color,
                                         lw=lw, ls=ls))

            # Component text (body right, body forward, magnitude)
            text_x = max(0.52, abs(u) + 0.08)
            axv.text(text_x, row_y[j] + 0.15,
                     f'R={bx:+.3f}  F={by:+.3f}  |.| = {mag:.3f}',
                     fontsize=7.5, color=color, va='center', ha='left',
                     fontfamily='monospace')

        # Y-axis labels
        axv.set_yticks(row_y)
        axv.set_yticklabels([s[0] for s in vec_specs], fontsize=8)

        axv.set_xlim(-0.55, 0.95)
        axv.set_ylim(-0.6, n_rows - 0.4)
        axv.axvline(0, color='k', lw=0.3, alpha=0.5)
        axv.set_xlabel('body frame (R=right, F=fwd)', fontsize=9)
        axv.grid(True, alpha=0.2, ls=':')
        axv.set_title('Vector decomposition\n(body-frame arrows + components)',
                      fontsize=10, pad=8)

        # ============================================================
        # SUPTITLE: timestamp, branch, deviation angle
        # ============================================================
        branch = BRANCH_LABELS.get(int(row['branch']), '?')
        play_state = 'PLAYING' if self.playing else 'PAUSED'

        # Color the deviation angle text: green if small, red if large
        dev_abs = abs(deviation_deg)
        if dev_abs < 10:
            dev_color = 'green'
        elif dev_abs < 45:
            dev_color = 'darkorange'
        else:
            dev_color = 'red'

        self.fig.suptitle(
            f't = {row["t_s"]:.3f}s   frame {self.idx+1}/{self.n}   '
            f'branch = {branch}   [{play_state}, {self.fps:.0f} fps]     '
            f'yaw = {row["vehicle_yaw_deg"]:+.1f}°   '
            f'closest = {cdist:.2f}m   '
            f'|accel_in| = {a_in:.2f}   '
            f'|setpoint| = {row["setpoint_accel_norm"]:.2f}',
            fontsize=10, y=0.98)

        # Deviation angle as a separate prominent text line
        self.fig.texts = [t for t in self.fig.texts
                          if not getattr(t, '_is_deviation', False)
                          and not getattr(t, '_is_hint', False)]
        dev_text = self.fig.text(
            0.5, 0.945,
            f'CP deviation (accel_in → setpoint_accel): {deviation_deg:+.1f}°',
            ha='center', fontsize=11, fontweight='bold', color=dev_color)
        dev_text._is_deviation = True
        hint_text = self.fig.text(
            0.5, 0.02,
            'space play/pause   ← → step (paused)   '
            '↑ ↓ speed   q quit',
            ha='center', va='bottom', fontsize=9, color='dimgray')
        hint_text._is_hint = True

    @staticmethod
    def _legend_anchor(dx, dy):
        """Map an 'away' direction (dx=right, dy=up) to a matplotlib legend
        loc + bbox_to_anchor placed in that corner, in axes coordinates."""
        # Choose corner by sign of dominant components
        right = dx > 0
        up = dy > 0
        if up and right:
            return 'upper right', (1.28, 1.12)
        if up and not right:
            return 'upper left', (-0.28, 1.12)
        if not up and right:
            return 'lower right', (1.28, -0.02)
        return 'lower left', (-0.28, -0.02)

    def run(self, save_gif=None):
        if save_gif:
            from matplotlib.animation import PillowWriter
            self.playing = True
            n_export = min(self.n, 600)
            self.idx = -1
            anim = FuncAnimation(
                self.fig, self._update,
                frames=n_export, interval=self.base_interval,
                blit=False, cache_frame_data=False)
            anim.save(save_gif, writer=PillowWriter(fps=int(max(1, self.fps))))
            print(f'saved {save_gif} ({n_export} frames)')
        else:
            self.anim = FuncAnimation(
                self.fig, self._update,
                interval=self.base_interval,
                blit=False, cache_frame_data=False)
            plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ulog', type=Path)
    ap.add_argument('--t0', type=float, default=None)
    ap.add_argument('--t1', type=float, default=None)
    ap.add_argument('--branch', choices=['all', 'main', 'idle', 'bail', 'pass'],
                    default='all')
    ap.add_argument('--fps', type=float, default=8.0)
    ap.add_argument('--save-gif', type=Path, default=None,
                    help='export to animated GIF (headless) instead of showing')
    args = ap.parse_args()

    if not args.ulog.exists():
        sys.exit(f'file not found: {args.ulog}')

    if args.save_gif:
        matplotlib.use('Agg')

    print(f'reading {args.ulog} ...')
    ulog = ULog(str(args.ulog), message_name_filter_list=[
        'cp_state', 'cp_trace', 'obstacle_distance_fused'])
    merged = load_merged(ulog)

    player = RadarPlayer(merged, t0=args.t0, t1=args.t1,
                         branch_filter=args.branch, fps=args.fps)
    print(f'{player.n} frames  '
          f't ∈ [{player.df["t_s"].min():.2f}, {player.df["t_s"].max():.2f}] s')
    player.run(save_gif=args.save_gif)


if __name__ == '__main__':
    main()
