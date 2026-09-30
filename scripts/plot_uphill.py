#!/usr/bin/env python3
"""
Plotting script for Ising Model Creutz-Thermal Uphill Current Phase Diagram.
Generates:
1. Publication-quality replica of the target phase diagram with shaded uphill currents zone.
2. 2D Heatmap of current flux J(beta, m_+).
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata

BETA_C = 0.5 * np.log(1.0 + np.sqrt(2.0))  # ~0.44068679


def m_onsager(beta):
    """Exact Onsager-Yang spontaneous magnetization for 2D square lattice Ising model."""
    beta = np.asarray(beta, dtype=float)
    m = np.zeros_like(beta)
    mask = beta > BETA_C
    b = beta[mask]
    term = np.sinh(2.0 * b) ** (-4)
    valid = term < 1.0
    m_vals = np.zeros_like(b)
    m_vals[valid] = (1.0 - term[valid]) ** 0.125
    m[mask] = m_vals
    return m


def setup_matplotlib_style():
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20,
        'lines.linewidth': 2.0,
        'font.family': 'sans-serif',
        'mathtext.fontset': 'dejavusans',
    })


def plot_phase_diagram(df, output_path):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Analytical curves
    beta_vals = np.linspace(BETA_C, 1.25, 1000)
    m_vals = m_onsager(beta_vals)

    # 1. Vertical critical beta dashed line
    ax.axvline(
        x=BETA_C,
        color='black',
        linestyle='--',
        linewidth=1.8,
        label=r'$\beta_c = 0.440686$'
    )

    # 2. Solid curve m_beta
    ax.plot(
        beta_vals,
        m_vals,
        color='black',
        linewidth=2.4,
        label=r'$m_\beta = [1 - \sinh^{-4}(2\beta)]^{1/8}$'
    )

    # 3. Step function at beta_c (from 0 to 0 on beta <= beta_c)
    beta_sub = np.linspace(0.2, BETA_C, 200)
    ax.plot(beta_sub, np.zeros_like(beta_sub), color='black', linewidth=2.4)
    ax.plot([BETA_C, BETA_C], [0.0, 0.0], color='black', linewidth=2.4)

    # 4. Highlight the Uphill Currents zone if data exists
    if df is not None and len(df) > 0 and 'j_flux' in df.columns:
        # Check if dataset includes phase/stability information
        has_stability = 'is_metastable' in df.columns or 'phase' in df.columns

        if has_stability:
            if 'is_metastable' in df.columns:
                target_mask = (df['j_flux'] > 0.0) & (df['is_metastable'] == 1)
            else:
                target_mask = (df['j_flux'] > 0.0) & (df['phase'] == 'metastable')
            uphill_pts = df[target_mask]
            print(f"Plotting with stability filter: {len(uphill_pts)} metastable uphill points (out of {(df['j_flux'] > 0.0).sum()} total positive current points).")
        else:
            target_mask = df['j_flux'] > 0.0
            uphill_pts = df[target_mask]

        if len(uphill_pts) >= 4:
            try:
                # Interpolate on fine grid
                b_grid = np.linspace(0.40, 1.22, 300)
                m_grid = np.linspace(0.0, 1.0, 300)
                B, M = np.meshgrid(b_grid, m_grid)

                # Interpolate indicator of valid metastable uphill state
                valid_interp = griddata(
                    (df['beta'], df['m_plus']),
                    target_mask.astype(float),
                    (B, M),
                    method='linear',
                    fill_value=0.0
                )

                # Mask values above Onsager curve and below beta_c
                M_theo = m_onsager(B)
                valid_uphill = (valid_interp > 0.5) & (M <= M_theo + 0.005) & (B >= BETA_C)

                # Filled region for uphill currents (metastable phase)
                ax.contourf(
                    B, M, valid_uphill.astype(float),
                    levels=[0.5, 1.5],
                    colors=['#ff9999'],
                    alpha=0.65
                )

                # Lower dashed boundary of the uphill metastable zone
                ax.contour(
                    B, M, valid_interp,
                    levels=[0.5],
                    colors=['black'],
                    linestyles=['--'],
                    linewidths=[1.8]
                )
            except Exception as e:
                print(f"Warning during contour generation: {e}")
                ax.scatter(
                    uphill_pts['beta'],
                    uphill_pts['m_plus'],
                    c='#ff6666',
                    s=25,
                    alpha=0.7,
                    label='Metastable uphill points'
                )
        elif len(uphill_pts) > 0:
            ax.scatter(
                uphill_pts['beta'],
                uphill_pts['m_plus'],
                c='#ff6666',
                s=25,
                alpha=0.7,
                label='Metastable uphill points'
            )

    # Axes configuration exactly matching the target diagram
    ax.set_xlim(0.2, 1.2)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$m_+$')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    ax.grid(True, linestyle=':', color='gray', alpha=0.5)

    ax.legend(
        loc='lower right',
        frameon=True,
        framealpha=0.9,
        edgecolor='lightgray',
        fancybox=True,
        borderpad=0.8
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    pdf_path = os.path.splitext(output_path)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close()
    print(f"Phase diagram saved to: {output_path} and {pdf_path}")


def plot_heatmap(df, output_path):
    if df is None or len(df) < 4 or 'j_flux' not in df.columns:
        print("Not enough data points for 2D heatmap plot.")
        return

    if df['beta'].nunique() < 2 or df['m_plus'].nunique() < 2:
        print("Not enough 2D spread for heatmap (need variations in both beta and m_plus).")
        return

    fig, ax = plt.subplots(figsize=(9, 8))

    b_min, b_max = df['beta'].min(), df['beta'].max()
    m_min, m_max = df['m_plus'].min(), df['m_plus'].max()

    b_grid = np.linspace(b_min, b_max, 250)
    m_grid = np.linspace(m_min, m_max, 250)
    B, M = np.meshgrid(b_grid, m_grid)

    J_interp = griddata(
        (df['beta'], df['m_plus']),
        df['j_flux'],
        (B, M),
        method='linear'
    )

    # Symmetrical divergent normalization around zero
    vmax = max(abs(np.nanmin(J_interp)), abs(np.nanmax(J_interp)))
    if vmax == 0.0 or np.isnan(vmax):
        vmax = 1e-4

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    cf = ax.contourf(
        B, M, J_interp,
        levels=60,
        cmap='coolwarm',
        norm=norm,
        extend='both'
    )

    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label(r'Stationary Current Flux $J$', rotation=270, labelpad=20)

    # J = 0 zero-current contour line
    ax.contour(
        B, M, J_interp,
        levels=[0.0],
        colors=['black'],
        linestyles=['--'],
        linewidths=[1.8]
    )

    # Theoretical curves
    beta_vals = np.linspace(max(BETA_C, b_min), min(1.25, b_max), 500)
    m_vals = m_onsager(beta_vals)

    if b_min <= BETA_C <= b_max:
        ax.axvline(
            x=BETA_C,
            color='white',
            linestyle=':',
            linewidth=2.0,
            label=rf'$\beta_c = {BETA_C:.6f}$'
        )

    ax.plot(
        beta_vals,
        m_vals,
        color='black',
        linewidth=2.2,
        label=r'$m_\beta$'
    )

    # Scatter of measured data points
    ax.scatter(
        df['beta'],
        df['m_plus'],
        c='black',
        s=8,
        alpha=0.3,
        label='Sample points'
    )

    ax.set_xlim(b_min, b_max)
    ax.set_ylim(m_min, m_max)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$m_+$')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.85)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    pdf_path = os.path.splitext(output_path)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close()
    print(f"Current heatmap saved to: {output_path} and {pdf_path}")


def plot_points(df, output_path):
    if df is None or len(df) == 0:
        print("No data available for points scatter plot.")
        return

    def create_figure(zoomed=False):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Theoretical curves
        b_max = 1.25 if not zoomed else df['beta'].max() + 0.03
        b_min_onsager = BETA_C
        beta_vals = np.linspace(b_min_onsager, b_max, 1000)
        m_vals = m_onsager(beta_vals)

        ax.axvline(
            x=BETA_C,
            color='black',
            linestyle='--',
            linewidth=1.8,
            label=r'$\beta_c = 0.440686$'
        )

        ax.plot(
            beta_vals,
            m_vals,
            color='black',
            linewidth=2.4,
            label=r'$m_\beta = [1 - \sinh^{-4}(2\beta)]^{1/8}$',
            zorder=5
        )

        if not zoomed:
            beta_sub = np.linspace(0.2, BETA_C, 200)
            ax.plot(beta_sub, np.zeros_like(beta_sub), color='black', linewidth=2.4, zorder=5)

        # Categorize points by phase and current
        has_phase = 'phase' in df.columns or 'is_metastable' in df.columns

        if has_phase:
            if 'is_metastable' in df.columns:
                mask_meta = (df['is_metastable'] == 1) & (df['j_flux'] > 0.0)
            else:
                mask_meta = (df['phase'] == 'metastable') & (df['j_flux'] > 0.0)

            if 'phase' in df.columns:
                mask_weak = (df['phase'] == 'weakly_unstable') & (df['j_flux'] > 0.0)
            else:
                mask_weak = (~mask_meta) & (df['j_flux'] > 0.0)

            if 'phase' in df.columns:
                mask_stable = (df['phase'] == 'stable') | (df['j_flux'] <= 0.0)
            else:
                mask_stable = df['j_flux'] <= 0.0

            mask_other = ~(mask_meta | mask_weak | mask_stable)

            if mask_stable.any():
                ax.scatter(
                    df.loc[mask_stable, 'beta'],
                    df.loc[mask_stable, 'm_plus'],
                    c='#1f77b4',
                    marker='o',
                    s=28,
                    alpha=0.75,
                    edgecolors='none',
                    label=f'Stable Downhill ($J \\leq 0$, n={mask_stable.sum()})',
                    zorder=3
                )

            if mask_weak.any():
                ax.scatter(
                    df.loc[mask_weak, 'beta'],
                    df.loc[mask_weak, 'm_plus'],
                    c='#ff7f0e',
                    marker='^',
                    s=32,
                    alpha=0.75,
                    edgecolors='none',
                    label=f'Weakly-unstable ($J > 0$, n={mask_weak.sum()})',
                    zorder=3
                )

            if mask_meta.any():
                ax.scatter(
                    df.loc[mask_meta, 'beta'],
                    df.loc[mask_meta, 'm_plus'],
                    c='#d62728',
                    marker='s',
                    s=36,
                    alpha=0.85,
                    edgecolors='black',
                    linewidths=0.5,
                    label=f'Metastable Uphill ($J > 0$, n={mask_meta.sum()})',
                    zorder=4
                )

            if mask_other.any():
                ax.scatter(
                    df.loc[mask_other, 'beta'],
                    df.loc[mask_other, 'm_plus'],
                    c='#9467bd',
                    marker='x',
                    s=26,
                    alpha=0.7,
                    label=f'Chaotic / Other (n={mask_other.sum()})',
                    zorder=2
                )
        else:
            mask_up = df['j_flux'] > 0.0
            mask_down = ~mask_up

            if mask_down.any():
                ax.scatter(
                    df.loc[mask_down, 'beta'],
                    df.loc[mask_down, 'm_plus'],
                    c='#1f77b4',
                    marker='o',
                    s=28,
                    alpha=0.75,
                    edgecolors='none',
                    label=f'Downhill ($J \\leq 0$, n={mask_down.sum()})',
                    zorder=3
                )

            if mask_up.any():
                ax.scatter(
                    df.loc[mask_up, 'beta'],
                    df.loc[mask_up, 'm_plus'],
                    c='#d62728',
                    marker='o',
                    s=30,
                    alpha=0.8,
                    edgecolors='none',
                    label=f'Uphill current ($J > 0$, n={mask_up.sum()})',
                    zorder=4
                )

        if not zoomed:
            ax.set_xlim(0.2, 1.2)
            ax.set_ylim(0.0, 1.0)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        else:
            b_pad = (df['beta'].max() - df['beta'].min()) * 0.05
            m_pad = (df['m_plus'].max() - df['m_plus'].min()) * 0.05
            ax.set_xlim(max(0.2, df['beta'].min() - b_pad), df['beta'].max() + b_pad)
            ax.set_ylim(max(0.0, df['m_plus'].min() - m_pad), min(1.02, df['m_plus'].max() + m_pad))

        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel(r'$m_+$')
        ax.grid(True, linestyle=':', color='gray', alpha=0.5)
        ax.legend(
            loc='lower right',
            frameon=True,
            framealpha=0.9,
            edgecolor='lightgray',
            fancybox=True,
            borderpad=0.8
        )
        plt.tight_layout()
        return fig

    # 1. Full phase space plot
    fig_full = create_figure(zoomed=False)
    fig_full.savefig(output_path, dpi=300)
    pdf_path = os.path.splitext(output_path)[0] + '.pdf'
    fig_full.savefig(pdf_path)
    plt.close(fig_full)
    print(f"Points phase plot saved to: {output_path} and {pdf_path}")

    # 2. Zoomed plot if data points are concentrated in a band
    if df['m_plus'].min() > 0.4 and df['beta'].min() >= 0.35:
        zoomed_path = os.path.splitext(output_path)[0] + '_zoomed.png'
        fig_zoom = create_figure(zoomed=True)
        fig_zoom.savefig(zoomed_path, dpi=300)
        plt.close(fig_zoom)
        print(f"Zoomed points plot saved to: {zoomed_path}")


def main():
    setup_matplotlib_style()

    parser = argparse.ArgumentParser(description="Plot Uphill Current phase diagram and heatmap from simulation results.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="results/creutz_thermal_scan.csv",
        help="Path to CSV file with simulation results (default: results/creutz_thermal_scan.csv)"
    )
    parser.add_argument(
        "--output-phase",
        type=str,
        default="results/uphill_phase_diagram.png",
        help="Output path for phase diagram plot (default: results/uphill_phase_diagram.png)"
    )
    parser.add_argument(
        "--output-heatmap",
        type=str,
        default="results/current_heatmap.png",
        help="Output path for heatmap plot (default: results/current_heatmap.png)"
    )
    parser.add_argument(
        "--output-points",
        type=str,
        default="results/points_by_phase.png",
        help="Output path for points scatter plot (default: results/points_by_phase.png)"
    )

    args = parser.parse_args()

    df = None
    if os.path.exists(args.input):
        try:
            df = pd.read_csv(args.input)
            print(f"Loaded {len(df)} simulation records from {args.input}")
        except Exception as e:
            print(f"Error reading CSV {args.input}: {e}")
    else:
        print(f"Notice: CSV file {args.input} not found yet. Generating base theoretical phase diagram.")

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_phase)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plot_phase_diagram(df, args.output_phase)
    if df is not None:
        plot_heatmap(df, args.output_heatmap)
        plot_points(df, args.output_points)


if __name__ == "__main__":
    main()
