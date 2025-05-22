import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from src.utils.config import LOGGING_LEVEL, PlotConfig

# Apply global style
config = PlotConfig()
config.apply()

# Logger setup
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Valid model prefixes
ALLOWED_MODELS = ('dt_net', 'it_net', 'ff_net', 'deadend_fill')


def filter_dataframe(df: pd.DataFrame, filters: dict[str, Any] | None = None) -> pd.DataFrame:
    """Filter DataFrame by specified column values."""
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for col, vals in filters.items():
        vals_list = vals if isinstance(vals, list) else [vals]
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        mask &= df[col].isin(vals_list)
    return df[mask]


def load_results(test_name: str, filters: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load and optionally filter test results for a given test."""
    path = Path('outputs') / 'tests' / test_name / 'results.csv'
    df = pd.read_csv(path)
    return filter_dataframe(df, filters)


def infer_model_type(df: pd.DataFrame) -> str:
    """Ensure exactly one model prefix and return it."""
    prefixes = {m for name in df['model_name'].unique() for m in ALLOWED_MODELS if m in name}
    if len(prefixes) != 1:
        raise ValueError(f'Expected one model type, found: {prefixes}')
    return prefixes.pop()


def plot_overall_acc_vs_perc(test_names: list[str], filters: dict[str, Any] | None, combined_dir: Path) -> None:
    """Plot and save overall accuracy vs train percolation for each test and combined."""
    data: list[tuple[float, str, pd.DataFrame]] = []
    palette = plt.get_cmap('tab10')
    # 1) Individual plots
    for idx, name in enumerate(test_names):
        df = load_results(name, filters)
        iters = df['test_iter'].unique()
        if len(iters) != 1:
            raise ValueError(f"Multiple test_iter for '{name}': {iters}")
        model_type = infer_model_type(df)
        acc_df = df.groupby('train_percolation')['correct'].mean().reset_index()
        data.append((iters[0], model_type, acc_df))

        fig, ax = plt.subplots(figsize=(12, 6))
        color = palette(idx)
        ax.plot(
            acc_df['train_percolation'],
            acc_df['correct'],
            label=model_type,
            color=color,
            linewidth=3,
            marker='o',
            markersize=6,
        )
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Train Percolation', ylabel='Accuracy', title=model_type)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Model Type')
        out = Path('outputs') / 'tests' / name / 'overall_acc_vs_perc.pdf'
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)

    # 2) Combined single plot
    if len(data) > 1:
        combined_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, (_, model_type, acc_df) in enumerate(data):
            color = palette(idx)
            ax.plot(
                acc_df['train_percolation'],
                acc_df['correct'],
                label=model_type,
                color=color,
                linewidth=6,
                marker='o',
                markersize=12,
                clip_on=False,
            )

        ax.set(xlim=(-0.01, 1), ylim=(0, 1), xlabel='Train Percolation', ylabel='Overall Test Accuracy')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Model Type')
        out_comb = combined_dir / 'combined_overall_acc_vs_perc.pdf'
        fig.savefig(out_comb, bbox_inches='tight')
        plt.close(fig)


def plot_acc_vs_perc(test_names: list[str], filters: dict[str, Any] | None, combined_dir: Path) -> None:
    """Plot accuracy vs test percolation, per test and combined."""
    cmap = plt.get_cmap('plasma')

    # 1) Individual plots (updated size & line width)
    for name in test_names:
        df = load_results(name, filters)
        # Ensure single test_maze_size
        if df['test_maze_size'].nunique() != 1:
            raise ValueError(f"Expected single test_maze_size for '{name}', found: {df['test_maze_size'].unique()}")
        # Ensure single test_iter
        if df['test_iter'].nunique() != 1:
            raise ValueError(f"Expected single test_iter for '{name}', found: {df['test_iter'].unique()}")
        model_type = infer_model_type(df)
        models = sorted(
            df['model_name'].unique(), key=lambda mn: df.loc[df['model_name'] == mn, 'train_percolation'].iat[0]
        )
        tps = [float(df[df['model_name'] == m]['train_percolation'].iat[0]) for m in models]
        vals = [tp if tp > 0 else 1e-3 for tp in tps]
        norm = mcolors.Normalize(vmin=min(vals), vmax=max(vals))

        # bigger figure:
        fig, ax = plt.subplots(figsize=(12, 6))
        for m, tp in zip(models, tps, strict=False):
            dfm = df[df['model_name'] == m]
            sub = dfm.groupby('test_percolation')['correct'].mean().reset_index()
            color = cmap(norm(tp if tp > 0 else 1e-3))
            # thicker lines:
            ax.plot(sub['test_percolation'], sub['correct'], marker='o', linewidth=2, label=f'{tp:.3f}', color=color)
            ax.axvline(tp, linestyle='--', color=color, linewidth=2)

        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Test Percolation', ylabel='Test Accuracy', title=model_type)

        fig.subplots_adjust(right=0.75)
        ax.legend(title='Train Perc.', loc='center left', bbox_to_anchor=(1.02, 0.5))

        out = Path('outputs') / 'tests' / name / 'acc_vs_perc.pdf'
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)

    # 2) Combined row of subplots
    if len(test_names) > 1:
        combined_dir.mkdir(parents=True, exist_ok=True)
        fig_comb, axes = plt.subplots(
            1,
            len(test_names),
            figsize=(10 * len(test_names), 6),
            sharey=True,
        )
        axes = np.atleast_1d(axes)

        for idx, name in enumerate(test_names):
            ax = axes[idx]
            df = load_results(name, filters)
            model_type = infer_model_type(df)
            models = sorted(
                df['model_name'].unique(), key=lambda mn: df.loc[df['model_name'] == mn, 'train_percolation'].iat[0]
            )
            tps = [float(df[df['model_name'] == m]['train_percolation'].iat[0]) for m in models]
            norm = mcolors.Normalize(vmin=min(tps), vmax=max(tps))

            for m, tp in zip(models, tps, strict=False):
                dfm = df[df['model_name'] == m]
                sub = dfm.groupby('test_percolation')['correct'].mean().reset_index()
                color = cmap(norm(tp))
                ax.plot(
                    sub['test_percolation'],
                    sub['correct'],
                    marker='o',
                    linewidth=5,  # thicker main line
                    label=f'{tp:.3f}',
                    color=color,
                )
                ax.axvline(
                    tp,
                    linestyle='--',
                    color=color,
                    linewidth=5,
                    clip_on=False,
                    zorder=3,
                )

            ax.set(
                xlim=(0.0, 1.0),
                ylim=(0.0, 1.0),
                title=model_type,
            )
            if idx == 0:
                ax.set_ylabel('Test Acc.')
            ax.set_xlabel('Test Perc.')
            ax.label_outer()

        # shared legend on the right, tighter spacing
        lines, labels = axes[0].get_legend_handles_labels()
        fig_comb.subplots_adjust(right=0.75, wspace=0.2)
        fig_comb.legend(lines, labels, title='Train Perc.', loc='center left', bbox_to_anchor=(0.8, 0.5))

        out_comb = combined_dir / 'combined_acc_vs_perc.pdf'
        fig_comb.savefig(out_comb, bbox_inches='tight')
        plt.close(fig_comb)


def plot_value_vs_size_perc(
    test_names: list[str],
    filters: dict[str, Any] | None = None,
    combined_dir: Path = Path('outputs') / 'visuals' / 'plots',
    value: str = 'correct',
) -> None:
    """Plot heatmaps of `value` vs size & percolation, per test and combined."""
    # verify column exists
    sample_df = load_results(test_names[0], filters)
    if value not in sample_df.columns:
        raise KeyError(f'Column not found: {value}')

    # infer colorbar label
    if value == 'correct':
        cbar_label = 'Test Accuracy'
    elif value.startswith('matches_'):
        field = value[len('matches_') :].replace('_', ' ').title()
        cbar_label = f'Agreement with {field}'
    else:
        cbar_label = value.replace('_', ' ').title()

    cmap = plt.get_cmap('coolwarm')
    all_summaries: list[tuple[str, str, list[tuple[str, float | None, float | None, pd.DataFrame]]]] = []

    # 1) collect per-test summaries
    for name in test_names:
        df = load_results(name, filters)
        model_type = infer_model_type(df)

        summary: list[tuple[str, float | None, float | None, pd.DataFrame]] = []
        for m in df['model_name'].unique():
            dfm = df[df['model_name'] == m]
            tp = float(dfm['train_percolation'].iat[0]) if pd.notna(dfm['train_percolation'].iat[0]) else None
            ts = float(dfm['train_maze_size'].iat[0]) if pd.notna(dfm['train_maze_size'].iat[0]) else None
            heat = dfm.groupby(['test_percolation', 'test_maze_size'])[value].mean().unstack(fill_value=np.nan)
            summary.append((m, tp, ts, heat))

        # sort so NaNs go last
        summary.sort(key=lambda x: x[1] if x[1] is not None else -1)
        all_summaries.append((name, model_type, summary))

        # 2) per-test heatmaps
        n = len(summary)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True, squeeze=False)
        axes = axes.flatten()

        # dynamic font sizes
        _, fh = fig.get_size_inches()
        title_fs = max(12, fh * 5)
        label_fs = max(10, fh * 4)
        tick_fs = max(8, fh * 3)

        first_star = None
        for ax, (m, tp, ts, heat) in zip(axes, summary, strict=False):
            cols = heat.columns
            dx = (cols[1] - cols[0]) if len(cols) > 1 else 1
            extent = [cols[0] - dx / 2, cols[-1] + dx / 2, heat.index[0], heat.index[-1]]
            im = ax.imshow(
                heat.values,
                origin='lower',
                aspect='auto',
                extent=extent,
                cmap=cmap,
                vmin=0,
                vmax=1,
            )

            # if only one subplot, extract "dt_net/original" from "models/dt_net/original.pth"
            if n == 1 and 'models/' in m and m.endswith('.pth'):
                sub = m.split('models/', 1)[1]  # "dt_net/original.pth"
                title = os.path.splitext(sub)[0]  # "dt_net/original"
            elif n == 1:
                title = m
            else:
                title = f'Train Perc. = {tp:.3f}' if tp is not None else 'Train Perc. = NaN'

            ax.set_title(title, fontsize=title_fs)
            ax.set_xlabel('Maze Size', fontsize=label_fs)
            if ax is axes[0]:
                ax.set_ylabel('Test Perc.', fontsize=label_fs)
            ax.tick_params(labelsize=tick_fs)

            if tp is not None and ts is not None:
                star = ax.scatter(
                    ts,
                    tp,
                    marker='*',
                    s=300,
                    edgecolors='gold',
                    facecolors='gold',
                    zorder=4,
                )
                if first_star is None:
                    first_star = star

        # colorbar
        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.06, pad=0.03, shrink=0.9)
        cbar.ax.tick_params(labelsize=tick_fs)
        cbar.ax.yaxis.label.set_size(label_fs)
        cbar.set_label(cbar_label, fontsize=label_fs)

        out_dir = Path('outputs') / 'tests' / name
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / 'acc_vs_size_perc.pdf', bbox_inches='tight')
        plt.close(fig)

    # 3) combined grid
    if not all_summaries:
        return

    n_tests = len(all_summaries)
    n_models = len(all_summaries[0][2])
    fig, axes = plt.subplots(
        n_tests,
        n_models,
        figsize=(4 * n_models, 3.5 * n_tests),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    _, fh = fig.get_size_inches()
    title_fs = max(12, fh * 5)
    label_fs = max(10, fh * 4)
    tick_fs = max(8, fh * 3)

    first_handle = None
    for i, (_name, model_type, summary) in enumerate(all_summaries):
        for j, (_m, tp, ts, heat) in enumerate(summary):
            ax = axes[i, j]
            cols = heat.columns
            dx = (cols[1] - cols[0]) if len(cols) > 1 else 1
            extent = [cols[0] - dx / 2, cols[-1] + dx / 2, heat.index[0], heat.index[-1]]
            im = ax.imshow(
                heat.values,
                origin='lower',
                aspect='auto',
                extent=extent,
                cmap=cmap,
                vmin=0,
                vmax=1,
            )

            ax.set_title(
                f'Train Perc. = {tp:.3f}' if tp is not None else 'Train Perc. = NaN',
                fontsize=title_fs,
            )
            if i == n_tests - 1:
                ax.set_xlabel('Maze Size', fontsize=label_fs)
            if j == 0:
                ax.set_ylabel(f'{model_type}\nTest Perc.', fontsize=label_fs)
            ax.tick_params(labelsize=tick_fs)

            if tp is not None and ts is not None:
                handle = ax.scatter(
                    ts,
                    tp,
                    marker='*',
                    s=300,
                    edgecolors='gold',
                    facecolors='gold',
                    zorder=4,
                    clip_on=False,
                    label=('Training distribution' if first_handle is None else '_nolegend_'),
                )
                if first_handle is None:
                    first_handle = handle

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.06, pad=0.03, shrink=0.9)
    cbar.ax.tick_params(labelsize=tick_fs)
    cbar.ax.yaxis.label.set_size(label_fs)
    cbar.set_label(cbar_label, fontsize=label_fs)

    combined_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(combined_dir / 'combined_acc_vs_size_perc.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_accuracy_by_iter(
    test_names: list[str],
    filters: dict[str, Any] | None = None,
    combined_dir: Path = Path('outputs/visuals/plots'),
    value: str | None = None,
) -> None:
    """Plot test accuracy vs test_percolation for each model and test_iter values, saving one PDF per test."""
    # Ensure combined_dir is a Path
    combined_dir = Path(combined_dir)

    for test_name in test_names:
        # Load and optionally filter results
        df = load_results(test_name, filters)

        # Determine which column to plot
        plot_col = value or 'correct'

        # Identify unique models
        models = sorted(df['model_name'].unique())
        if not models:
            raise ValueError(f"No models found for test '{test_name}'")

        # Create one column per model
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), sharey=True)
        if n_models == 1:
            axes = [axes]

        # Colormap for iterations
        cmap = matplotlib.cm.get_cmap('tab10')

        for idx, ax in enumerate(axes):
            model = models[idx]
            df_model = df[df['model_name'] == model]
            iterations = sorted(df_model['test_iter'].unique())
            if not iterations:
                continue
            # Normalize iteration values for consistent coloring
            norm = mcolors.Normalize(vmin=iterations[0], vmax=iterations[-1])

            for iter_val in iterations:
                df_iter = df_model[df_model['test_iter'] == iter_val]
                summary = df_iter.groupby('test_percolation')[plot_col].mean().reset_index()
                color = cmap(norm(iter_val))
                ax.plot(
                    summary['test_percolation'],
                    summary[plot_col],
                    label=f'iter={iter_val}',
                    marker='o',
                    linewidth=2,
                    color=color,
                )

            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Test Percolation', title=model)
            if idx == 0:
                ax.set_ylabel(plot_col.replace('_', ' ').capitalize())
            ax.legend(title='Test Iter', loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()

        # Save PDF in outputs/tests/<test_name>/
        out_dir = Path('outputs') / 'tests' / test_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'accuracy_by_iter.pdf'
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)


def plot_test(
    test_names: list[str], plot_type: str, filters: dict[str, Any] | None = None, value: str | None = None
) -> None:
    """Entry point: routes to specific plot functions and manages combined output."""
    combined_dir = Path('outputs') / 'visuals' / 'plots'
    handlers = {
        'overall_acc_vs_perc': plot_overall_acc_vs_perc,
        'acc_vs_perc': plot_acc_vs_perc,
        'value_vs_size_perc': plot_value_vs_size_perc,
        'accuracy_by_iter': plot_accuracy_by_iter,
    }
    if plot_type not in handlers:
        raise ValueError(f'Unknown plot type: {plot_type}')
    handlers[plot_type](test_names, filters, combined_dir, value)  # type: ignore


def plot_mazes(
    names_mazes: Sequence[tuple[str, torch.Tensor]],
    file_name: str = 'outputs/visuals/mazes/mazes',
    dpi: int = 300,
    figsize: float = 3.5,
    title_pad: float = 8,
) -> Figure:
    """Plot each (name, tensor or list of tensors) as a column; rows = batch items."""
    # 1) drop any empty columns
    names_mazes = [(n, m) for n, m in names_mazes if m is not None]
    if not names_mazes:
        raise ValueError('Need at least one non-None (name, tensor) pair.')

    # 2) stack / normalize to [B, C, H, W]
    proc = []
    for name, mazes in names_mazes:
        if isinstance(mazes, (list, tuple)):
            x = torch.stack(mazes, 0)
        elif torch.is_tensor(mazes):
            x = mazes
        else:
            raise TypeError(f"Values for '{name}' must be a Tensor or list/tuple of them.")

        # now force a 4-D shape: [B, C, H, W]
        if x.ndim == 2:  # H×W → [1,1,H,W]
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            if x.shape[0] in (1, 3):  # C×H×W → [1,C,H,W]
                x = x.unsqueeze(0)
            else:  # B×H×W → [B,1,H,W]
                x = x.unsqueeze(1)
        elif x.ndim == 4:
            pass
        else:
            raise ValueError(f"Invalid shape {tuple(x.shape)} for '{name}'.")

        proc.append((name, x))

    # 3) make sure all batch sizes agree
    batch_sizes = [x.shape[0] for _, x in proc]
    if len(set(batch_sizes)) != 1:
        raise ValueError(f'All inputs must have the same batch size; got {batch_sizes}')
    batch = batch_sizes[0]
    n_cols = len(proc)

    # 4) build the grid
    fig, axes = plt.subplots(
        batch,
        n_cols,
        figsize=(figsize * n_cols, figsize * batch),
        dpi=dpi,
        squeeze=False,
    )
    # a bit of breathing room
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    # 5) draw each cell exactly once
    for r in range(batch):
        for c, (name, x) in enumerate(proc):
            ax = axes[r, c]
            ax.clear()
            img = x[r].cpu().numpy()
            if img.ndim == 3:  # C×H×W → H×W×C
                img = img.transpose(1, 2, 0)
                ax.imshow(img)
            else:  # H×W
                ax.imshow(img, cmap='gray')
            # only title the top row
            if r == 0:
                ax.set_title(name, pad=title_pad)
            ax.axis('off')

    # 6) save + close
    fig.tight_layout(pad=(title_pad / 72) + 0.1)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    fig.savefig(f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    return fig
