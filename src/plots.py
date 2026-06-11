import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


def plot_forecast_vs_actual(y_true, y_pred,
                            title="",
                            save_path=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true, label="True", alpha=0.8)
    ax.plot(y_pred, label="Predict", linestyle="--")
    ax.set(title=title, xlabel="Hour", ylabel="Temperature (°C)")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metric_per_fold(scores,
                         metric: str = "MAE",
                         ylabel: str = None,
                         save_path=None):
    """
    scores : list of dicts (1 per fold) or DataFrame, with columns
    split, model and metric, 1 line per model
    """
    df = pd.DataFrame(scores)
    pivot = df.pivot(index="Split", columns="model", values=metric)
    fig, ax = plt.subplots(figsize=(14, 4))
    pivot.plot(ax=ax, marker="o")
    ax.set_xlabel("Fold")
    ax.set_ylabel(ylabel or f"{metric} (°C)")
    ax.set_title(f"{metric} per fold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_fold_prediction(scores, 
                         split, 
                         model: str = None, 
                         save_path=None):
    """
    plot true vs predict for 1 or + folds, subplot
    - scores : list of dicts or DataFrame ; 
    - split : int or list of ints ;
    - model : model name if several
    """
    df = pd.DataFrame(scores)
    if model is not None:
        df = df[df["model"] == model]

    splits = [split] if isinstance(split, int) else list(split)
    fig, axes = plt.subplots(len(splits), 1, figsize=(14, 4 * len(splits)), squeeze=False)

    for ax, sp in zip(axes.ravel(), splits):
        row = df[df["Split"] == sp].iloc[0]
        ax.plot(row["timestamp"], row["y_test"], color="steelblue", linewidth=1.5, label="real")
        ax.plot(row["timestamp"], row["y_pred"], color="tomato", linewidth=1.5, linestyle="--", label="predicted")
        ax.fill_between(row["timestamp"], row["y_test"], row["y_pred"], alpha=0.15, color="orange")
        ax.set_title(
            f"{row['model']} — Fold {int(row['Split'])} ({row['Month']} {row['Window']})  "
            f"MAE={row['MAE']:.2f}°C  R²={row['R2']:.3f}"
        )
        ax.set_ylabel("°C")
        ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_residuals_distribution(scores, 
                                save_path=None):
    """
    residuals histogram + th normal curve per model
    - scores : list of dicts (1 per fold) or df
        with columns : model, y_test, y_pred
    """
    df = pd.DataFrame(scores)
    residuals = {
        name: np.concatenate(g["y_pred"].values) - np.concatenate(g["y_test"].values)
        for name, g in df.groupby("model")
    }
    fig, axes = plt.subplots(1, len(residuals), figsize=(7 * len(residuals), 5))
    axes = np.atleast_1d(axes)

    for ax, (model_name, res) in zip(axes, residuals.items()):
        mu, std = res.mean(), res.std()
        x = np.linspace(res.min(), res.max(), 200)
        ax.hist(res, bins=40, density=True, color="steelblue", edgecolor="white", alpha=0.7, label="residuals")
        ax.plot(x, norm.pdf(x, mu, std), color="tomato", linewidth=2, label=f"N({mu:.2f}, {std:.2f})")
        ax.axvline(0,  color="black",  linewidth=1.5, linestyle="--", label="zero")
        ax.axvline(mu, color="orange", linewidth=1.5, linestyle=":",  label=f"mean={mu:.2f}")
        ax.set_title(model_name)
        ax.set_xlabel("Residual (°C)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    fig.suptitle("Residuals distribution vs normal law", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_residuals_timeline(scores, 
                            selected_folds: list = None, 
                            save_path=None):
    """
    residuals over time, 1 subplot per model
    - scores : list of dicts or DataFrame, with columns
        model, Split, timestamp, y_test, y_pred
        selected_folds=None → all
    """
    df = pd.DataFrame(scores)
    if selected_folds is not None:
        df = df[df["Split"].isin(selected_folds)]

    models = list(df["model"].unique())
    fig, axes = plt.subplots(len(models), 1, figsize=(16, 4 * len(models)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, model_name in zip(axes, models):
        sub = df[df["model"] == model_name].sort_values("Split")
        ts  = np.concatenate(sub["timestamp"].values)
        res = np.concatenate(sub["y_pred"].values) - np.concatenate(sub["y_test"].values)

        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.scatter(ts, res, s=8, alpha=0.5, color="steelblue")
        ax.fill_between(ts, res, 0, where=(res > 0), color="tomato",    alpha=0.2, label="over-estimation")
        ax.fill_between(ts, res, 0, where=(res < 0), color="steelblue", alpha=0.2, label="under-estimation")
        ax.set_title(f"{model_name} — residuals  mean={res.mean():.2f}°C  std={res.std():.2f}°C")
        ax.set_ylabel("Residual (°C)")
        ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparaison(results_df: pd.DataFrame, metric="RMSE", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    results_df.groupby("model")[metric].mean().sort_values().plot.barh(ax=ax)
    ax.set_title(f"Model comparison — mean {metric} (CV)")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_fold_spans(df: pd.DataFrame, time_col: str = "timestamp", target_col: str = "tempint", save_path=None):
    """Time series + per-fold test periods highlighted."""
    fold_cols = [c for c in df.columns if c.startswith("fold_")]
    colors    = plt.cm.tab10.colors

    spans = {
        col: (df.loc[df[col] == 1, time_col].min(), df.loc[df[col] == 1, time_col].max())
        for col in fold_cols
    }

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(df[time_col], df[target_col], color="steelblue", linewidth=0.6, label=target_col)
    [ax.axvspan(t_min, t_max, alpha=0.3, color=colors[i % len(colors)])
     for i, (t_min, t_max) in enumerate(spans.values())]

    ax.set_title("Test periods per fold")
    ax.set_ylabel("°C")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_serre_vs_meteo(df_serre: pd.DataFrame, 
                        df_meteo: pd.DataFrame, 
                        save_path=None):
    """
    compare greenhouse data vs external weather data
    - df_serre : greenhouse, with 'ts_ceil', 'tempint', 'humedadint'
    - df_meteo : weather, with 'ts', 'temperature_2m', 'relative_humidity_2m'
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Temperature
    if 'tempint' in df_serre.columns:
        temp_col = [c for c in df_meteo.columns if 'temperature' in c.lower() or 'temp' in c.lower()]
        if temp_col:
            axes[0].plot(df_serre['ts'], df_serre['tempint'], color='red', label='Greenhouse U5', alpha=0.3, linewidth=1.5)
            axes[0].plot(df_meteo['ts'], df_meteo[temp_col[0]], color='red', label='External weather', alpha=0.5, linewidth=1.5)
            axes[0].set_title('Temperature comparison', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Temperature (°C)', fontsize=11)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

    # Humidity
    if 'humedadint' in df_serre.columns:
        humidity_col = [c for c in df_meteo.columns if 'humidity' in c.lower() or 'humid' in c.lower()]
        if humidity_col:
            axes[1].plot(df_serre['ts'], df_serre['humedadint'], color='blue', label='Greenhouse U5', alpha=0.5, linewidth=1.5)
            axes[1].plot(df_meteo['ts'], df_meteo[humidity_col[0]], color='blue', label='External weather', alpha=0.2, linewidth=1.5)
            axes[1].set_title('Humidity comparison', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Humidity (%)', fontsize=11)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
