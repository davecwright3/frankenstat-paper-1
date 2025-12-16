#!/usr/bin/env python3

import json
from pathlib import Path

import cyclopts
import discovery as ds
import matplotlib.pyplot as plt
import numpy as np
from jax.typing import ArrayLike
from loguru import logger
from scipy import stats

import discovery_utils as du

app = cyclopts.App()


def figsize(scale):
    """Calculate figure size using golden ratio for publication-quality plots.

    Parameters
    ----------
    scale : float
        Scaling factor for the figure width.

    Returns
    -------
    list[float]
        Figure size as [width, height] in inches.

    """
    fig_width_pt = 513.17  # 469.755    # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def set_publication_params(param_dict=None, scale=0.5):
    """Set matplotlib parameters for publication-quality plots.

    Parameters
    ----------
    param_dict : dict, optional
        Additional matplotlib parameters to override defaults.
    scale : float, optional
        Scaling factor for figure size, by default 0.5.

    """
    plt.rcParams.update(plt.rcParamsDefault)
    params = {
        "backend": "pdf",
        "axes.labelsize": 10,
        "lines.markersize": 4,
        "font.size": 10,
        "xtick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.major.size": 6,
        "ytick.minor.size": 3,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.markeredgewidth": 1,
        "axes.linewidth": 1.2,
        "legend.fontsize": 7,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.dpi": 200,
        "path.simplify": True,
        "font.family": "serif",
        # 'font.serif':'Times New Roman',
        #'text.latex.preamble': [r'\usepackage{amsmath}'],
        "text.usetex": False,
        "figure.figsize": figsize(scale),
    }

    if param_dict is not None:
        params.update(param_dict)

    plt.rcParams.update(params)


def make_plot(
    fname: str | Path,
    xs: list[np.ndarray],
    cdfs: list[np.ndarray],
    sigmas: list[float],
    snrs: list[float],
):
    """Create p-value comparison plot for multiple PTAs.

    Plots the p-value distributions and SNR values for the combined PTA,
    FrankenPTA, and three split PTAs with significance level reference lines.

    Parameters
    ----------
    fname : str or Path
        Output filename for the saved plot.
    xs : list[np.ndarray]
        List of SNR distributions for each PTA (5 elements).
    cdfs : list[np.ndarray]
        List of cumulative distribution functions for each PTA (5 elements).
    sigmas : list[float]
        List of significance levels in sigma for each PTA (5 elements).
    snrs : list[float]
        List of observed SNR values for each PTA (5 elements).

    """
    set_publication_params(scale=1.0)

    colors = ["purple", "orange", "green", "blue", "black"]
    pta_names = ["Combined", "Franken", "PTA 1", "PTA 2", "PTA 3"]
    plt.figure()
    for pta_name, color, cdf, x, snr, sigma in zip(
        pta_names,
        colors,
        cdfs,
        xs,
        snrs,
        sigmas,
        strict=False,
    ):
        plt.plot(x, 1 - cdf, color=color, linestyle="dashed")
        plt.axvline(
            snr,
            color=color,
            linestyle="dashed",
            label=f"{pta_name} p-val {sigma:.2f}σ",
        )

    plt.text(-4, 0.18, r"$1\sigma$", color="grey")
    plt.axhline(0.15866, color="grey", linestyle="dotted")

    plt.axhline(0.02275, color="grey", linestyle="dotted")
    plt.text(-4, 0.026, r"$2\sigma$", color="grey")

    plt.axhline(1.349e-3, color="grey", linestyle="dotted")
    plt.text(-4, 1.6e-3, r"$3\sigma$", color="grey")

    plt.axhline(3.167e-5, color="grey", linestyle="dotted")
    plt.text(-4, 3.8e-5, r"$4\sigma$", color="grey")

    plt.axhline(2.866e-7, color="grey", linestyle="dotted")
    plt.text(-4, 3.4e-7, r"$5\sigma$", color="grey")

    plt.yscale("log")
    plt.xlabel("SNR")
    plt.ylabel(r"$p$-value")
    plt.ylim(bottom=1e-8)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(fname)


@app.command
def get_pvalue(
    data_dir: str | Path,
    save_dir: str | Path,
    fname_prefix: str,
    params: dict[str, ArrayLike] | str | Path,
    psr_prefix: str = "",
    rn_include: bool = True,
    rn_comp: int = 30,
    gw_comp: int = 7,
):
    """Calculate p-value and null distribution using the optimal statistic.

    Computes the signal-to-noise ratio (SNR) and p-value for gravitational wave
    detection using the optimal statistic (OS) on a pulsar timing array.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing pulsar feather files.
    save_dir : str or Path
        Directory where results will be saved.
    fname_prefix : str
        Prefix for output filenames.
    params : dict[str, ArrayLike] or str or Path
        Maximum likelihood parameter estimates, either as a dictionary or
        path to a JSON file containing the parameters.
    psr_prefix : str, optional
        Prefix to filter pulsar files, by default "".
    rn_include : bool, optional
        Whether to include per-pulsar red noise in the model, by default True.
    rn_comp : int, optional
        Number of red noise frequency components, by default 30.
    gw_comp : int, optional
        Number of gravitational wave frequency components, by default 7.

    Returns
    -------
    cdf_dist : np.ndarray
        Cumulative distribution function of the null SNR distribution.
    sigma : float
        Significance level in sigma.
    snr : float
        Observed signal-to-noise ratio.
    snr_dist : np.ndarray
        Null SNR distribution values.

    """
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    if isinstance(params, str) or isinstance(params, Path):
        params = Path(params)
        with params.open("r") as f:
            params: dict[str, ArrayLike] = json.load(f)

    psrs = du.read_pulsar_feathers(data_dir, prefix=psr_prefix)

    curn = du.make_curn_theorist_os(
        psrs,
        array=False,
        include_red_noise=rn_include,
        rn_comp=rn_comp,
        gw_comp=gw_comp,
    )
    logger.debug("likelihood created")
    os = ds.OS(curn)

    new_params = {}
    for psr in psrs:
        new_params.update(psr.noisedict)  # contains rn estimate

    new_params.update(params)

    # new_params.update({"gw_gamma": gw_gamma, "gw_log10_A": gw_log10_A})
    new_params.update(
        {"gw_gamma": params["crn_gamma"], "gw_log10_A": params["crn_log10_A"]},
    )
    logger.debug("params updated")

    snr, cdf_val, snr_dist, cdf_dist = os.get_fixedpar_os_distribution_and_pval(
        new_params,
    )
    logger.debug("pval calc done")
    pval = 1 - cdf_val

    sigma = stats.norm.isf(np.abs(pval))
    logger.debug(f"{sigma=}")

    if not isinstance(save_dir, Path):
        save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    np.save(save_dir / f"{fname_prefix}-cdf", cdf_dist)
    np.save(save_dir / f"{fname_prefix}-pval", pval)
    np.save(save_dir / f"{fname_prefix}-snr", snr)
    np.save(save_dir / f"{fname_prefix}-snr_dist", snr_dist)
    np.save(save_dir / f"{fname_prefix}-sigma", sigma)

    return cdf_dist, sigma, snr, snr_dist


@app.command
def get_pvalues_five_pta(
    data_dir: str | Path,
    make_plots: bool = True,
    do_spna_max_likelihood: bool = False,
    rn_include: bool = True,
    rn_comp: int = 30,
    gw_comp: int = 7,
):
    """Calculate p-values for all five PTAs in FrankenStat analysis.

    Computes p-values and significance levels for the combined PTA, FrankenPTA,
    and three split PTAs, optionally generating a comparison plot.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing all PTA data subdirectories.
    make_plots : bool, optional
        Whether to generate p-value comparison plot, by default True.
    do_spna_max_likelihood : bool, optional
        Whether to use SPNA maximum likelihood results subdirectories,
        by default False.
    rn_include : bool, optional
        Whether to include per-pulsar red noise in the model, by default True.
    rn_comp : int, optional
        Number of red noise frequency components, by default 30.
    gw_comp : int, optional
        Number of gravitational wave frequency components, by default 7.

    """
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    if do_spna_max_likelihood:
        feather_string = "fake_feathers_spna"
        feather_pta_string = "fake_feathers_pta"
        franken_feather_string = "franken_psrs"
    else:
        feather_string = "fake_feathers"
        feather_pta_string = "fake_feathers_pta"
        franken_feather_string = "franken_psrs"

    full_feather_dir = data_dir / feather_string
    franken_feather_dir = data_dir / franken_feather_string

    cdfs = []
    sigmas = []
    snrs = []
    snr_dists = []

    logger.debug(f"Working on {full_feather_dir}")
    cdf, sigma, snr, snr_dist = get_pvalue(
        full_feather_dir,
        data_dir,
        fname_prefix="combined",
        params=full_feather_dir / "max-likelihood-estimate.json",
        rn_include=rn_include,
        rn_comp=rn_comp,
        gw_comp=gw_comp,
    )

    cdfs.append(cdf)
    sigmas.append(sigma)
    snrs.append(snr)
    snr_dists.append(snr_dist)

    logger.debug(f"Working on {franken_feather_dir}")
    cdf, sigma, snr, snr_dist = get_pvalue(
        franken_feather_dir,
        data_dir,
        psr_prefix="franken",
        fname_prefix="franken",
        params=franken_feather_dir / "max-likelihood-estimate.json",
        rn_include=rn_include,
        rn_comp=rn_comp,
        gw_comp=gw_comp,
    )

    cdfs.append(cdf)
    sigmas.append(sigma)
    snrs.append(snr)
    snr_dists.append(snr_dist)

    for i in range(1, 4):
        feather_dir = data_dir / f"{feather_pta_string}_{i}"
        logger.debug(f"Working on {feather_dir}")
        cdf, sigma, snr, snr_dist = get_pvalue(
            feather_dir,
            data_dir,
            fname_prefix=f"pta-{i}",
            params=feather_dir / "max-likelihood-estimate.json",
            rn_include=rn_include,
            rn_comp=rn_comp,
            gw_comp=gw_comp,
        )

        cdfs.append(cdf)
        sigmas.append(sigma)
        snrs.append(snr)
        snr_dists.append(snr_dist)

    if make_plots:
        make_plot(data_dir / "p-value-plot.png", snr_dists, cdfs, sigmas, snrs)


@app.command
def get_pvalues_five_pta_same_params(
    data_dir: str | Path,
    make_plots: bool = True,
    do_spna_max_likelihood: bool = False,
    rn_include: bool = True,
    rn_comp: int = 30,
    gw_comp: int = 7,
):
    """Calculate p-values for all five PTAs using combined PTA parameters.

    Computes p-values and significance levels for all PTAs (combined, Franken,
    and three splits) using the same parameter estimates from the combined PTA.
    This allows for direct comparison of detection significance when using
    identical noise parameters.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing all PTA data subdirectories.
    make_plots : bool, optional
        Whether to generate p-value comparison plot, by default True.
    do_spna_max_likelihood : bool, optional
        Whether to use SPNA maximum likelihood results subdirectories,
        by default False.
    rn_include : bool, optional
        Whether to include per-pulsar red noise in the model, by default True.
    rn_comp : int, optional
        Number of red noise frequency components, by default 30.
    gw_comp : int, optional
        Number of gravitational wave frequency components, by default 7.

    """
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    if do_spna_max_likelihood:
        feather_string = "fake_feathers_spna"
        feather_pta_string = "fake_feathers_pta"
        franken_feather_string = "franken_psrs"
    else:
        feather_string = "fake_feathers"
        feather_pta_string = "fake_feathers_pta"
        franken_feather_string = "franken_psrs"

    full_feather_dir = data_dir / feather_string
    franken_feather_dir = data_dir / franken_feather_string

    cdfs = []
    sigmas = []
    snrs = []
    snr_dists = []

    logger.debug(f"Working on {full_feather_dir}")
    cdf, sigma, snr, snr_dist = get_pvalue(
        full_feather_dir,
        data_dir,
        fname_prefix="combined-params-combined",
        params=full_feather_dir / "max-likelihood-estimate.json",
        rn_include=rn_include,
        rn_comp=rn_comp,
        gw_comp=gw_comp,
    )

    cdfs.append(cdf)
    sigmas.append(sigma)
    snrs.append(snr)
    snr_dists.append(snr_dist)

    logger.debug(f"Working on {franken_feather_dir}")
    cdf, sigma, snr, snr_dist = get_pvalue(
        franken_feather_dir,
        data_dir,
        psr_prefix="axis",
        fname_prefix="combined-params-franken",
        params=full_feather_dir / "max-likelihood-estimate.json",
        rn_include=rn_include,
        rn_comp=rn_comp,
        gw_comp=gw_comp,
    )

    cdfs.append(cdf)
    sigmas.append(sigma)
    snrs.append(snr)
    snr_dists.append(snr_dist)

    for i in range(1, 4):
        feather_dir = data_dir / f"{feather_pta_string}_{i}"
        logger.debug(f"Working on {feather_dir}")
        cdf, sigma, snr, snr_dist = get_pvalue(
            feather_dir,
            data_dir,
            fname_prefix=f"combined-params-pta-{i}",
            params=full_feather_dir / "max-likelihood-estimate.json",
            rn_include=rn_include,
            rn_comp=rn_comp,
            gw_comp=gw_comp,
        )

        cdfs.append(cdf)
        sigmas.append(sigma)
        snrs.append(snr)
        snr_dists.append(snr_dist)

    if make_plots:
        make_plot(
            data_dir / "combined-params-p-value-plot.png",
            snr_dists,
            cdfs,
            sigmas,
            snrs,
        )


if __name__ == "__main__":
    app()
