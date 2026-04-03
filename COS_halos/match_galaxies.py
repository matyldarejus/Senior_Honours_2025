# match_galaxies.py
# DEPRECATED - NOT ENOUGH DATA FROM COS-HALOS - especially for quenched
# For each simulated galaxy in a sample HDF5 file, find the best-matching
# observed galaxy from COS-Halos (via pyigm) using:
#   - exact SF/GV/Q category match  (same sSFR thresholds as the simulation)
#   - expanding search window in (log Mstar, log sSFR) until a match is found
#   - nearest neighbour within that window by Euclidean distance
#   - choose_mask to prevent the same obs galaxy being matched twice
#   - quenched sSFR clipping so the -14 floor doesn't distort distance

import os
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyigm.cgm.cos_halos import COSHalos
from pyigm.cgm import cos_halos as pycsh

# Match tolerances 
MASS_RANGE_INIT = 0.10   # starting half-width for mass search
SSFR_RANGE_INIT = 0.10   # starting half-width for sSFR search
MASS_RANGE_LIM  = 0.15   # maximum allowed mass half-width
SSFR_RANGE_LIM  = 0.25   # maximum allowed sSFR half-width
RANGE_STEP      = 0.01   # expansion step per iteration


def quench_thresh(z):
    """
    Quenching threshold in log sSFR [yr^-1]
    """
    return -1.8 + 0.3 * z - 9.0


def ssfr_category(log_ssfr, z):
    """
    Return a string category for a single galaxy

    Accepts:
        log_ssfr : float
            log10(sSFR / yr^-1).  Fully quenched galaxies should be -14.0
        z : float
            Redshift of the galaxy (used for the quenching threshold)

    Returns:
        str : 'SF', 'GV', or 'Q'
    """
    thresh = quench_thresh(z)
    if log_ssfr == -14.0:
        return 'Q'
    elif log_ssfr >= thresh:
        return 'SF'
    elif log_ssfr >= thresh - 1.0:
        return 'GV'
    else:
        return 'Q'

def load_observational_sample(survey_name, loader):
    """
    Load COS-Halos and extract the galaxy properties,
    and return a numpy array for matching

    Returned dtype fields:
        survey    : str, 'COS-Halos'
        cgm_id    : str, identifier string from the survey
        log_mstar : f64, log10(M* / Msun)
        log_ssfr  : f64, log10(sSFR / yr^-1); -14 for quenched / no SFR data
        redshift  : f64, galaxy redshift
        category  : str, 'SF', 'GV', or 'Q'
    """
    records = []

    print(f'    Loading {survey_name}....')
    cgm_survey = loader()
    cgm_survey.load_sys()
    print(f'        SUCCESS!')

    for cgm in cgm_survey.cgm_abs:
        gal = cgm.galaxy
        print(gal)
        
        # Stellar mass
        try:
            log_mstar = float(gal.stellar_mass)
        except Exception:
            continue  # skip galaxies with no stellar mass

        # sSFR
        try:
            sfr       = float(gal.sfr[1])              # linear Msun/yr
            log_mstar = float(gal.stellar_mass)        # log Msun
            mstar_lin = 10 ** log_mstar                # convert to linear
            if sfr <= 0 or mstar_lin <= 0:
                log_ssfr = -14.0
            else:
                log_ssfr = float(np.log10(sfr / mstar_lin))
        except Exception:
            log_ssfr = -14.0

        # Redshift
        try:
            z = float(gal.z)
        except Exception:
            continue

        cat = ssfr_category(log_ssfr, z)

        if cat == 'Q':
            print(f'For quenched galaxy {str(cgm.name)} log sSFR = {log_ssfr}')

        records.append({
            'survey':    survey_name,
            'cgm_id':    str(cgm.name),
            'log_mstar': log_mstar,
            'log_ssfr':  log_ssfr,
            'redshift':  z,
            'category':  cat,
        })

    if not records:
        raise RuntimeError('No galaxy data could be loaded.')

    dt = np.dtype([
        ('survey',    'U20'),
        ('cgm_id',    'U40'),
        ('log_mstar', np.float64),
        ('log_ssfr',  np.float64),
        ('redshift',  np.float64),
        ('category',  'U2'),
    ])
    obs = np.array(
        [(r['survey'], r['cgm_id'], r['log_mstar'], r['log_ssfr'],
          r['redshift'], r['category']) for r in records],
        dtype=dt,
    )
    print(f'Loaded {len(obs)} observational galaxies '
          f'({np.sum(obs["category"] == "SF")} SF, '
          f'{np.sum(obs["category"] == "GV")} GV, '
          f'{np.sum(obs["category"] == "Q")} Q)')
    return obs

# Find the nearest neighbour galaxies for each simulated one
def find_matches(sim_mass, sim_ssfr, sim_z, obs):
    """
    For each simulated galaxy find the closest observational galaxy that:
      1. shares the same SF/GV/Q category, and
      2. falls within a (log Mstar, log sSFR) tolerance window

    The window starts at MASS_RANGE_INIT / SSFR_RANGE_INIT and expands by
    RANGE_STEP until a candidate is found or the limits are exceeded

    Distance uses quenched sSFR clipping: for quenched galaxies the sSFR of
    obs candidates is clipped to the sim galaxy's sSFR before computing the
    deviation, so the -14 floor doesn't inflate distances unfairly

    A choose_mask prevents the same obs galaxy being matched to more than one
    sim galaxy.  Galaxies are processed in descending mass order (heaviest
    first) so high-mass galaxies get first pick

    Accepts:
        sim_mass  : (N,) array, log Mstar [log Msun]
        sim_ssfr  : (N,) array, log sSFR  [log yr^-1]
        sim_z     : (N,) array, redshift
        obs       : data array from load_observational_sample()

    Returns:
        match_idx   : (N,) int array, index into obs for best match (-1 if none)
        match_dist  : (N,) float array, Euclidean distance in (dM*, dsSFR) [dex]
        sim_cat     : (N,) list of str, SF/GV/Q category for each sim galaxy
    """
    n_sim = len(sim_mass)

    match_idx  = np.full(n_sim, -1,  dtype=int)
    match_dist = np.full(n_sim, np.nan)
    sim_cat    = [ssfr_category(sim_ssfr[i], sim_z[i]) for i in range(n_sim)]

    # track which obs galaxies are still available
    choose_mask = np.ones(len(obs), dtype=bool)

    # process sim galaxies in descending mass order (heaviest first)
    order = np.argsort(sim_mass)[::-1]

    for i in order:
        cat    = sim_cat[i]
        thresh = quench_thresh(sim_z[i])

        mass_range = MASS_RANGE_INIT
        ssfr_range = SSFR_RANGE_INIT
        found      = False

        while not found:
            # Create category and window masks
            cat_mask  = obs['category'] == cat
            mass_mask = (obs['log_mstar'] >= sim_mass[i] - mass_range) & \
                        (obs['log_mstar'] <= sim_mass[i] + mass_range)

            if cat == 'Q':
                # For quenched: only an upper sSFR bound
                ssfr_mask = obs['log_ssfr'] <= sim_ssfr[i] + ssfr_range
            else:
                ssfr_mask = (obs['log_ssfr'] >= sim_ssfr[i] - ssfr_range) & \
                            (obs['log_ssfr'] <= sim_ssfr[i] + ssfr_range)

            pool_mask    = cat_mask & mass_mask & ssfr_mask & choose_mask
            pool_indices = np.where(pool_mask)[0]

            if len(pool_indices) > 0:
                found = True
            else:
                if mass_range >= MASS_RANGE_LIM and ssfr_range >= SSFR_RANGE_LIM:
                    print(f'[WARNING] No {cat} match found for sim galaxy {i} '
                          f'(log M*={sim_mass[i]:.2f}, log sSFR={sim_ssfr[i]:.2f}) '
                          f'within tolerance limits.')
                    break
                mass_range = min(mass_range + RANGE_STEP, MASS_RANGE_LIM)
                ssfr_range = min(ssfr_range + RANGE_STEP, SSFR_RANGE_LIM)
                print(f'  Expanding window for sim galaxy {i}: '
                      f'mass ±{mass_range:.2f}, sSFR ±{ssfr_range:.2f} dex')

        if not found:
            continue

        # Quenched sSFR clipping (need explanation)
        # Clip obs sSFR to the sim value for quenched galaxies so the -14
        # Floor doesn't artificially inflate sSFR distances
        pool          = obs[pool_indices]
        ssfr_for_dist = pool['log_ssfr'].copy()
        if cat == 'Q':
            ssfr_for_dist = np.where(
                ssfr_for_dist < thresh,
                sim_ssfr[i],
                ssfr_for_dist,
            )

        mass_dev = np.abs(sim_mass[i] - pool['log_mstar'])
        ssfr_dev = np.abs(sim_ssfr[i] - ssfr_for_dist)
        dist     = np.sqrt(mass_dev**2 + ssfr_dev**2)

        best             = np.argmin(dist)
        match_idx[i]     = pool_indices[best]
        match_dist[i]    = dist[best]
        choose_mask[pool_indices[best]] = False  # mark as used

    return match_idx, match_dist, sim_cat




def make_per_galaxy_plots(sim_mass, sim_ssfr, obs, match_idx, sim_cat, plot_dir):
    """
    Save one scatter plot per sim galaxy showing the sim galaxy and its match
    """
    os.makedirs(plot_dir, exist_ok=True)

    for i, idx in enumerate(match_idx):
        fig, ax = plt.subplots(figsize=(5, 4))
        # Full obs pool in the background
        ax.scatter(obs['log_mstar'], obs['log_ssfr'],
                   c='lightgrey', s=10, zorder=1, label='Obs pool')
        # The sim galaxy
        ax.scatter(sim_mass[i], sim_ssfr[i],
                   c='black', marker='x', s=60, zorder=3,
                   label=f'Sim galaxy {i} ({sim_cat[i]})')
        # The matched obs galaxy
        if idx >= 0:
            ax.scatter(obs['log_mstar'][idx], obs['log_ssfr'][idx],
                       c='red', s=60, zorder=4,
                       label=f'{obs["cgm_id"][idx]} ({obs["survey"][idx]})')
        ax.set_xlabel('log M* [log Msun]')
        ax.set_ylabel('log sSFR [log yr$^{-1}$]')
        ax.set_ylim(-14.5, -8.0)
        ax.legend(fontsize=7)
        ax.set_title(f'Sim galaxy {i}')
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'sim_gal_{i:04d}.png'), dpi=100)
        plt.close(fig)


def make_summary_plot(sim_mass, sim_ssfr, obs, match_idx, model, wind, snap, plot_dir):
    """
    Save a single summary plot of all sim galaxies and their matched obs counterparts
    """
    os.makedirs(plot_dir, exist_ok=True)

    matched  = match_idx >= 0
    fig, ax  = plt.subplots(figsize=(7, 5))

    ax.scatter(obs['log_mstar'], obs['log_ssfr'],
               c='lightgrey', s=12, zorder=1, label='Obs pool')
    ax.scatter(sim_mass[matched],  sim_ssfr[matched],
               c='steelblue', s=20, zorder=2, label='Sim (matched)')
    ax.scatter(sim_mass[~matched], sim_ssfr[~matched],
               c='steelblue', s=20, marker='x', zorder=2, label='Sim (unmatched)')

    # Lines connecting each matched pair
    for i in np.where(matched)[0]:
        ax.plot(
            [sim_mass[i], obs['log_mstar'][match_idx[i]]],
            [sim_ssfr[i], obs['log_ssfr'][match_idx[i]]],
            c='salmon', lw=0.5, zorder=1,
        )

    valid_obs = match_idx[matched]
    ax.scatter(obs['log_mstar'][valid_obs], obs['log_ssfr'][valid_obs],
               c='red', s=30, zorder=3, label='Matched obs')

    ax.set_xlabel('log M* [log Msun]')
    ax.set_ylabel('log sSFR [log yr$^{-1}$]')
    ax.set_ylim(-14.5, -8.0)
    ax.legend(fontsize=8)
    ax.set_title(f'{model} {wind} snap {snap} — obs matching summary')
    fig.tight_layout()
    out = os.path.join(plot_dir, f'{model}_{wind}_{snap}_match_summary.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'Summary plot saved to {out}')


# Helper functions 

def load_sim_sample(sample_file):
    """
    Read the galaxy sample HDF5 produced by the sample-picker script
    """

    with h5py.File(sample_file, 'r') as hf:
        mass    = hf['mass'][:]
        ssfr    = hf['ssfr'][:]
        pos     = hf['position'][:]
        gal_ids = hf['gal_ids'][:]
    return gal_ids, mass, ssfr, pos


def save_matches(out_file, gal_ids, sim_mass, sim_ssfr, sim_z, sim_cat,
                 obs, match_idx, match_dist):
    """
    Write matched pairs to an HDF5 file
    """
    matched_mask = match_idx >= 0

    with h5py.File(out_file, 'w') as hf:
        hf.create_dataset('sim_gal_ids',   data=gal_ids)
        hf.create_dataset('sim_log_mstar', data=sim_mass)
        hf.create_dataset('sim_log_ssfr',  data=sim_ssfr)
        hf.create_dataset('sim_redshift',  data=sim_z)
        hf.create_dataset('sim_category',  data=np.array(sim_cat, dtype='S2'))

        hf.create_dataset('match_obs_index', data=match_idx)
        hf.create_dataset('match_distance',  data=match_dist)
        hf.create_dataset('matched',         data=matched_mask.astype(int))

        valid = match_idx[matched_mask]
        hf.create_dataset('obs_survey',    data=np.array(obs['survey'][valid],   dtype='S20'))
        hf.create_dataset('obs_cgm_id',    data=np.array(obs['cgm_id'][valid],   dtype='S40'))
        hf.create_dataset('obs_log_mstar', data=obs['log_mstar'][valid])
        hf.create_dataset('obs_log_ssfr',  data=obs['log_ssfr'][valid])
        hf.create_dataset('obs_redshift',  data=obs['redshift'][valid])
        hf.create_dataset('obs_category',  data=np.array(obs['category'][valid], dtype='S2'))

        hf.attrs['n_sim']       = len(gal_ids)
        hf.attrs['n_matched']   = int(matched_mask.sum())
        hf.attrs['n_unmatched'] = int((~matched_mask).sum())
        hf.attrs['match_method'] = (
            'nearest neighbour: exact SF/GV/Q category match, '
            'expanding window in (log Mstar, log sSFR), '
            'quenched sSFR clipping, choose_mask deduplication'
        )

    print(f'Saved {matched_mask.sum()}/{len(gal_ids)} matched pairs to {out_file}')

def main():
    model = sys.argv[1]
    wind  = sys.argv[2]
    snap  = sys.argv[3]

    sample_dir  = f'/home/matylda/SHP/Data/samples/'
    sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'
    out_file    = f'{sample_dir}{model}_{wind}_{snap}_matched_obs.h5'
    plot_dir    = f'/home/matylda/SHP/Plots/cos_comparison/'

    gal_ids, sim_mass, sim_ssfr, sim_pos = load_sim_sample(sample_file)
    
    sim_z = np.zeros_like(sim_mass, dtype=float) # zero redshift
    
    print(f'Loaded {len(gal_ids)} simulated galaxies from {sample_file}')
    
    # Load observational data from COS-Halos
    obs = load_observational_sample('COS-Halos', COSHalos)

    # Match
    match_idx, match_dist, sim_cat = find_matches(sim_mass, sim_ssfr, sim_z, obs)

    # Console summary
    print('\nMatching summary:')
    for cat in ['SF', 'GV', 'Q']:
        mask  = np.array(sim_cat) == cat
        n_cat = mask.sum()
        n_ok  = (match_idx[mask] >= 0).sum()
        if n_cat:
            med_dist = np.nanmedian(match_dist[mask])
            print(f'  {cat}: {n_ok}/{n_cat} matched  |  median dist = {med_dist:.3f} dex')

    poor_thresh = np.sqrt(MASS_RANGE_LIM**2 + SSFR_RANGE_LIM**2)
    poor = (match_dist > poor_thresh) & (match_idx >= 0)
    if poor.any():
        print(f'\n[WARNING] {poor.sum()} matches exceed the tolerance-limit distance '
              f'({poor_thresh:.3f} dex) - these were the best available but may be poor.')

    # Save matches
    save_matches(out_file, gal_ids, sim_mass, sim_ssfr, sim_z, sim_cat,
                 obs, match_idx, match_dist)

    # Generate plots
    print('\nGenerating diagnostic plots...')
    make_per_galaxy_plots(sim_mass, sim_ssfr, obs, match_idx, sim_cat, plot_dir)
    make_summary_plot(sim_mass, sim_ssfr, obs, match_idx, model, wind, snap, plot_dir)
    print(f'Plots saved to {plot_dir}')


if __name__ == '__main__':
    main()