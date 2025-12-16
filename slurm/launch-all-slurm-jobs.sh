#!/usr/bin/env sh

# ==============================================================================
# SIMULATE COMBINED PTA
# ==============================================================================
# Simulate full, combined data
# `simulate_pull_from_priors_parallel` on CPU
simulate_combined_id=$(sbatch --parsable simulate-combined.sh)
sleep 1
echo "Simulate combined $simulate_combined_id"

# ==============================================================================
# NOISE ANALYSIS - COMBINED PTA
# ==============================================================================
# Run single pulsar noise analyses (varied white and red noise) on the full data
# Only start after the previous job's corresponding array element has finished
# `spna_max_likelihood_serial` on GPU. It's faster than parallel on CPUs, I checked (twice)
spna_combined_id=$(sbatch --parsable \
    --dependency=aftercorr:"${simulate_combined_id}" \
    spna-combined.sh)
sleep 1
echo "SPNA combined $spna_combined_id"
# ==============================================================================
# REFIT TIMING MODELS - COMBINED PTA
# ==============================================================================
# Add the estimated noise parameters to the timing model, then refit
# `update_timing_models` on CPU.
refit_tm_spna_combined_id=$(sbatch --parsable \
    --dependency=aftercorr:"${spna_combined_id}" \
    refit-tm-spna-combined.sh)
sleep 1
echo "Refit combined $refit_tm_spna_combined_id"
# ==============================================================================
# HD ANALYSIS - COMBINED PTA
# ==============================================================================
# Run an HD max likelihood estimate over the entire PTA
# `run_array_max_like` on GPU.
hd_max_like_combined_id=$(sbatch --parsable \
    --dependency=aftercorr:"${refit_tm_spna_combined_id}" \
    hd-max-like-combined.sh)
sleep 1
echo "Refit combined $hd_max_like_combined_id"
# ==============================================================================
# SPLIT INTO SUB-PTAS
# ==============================================================================
# Split into three PTAs
# `split_simulated_pta_parallel` on CPU.
split_pta_id=$(sbatch --parsable \
    --dependency=aftercorr:"${refit_tm_spna_combined_id}" \
    split-pta.sh)
sleep 1
echo "Split PTAs $split_pta_id"
# ==============================================================================
# NOISE ANALYSIS - INDIVIDUAL PTAS
# ==============================================================================
# Do single pulsar noise analyses for each PTA
# `spna_max_likelihood_serial` on GPU. 
spna_pta_1_id=$(sbatch --parsable \
    --dependency=aftercorr:"${split_pta_id}" \
    spna-pta-1.sh)

sleep 1
echo "SPNA PTA 1 $spna_pta_1_id"
# `spna_max_likelihood_serial` on GPU. 
spna_pta_2_id=$(sbatch --parsable \
    --dependency=aftercorr:"${split_pta_id}" \
    spna-pta-2.sh)
sleep 1
echo "SPNA PTA 2 $spna_pta_2_id"
# `spna_max_likelihood_serial` on GPU. 
spna_pta_3_id=$(sbatch --parsable \
    --dependency=aftercorr:"${split_pta_id}" \
    spna-pta-3.sh)
sleep 1
echo "SPNA PTA 3 $spna_pta_2_id"
# ==============================================================================
# REFIT & FRANKENIZE
# ==============================================================================
# Add the noise parameters and refit timing model for each pta
# Also frankenize the pulsars
# `update_timing_models` on CPU
refit_tm_spna_pta_1_id=$(sbatch --parsable \
    --dependency=aftercorr:"${spna_pta_1_id}" \
    refit-tm-spna-pta-1.sh)
sleep 1
echo "Refit combined $refit_tm_spna_pta_1_id"


# `update_timing_models` on CPU
refit_tm_spna_pta_2_id=$(sbatch --parsable \
    --dependency=aftercorr:"${spna_pta_2_id}" \
    refit-tm-spna-pta-2.sh)
sleep 1
echo "Refit combined $refit_tm_spna_pta_2_id"


# `update_timing_models` on CPU
refit_tm_spna_pta_3_id=$(sbatch --parsable \
    --dependency=aftercorr:"${spna_pta_3_id}" \
    refit-tm-spna-pta-3.sh)
sleep 1
echo "Refit combined $refit_tm_spna_pta_3_id"

# `frankenize_split_pta` on CPU.
frankenize_id=$(sbatch --parsable \
    --dependency=aftercorr:"${refit_tm_spna_pta_1_id}: \
    ${refit_tm_spna_pta_2_id}: \
    ${refit_tm_spna_pta_3_id}" \
    frankenize.sh)

sleep 1
echo "Refit and Frankenize $frankenize_id"

# ==============================================================================
# NOISE ANALYSIS - FRANKENSTAT PTAS
# ==============================================================================
# Do single pulsar noise analyses for franken pulsars 
# `spna_max_likelihood_serial` on GPU. 
spna_franken_id=$(sbatch --parsable \
    --dependency=aftercorr:"${frankenize_id}" \
    spna-franken.sh)

sleep 1
echo "SPNA PTA 1 $spna_pta_1_id"

# ==============================================================================
# UPDATE NOISE DICTIONARIES: FRANKENSTAT
# ==============================================================================
# Don't refit timing models, just update noise dictionary
update_noise_dict_franken_id=$(sbatch --parsable \
    --dependency=aftercorr:"${spna_franken_id}" \
    update-noise-dicts-franken.sh)


# ==============================================================================
# HD ANALYSIS - SPLIT PTAS
# ==============================================================================
# Run an HD max likelihood estimate over the entire PTA(s)
# `run_array_max_like` on GPU.
hd_max_like_pta_1_id=$(sbatch --parsable \
    --dependency=aftercorr:"${refit_tm_spna_pta_1_id}" \
    hd-max-like-pta-1.sh)
sleep 1
echo "HD max like PTA 1 $hd_max_like_pta_1_id"

# `run_array_max_like` on GPU.
hd_max_like_pta_2_id=$(sbatch --parsable \
    --dependency=aftercorr:"${refit_tm_spna_pta_2_id}" \
    hd-max-like-pta-2.sh)
sleep 1
echo "HD max like PTA 2 $hd_max_like_pta_2_id"

# `run_array_max_like` on GPU.
hd_max_like_pta_3_id=$(sbatch --parsable \
    --dependency=aftercorr:"${refit_tm_spna_pta_3_id}" \
    hd-max-like-pta-3.sh)
sleep 1
echo "HD max like PTA 3 $hd_max_like_pta_3_id"

# `run_array_max_like` on GPU.
hd_max_like_franken_id=$(sbatch --parsable \
    --dependency=aftercorr:"${update_noise_dict_franken_id}" \
    hd-max-like-franken.sh)
sleep 1
echo "HD max like Franken $hd_max_like_franken_id"

# ==============================================================================
# STATISTICAL ANALYSIS
# ==============================================================================
# Run p-value calculation
# This expects all 5 PTAs to be complete
# `get_pvalues_three_pta`
p_value_id=$(sbatch --parsable --dependency=aftercorr:"${hd_max_like_combined_id}: \
    ${hd_max_like_pta_1_id}: \
    ${hd_max_like_pta_2_id}: \
    ${hd_max_like_pta_3_id}: \
    ${hd_max_like_franken_id}" \
    get-all-pvalues.sh)
sleep 1
echo "P values $p_value_id"
