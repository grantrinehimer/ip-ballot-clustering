## Directory Decriptions

- all_models (from dropbox, contains zips for each number of candidates, **removed due to storage constraints**)
    -  This directory is 70 GB so it is not included.
- 7CandidateElectionPlots (used by corresponding notebook)
    - stdev_coverage
    - stdev_v_distance
    - stdev_v_distance_by_cluster
    - stdev_v_stdev
    - stdev_v_stdev_weighted
- archived_notebooks
    - Outdated notebooks
- Edinburgh2017Ward2Plots
    - models (**removed due to storage constraints**)
    - results
        - all_ballots_histograms
        - cluster_histograms
        - distance_scatter
        - mds
        - value_histograms
        - centroids_silhouette_fraction_eq.txt
- feb_25_runs (most updated cluster runs)
- scot-elex-main (all election csvs)


## File Descriptions

### Python Files
- ip_models.py
    - Contains all current IP models for clustering
- analysis.py
    - Contains some helper functions for conversion and analysis of IP model solutions
- Clustering_Functions.py
    - Contains utility functions primarily used for conversions to embeddings and loading election csvs

### Notebooks

- 7CandidateElectionPlots.ipynb
    - Uses 2_and_3_clusterings.pkl to generate plots of centroid distances over all 7 candidate elections.
- Edinburgh2017Ward2Plots.ipynb
    - Uses IP models in the identically named diretory to generate several plots. Note that these models are outdated.
- ModelsToDataframe.ipynb
    - Takes .sol model files in all_models and feb_25_runs directories and converts them into pkl/csv files.

### Data Files
- 2_and_3_clusterings.pkl
    - Contains a variety of correct 2 and 3 candidate clusterings for both IP models and approximation algorithms
- 2_cluster_results_feb25_updated.pkl
    - Contains results of feb_25_runs (2 clusters)
- 3_cluster_results_feb25_updated.pkl
    - Contains results of feb_25_runs (3 clusters)
- edin_17_2_full_continuous.sol
    - The full continuous solution file for edin_17_2.
- edin_17_2_solutions.sol
    - All the most recently updated solutions for edin_17_2
- kris_e172.csv
    - Includes some additional results of algorithms from Kris for edin_17_2
- outdated_all_model_results.pkl (**removed due to storage constraints**)
    - Outdated results from the initial cluster runs (from results in all_models directory)



