# Confound regression models for ISC analysis
Confound regression models for ISC analysis

### Workflow
1. Use `narratives_metadata.py` and `movies_metadata.py` to construct JSON files containing filenames and metadata for the `narratives` and `movies` datasets.
2. Use `model_specification.py` to construct JSON specifying different confound models.
3. Use `extract_confounds.py` to extract confound variables from TSV files output by fMRIPrep.
4. Use `parcel_masks.py` to construct individual ROI masks and atlas parcellation files.
5. Use `roi_average.py` and `atlas_average.py` to extract average fMRI time series for prespecified ROIs or a whole-brain parcellation, respectively.
6. Use `confound_regression.py` to run confound regression on ROIs/parcels (requres AFNI installation for `3dTproject`).
7. Use `parcel_isc.py` to run intersubject correlation (ISC) analysis for all confound models in each ROI/parcel (requires `brainiak` installation).
8. Use `roi_plot.py` and `atlas_plot.py` to visualize ROI/parcellation results (requires Connectome Workbench for surface plotting).
