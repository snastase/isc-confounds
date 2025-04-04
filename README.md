# Confound regression models for ISC analysis
This repository accompanies a manuscript in preparation comparing different confound regression models for intersubject correlation analysis with naturalistic stimuli. Intersubject correlation (ISC) analysis has become a popular method for capturing synchronized, stimulus-evoked shared brain activity in naturalistic contexts ([Hasson et al., 2004](https://doi.org/10.1126/science.1089506); [Nastase et al., 2019](https://doi.org/10.1093/scan/nsz037)). Inspired by related work on resting-state functional connectivity ([Ciric et al., 2017](https://doi.org/10.1016/j.neuroimage.2017.03.020); [Parkes et al., 2018](https://doi.org/10.1016/j.neuroimage.2017.12.073)), we compared how a number of different combinations of confound variables (extracted using fMRIPrep; [Esteban et al., 2019](https://doi.org/10.1038/s41592-018-0235-4)) impact ISC estimates. We evaluate these models on $N = 754$ spoken story-listening scans from the [Narratives](https://snastase.github.io/datasets/ds002345) dataset ([Nastase et al., 2021](https://doi.org/10.1038/s41597-021-01033-3)) and $N = 291$ movie-watching scans from *The Grand Budapest Hotel* ([Visconti di Oleggio Castello et al., 2020](https://doi.org/10.1038/s41597-020-00735-4)), the *Life* nature documentary ([Nastase et al., 2017](https://doi.org/10.1093/cercor/bhx138)), and *Raiders of the Lost Ark* ([Nastase, 2018](https://www.proquest.com/openview/e78f49f73687a128fe2116ef094a23b1)).

This work was presented in a poster at the annual meeting of the Organization for Human Brain Mapping (OHBM) 2024 in Seoul, South Korea: [`poster`](https://docs.google.com/presentation/d/1V2ZUN3QgV_whc2ZrVCxQ0LT-dbZ0IcjRLKz04R-UrrU/edit?usp=sharing)

If you find this work helpful, please cite the following reference:
- Nastase, S. A., & Hasson, U. (2024, June). *Confound regression models for intersubject correlation analysis with naturalistic stimuli*. Poster presented at the annual meeting of the Organization for Human Brain Mapping in Seoul, South Korea. https://github.com/snastase/isc-confounds

#### Workflow
1. Use `narratives_metadata.py` and `movies_metadata.py` to construct JSON files containing filenames and metadata for the `narratives` and `movies` datasets.
2. Use `model_specification.py` to construct JSON specifying different confound models.
3. Use `extract_confounds.py` to extract confound variables from TSV files output by fMRIPrep.
4. Use `parcel_masks.py` to construct individual ROI masks and atlas parcellation files.
5. Use `roi_average.py` and `atlas_average.py` to extract average fMRI time series for each prespecified ROI or a whole-brain parcellation, respectively.
6. Use `confound_regression.py` to run confound regression on ROIs/parcels (requres AFNI installation for `3dTproject`).
7. Use `parcel_isc.py` to run intersubject correlation (ISC) analysis for all confound models in each ROI/parcel (requires `brainiak` installation).
8. Use `roi_plot.py` and `atlas_plot.py` to visualize ROI/parcellation results (requires Connectome Workbench for surface plotting).

#### References
- Ciric, R., Wolf, D. H., Power, J. D., Roalf, D. R., Baum, G. L., Ruparel, K., Shinohara, R. T., Elliott, M. A., Eickhoff, S. B., Davatzikos, C., Gur, R. C., Gur, R., E., Bassett, D. S., & Satterthwaite, T. D. (2017). Benchmarking of participant-level confound regression strategies for the control of motion artifact in studies of functional connectivity. *NeuroImage*, *154*, 174–187. https://doi.org/10.1016/j.neuroimage.2017.03.020
- Esteban, O., Markiewicz, C. J., Blair, R. W., Moodie, C. A., Ilkay Isik, A., Erramuzpe, A., Kent, J. D., Goncalves, M., DuPre, E., Snyder, M., Oya, H., Ghosh, S. S., Wright, J., Durnez, J., Poldrack, R. A., & Gorgolewski, K. J. (2019). fMRIPrep: a robust preprocessing pipeline for functional MRI. *Nature Methods*, *16*, 111–116. https://doi.org/10.1038/s41592-018-0235-4
- Hasson, U., Nir, Y., Levy, I., Fuhrmann, G., & Malach, R. (2004). Intersubject synchronization of cortical activity during natural vision. *Science*, *303*(5664), 1634–1640. https://doi.org/10.1126/science.1089506
- Nastase, S. A. (2018). *The Geometry of Observed Action Representation During Natural Vision*. Dartmouth College. https://www.proquest.com/openview/e78f49f73687a128fe2116ef094a23b1
- Nastase, S. A., Connolly, A. C., Oosterhof, N. N., Halchenko, Y. O., Guntupalli, J. S., Visconti di Oleggio Castello, M., Gors, J., Gobbini, M. I., & Haxby, J. V. (2017). Attention selectively reshapes the geometry of distributed semantic representation. *Cerebral Cortex*, *27*(8), 4277–4291. https://doi.org/10.1093/cercor/bhx138
- Nastase, S. A., Gazzola, V., Hasson, U., & Keysers, C. (2019). Measuring shared responses across subjects using intersubject correlation. *Social Cognitive and Affective Neuroscience*, *14*(6), 667–685. https://doi.org/10.1093/scan/nsz037
- Nastase, S. A., Liu, Y.-F., Hillman, H., Zadbood, A., Hasenfratz, L., Keshavarzian, N., Chen, J., Honey, C. J., Yeshurun, Y., Regev, M., Nguyen, M., Chang, C. H. C., Baldassano, C., Lositsky, O., Simony, E., Chow, M. A., Leong, Y. C., Brooks, P. P., Micciche, E., Choe, G., Goldstein, A., Vanderwal, T., Halchenko, Y. O., Norman, K. A., & Hasson, U. (2021). The "Narratives" fMRI dataset for evaluating models of naturalistic language comprehension. *Scientific Data*, *8*, 250. https://doi.org/10.1038/s41597-021-01033-3
- Parkes, L., Fulcher, B., Yücel, M., & Fornito, A. (2018). An evaluation of the efficacy, reliability, and sensitivity of motion correction strategies for resting-state functional MRI. *NeuroImage*, *171*, 415–436. https://doi.org/10.1016/j.neuroimage.2017.12.073
- Visconti di Oleggio Castello, M., Chauhan, V., Jiahui, G., & Gobbini, M. I. (2020). An fMRI dataset in response to "The Grand Budapest Hotel", a socially-rich, naturalistic movie. *Scientific Data*, *7*, 383. https://doi.org/10.1038/s41597-020-00735-4



