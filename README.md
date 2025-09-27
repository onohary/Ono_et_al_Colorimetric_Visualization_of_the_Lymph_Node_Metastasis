# Ono et al Colorimetric Visualization of the Lymph Node Metastasis

This repository contains the analysis scripts and data used in the paper titled "Colorimetric Visualization of the Lymph Node Metastasis Based on the Size-Dependent Transport of Conjugated Polymer Nanoparticles".

## Repository structure
```
├── machine_learning/
│   └── ML.py                         # Python script for machine learning analysis
│
├── data/
│   ├── Accumulated_Pdot_amount.csv   # Dataset: accumulated Pdot amount
│   ├── POLN_weight.csv               # Dataset: popliteal lymph node weight
│   └── Primary_tumor_diameter.csv    # Dataset: primary tumor diameter
│
├── statistical_analysis/
│   ├── Fig5.R                        # R script for reproducing Figure 5
│   ├── ED_Fig1.R                     # R script for reproducing Extended Data Figure 1
│   ├── ED_Fig2.R                     # R script for reproducing Extended Data Figure 2
│   ├── SI_Fig2.R                     # R script for reproducing Supplementary Figure 2a and 2b
│   └── SI_Fig5.R                     # R script for reproducing Supplementary Figure 5
```

## Requirements

- Python 3.x  
  - scikit-learn  
  - numpy, pandas
- R (≥ 4.0)  
  - Hotelling

   - Each R script in `statistical_analysis/` corresponds to one figure in the paper.  

