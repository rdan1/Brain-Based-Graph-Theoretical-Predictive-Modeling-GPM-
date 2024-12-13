# Brain-Based Graph-Theoretical Predictive Modeling (GPM):
This repository provides scripts for implementing the brain-based graph-theoretical predictive modeling (GPM) to predict phenotypes from functional connectomes. 

## Scripts

### Step 1: Compute Graph-Theoretical Metrics
use: compute_graph_measures.m to extract graph-theoretical metrics from functional connectomes.

### Step 2: Run the GPM to Predict a Phenotype

Choose one of the following options for running the GPM (options differ in feature selection and/or CV):
1. GPM_LOOCV_alternative1.m: feature selection is done using correlation
2. GPM_LOOCV_alternative2.m: feature selection is done using multiple regression, accounting for covariates and interactions
3. GPM_10Fold_CV_alternative2.m: 10-fold cross-validation instead of leave-one-out cross-validation (LOOCV)
4. GPM_10Fold_CV_ENet.m: feature selection is done using elastic net regularization

### References:
If you use this code, please cite:
Dan, R., Whitton, A.E., Treadway, M.T. et al. (2024) Brain-based graph-theoretical predictive modeling to map the trajectory of anhedonia, impulsivity, and hypomania from the human functional connectome.  Neuropsychopharmacol. 49, 1162–1170. https://doi.org/10.1038/s41386-024-01842-1
https://www.nature.com/articles/s41386-024-01842-1

For questions, contact:
Rotem Dan
rdanyogev@mclean.harvard.edu

### License
This code is distributed under the MIT License. See LICENSE for details.