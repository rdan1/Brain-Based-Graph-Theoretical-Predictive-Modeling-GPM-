
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        BRAIN-BASED GRAPH-THEORETICAL PREDICTIVE MODELING (GPM)
%   The GPM employs graph theory to define brain predictors of a phenotype (e.g., behavior, symptom), within a cross-validation framework.
%   Graph-theoretical measures are computed from the brain's functional connectome.
%   
%   Working and null models: 
%           The script includes a comparison of the GPM ('working model') to a corresponding null model. The null model is defined as the same cross-validated model but without the brain predictors.
%           This is relevant for predictive models that include covariates other than the graph-theoretical predictors (e.g., baseline clinical scores, sex, age).
%           It allows for testing the added effect of the brain graph-theoretical predictors beyond the effect of the covariates.
%
%   Required inputs:
%           all_phenotype (y): a phenotype to predict (e.g., behavior, cognition, clinical symptom)
%           all_graph (x): a matrix with graph-theoretical measures (can be computed using the BCT) 
%
%   Outputs:
%           yhat: predicted scores.
%           MSE, correlation: predictive performance metrics
%   
%   References:
%           If you use this code, please cite:
%           Dan, R., Whitton, A.E., Treadway, M.T. et al. Brain-based graph-theoretical predictive modeling to map the trajectory of anhedonia, impulsivity, and hypomania from the human functional connectome. 
%               Neuropsychopharmacol. 49, 1162–1170 (2024). https://doi.org/10.1038/s41386-024-01842-1
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     This script includes permutation testing to obtain a p value for correlation between predicted and observed scores.
%         If correlation is used as a predictive performance metric, p value should be obtained through permutation testing.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ---------------------INPUTS---------------------
% phenotype for prediction
load(fullfile(path/to/outcome,'all_phenotype.mat')); 

% graph-theoretical measures
load(fullfile(path/to/graph/matrix,'all_graph.mat'));                                                                                       % matrix with all graph-theoretical measures. dimensions: graph measures x subs

% optional: covariates (e.g., demographic or other baseline clinical measures)
load(fullfile(path/to/covariates,'all_covariate1.mat')); 
load(fullfile(path/to/covariates,'all_covariate2.mat')); 

num_subs = length(all_phenotype);

%% calculate the true prediction correlation
[true_prediction_r_work, true_prediction_r_null] = GPM_predict(all_phenotype, all_graph, all_covariate1, all_covariate2);

num_iterations = 10000;                                                                                                                      % number of iterations for permutation testing
prediction_r_work = zeros(num_iterations,1);                prediction_r_null = zeros(num_iterations,1);
prediction_r_work(1,1) = true_prediction_r_work;          prediction_r_null(1,1) = true_prediction_r_null;                                   % true predictions

%% permutations: random shuffling of data labels to create estimated distributions of the test statistic
for it = 2:num_iterations
    fprintf('\n Performing iteration %d out of %d', it, num_iterations);
    rnd = randperm(num_subs);
    new_phenotype = all_phenotype(rnd);
    [prediction_r_work(it,1), prediction_r_null(it,1)] = GPM_predict(new_phenotype, all_graph, all_covariate1, all_covariate2);
end

%% obtaining p values for work and null models
sorted_prediction_r_work = sort(prediction_r_work(:,1),'descend');
position_work = find(sorted_prediction_r_work==true_prediction_r_work);
pval_work = position_work(1)/num_iterations;

sorted_prediction_r_null = sort(prediction_r_null(:,1),'descend');
position_null = find(sorted_prediction_r_null==true_prediction_r_null);
pval_null = position_null(1)/num_iterations;
