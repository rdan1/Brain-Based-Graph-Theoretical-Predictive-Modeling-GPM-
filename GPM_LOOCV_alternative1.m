
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

%% ---------------------INPUTS---------------------
% phenotype for prediction
load(fullfile(path/to/outcome,'all_phenotype.mat')); 

% graph-theoretical measures
load(fullfile(path/to/graph/matrix,'all_graph.mat'));   % matrix with all graph-theoretical measures. dimensions: graph measures x subs

% optional: covariates (e.g., demographic or other baseline clinical measures)
load(fullfile(path/to/covariates,'all_covariate1.mat')); 
load(fullfile(path/to/covariates,'all_covariate2.mat')); 

%% ---------------------PREDICTORS AND OUTCOME--------------------------------
% null model: includes only covariates (i.e., all predictors except for graph-theoretical measures)

y = all_phenotype;                                                              % outcome. dimensions: subj X 1
X_null = [all_covariate1 all_covariate2];                              % covariates. dimensions: subj X variables
        
%% ---------------LEAVE ONE OUT CROSS VALIDATION (LOOCV)-------------------- 
% In each iteration, the data is divided into a training set and a testing set. Model is built on training set and prediction is done for testing set
% see alternative script for 10-fold CV (recommended)
n = length(y);
num_graph = size(all_graph,1);

for i = 1:n
            fprintf ('\n Leaving out subj #%4.0f', i);     
        
            XTrain_null = X_null;
            XTrain_null(i,:) = [];              % Train set: N-1 (excluding one participant)
            XTest_null = X_null(i,:);        % Test set: the participant who was excluded from model training
        
            train_graph = all_graph;
            train_graph(:,i) = [];                % Train set: N-1 (excluding one participant)
            test_graph = all_graph(:,i);     % Test set: the participant who was excluded from model training
        
            yTrain = y;
            yTrain(i,:) = [];                       % Train set: N-1 (excluding one participant)
            yTest = y(i,:);                         % Test set: the participant who was excluded from model training
        
           %% ---------------(1) WORKING MODEL (GPM)---------------------   
           %---------------BUILDING MODEL ON TRAINING DATA-------
           % step 1: correlate all graph measures with phenotype. Can choose between Pearson or Spearman correlation
            [r_mat, p_mat] = corr(train_graph',yTrain,'rows','complete');
            % [r_mat, p_mat] = corr(train_graph',yTrain,'type','Spearman','rows','complete');            % Alternative option: use Spearman instead of Pearson's correlation           
        
           % step 2: feature selection. Selecting best graph metric: the one that results in the model with the lowest p value
           [~, best_graph_index] = min(p_mat);      
           best_graph_vec(i) = best_graph_index; 
        
           % step 3: model building
           XTrain = [train_graph(best_graph_index,:)' XTrain_null];
           mdl = fitlm(XTrain,yTrain);
           beta_mat(i,:) = mdl.Coefficients.Estimate;
        
           %---------------------TEST DATA: PREDICTION ----------------  
           XTest = [test_graph(best_graph_index,:)' XTest_null];
           XTest0 = [ones(size(XTest,1),1) XTest];                                      % including a first col with ones for the intercept (which is included within beta_mat)
           yhat(i,1) = XTest0*beta_mat(i,:)'; 
        
           %% ---------------------(2) NULL MODEL---------------------     
            %---------------BUILDING MODEL ON TRAINING DATA-------
        
           mdl_null = fitlm(XTrain_null,yTrain);
           beta_mat_null(i,:) = mdl_null.Coefficients.Estimate;
        
           %---------------------TEST DATA: PREDICTION ----------------
           XTest0_null = [ones(size(XTest_null,1),1) XTest_null];                
           yhat_null(i,1) = XTest0_null*beta_mat_null(i,:)';
end
      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                           PREDICTIVE PERFORMANCE METRICS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ---------------------(1) WORKING MODEL (GPM)--------------------- 
% correlation
[r_work,p_work] = corr(y, yhat,'rows','complete');    % permutations are needed to obtain p value - see additional script

% MSE
err = y-yhat;
MSE = 1/length(y)*(err'*err);

% alternative performance measures:
% RMSE=sqrt(MSE);
% NRMSE=RMSE/mean(y_nonan);

%% ---------------------(2) NULL MODEL--------------------- 
% correlation
[r_null,p_null] = corr(y, yhat_null,'rows','complete');   % permutations are needed to obtain p value - see additional script

% MSE
err_null = y-yhat_null;
MSE_null = 1/length(y)*(err_null'*err_null);

% alternative performance measures:
% RMSE_null=sqrt(MSE_null);
% NRMSE_null=RMSE_null/mean(y_nonan);

%% relative diff in MSE
MSE_diff = (MSE_null-MSE)/MSE_null;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                         FIGURES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Figure 1 - working model (GPM)
h1 = figure();
set(h1,'color','w');

hold on
scatter(y,yhat,'Marker','o','MarkerEdgeColor','k','LineWidth',2)

%Plot a least-squared fitted line
P = polyfit(y,yhat,1);                             % 1st order polynomial (linear)
Yfit = P(1)*y+P(2);                                % y-values of the fitted line
plot(y,Yfit,'k','LineWidth',2); 

legend ('off')
box off
xlabel('Observed scores','FontWeight','bold','FontName','Arial')
ylabel('Predicted scores','FontWeight','bold','FontName','Arial')
ax = gca; % current axes
ax.FontSize = 10;
ax.FontName = 'Arial';
ax.FontWeight = 'bold';
ax.TickDir = 'out';
ax.TickLength = [0.024 0.024];

%% Figure 2 - null model
h2 = figure();
set(h2,'color','w');

hold on
scatter(y,yhat_null,'Marker','o','MarkerEdgeColor','k','LineWidth',2)

%Plot a least-squared fitted line
P = polyfit(y,yhat_null,1);                             % 1st order polynomial (linear)
Yfit = P(1)*y+P(2);                                        % y-values of the fitted line
plot(y,Yfit,'k','LineWidth',2); 

legend ('off')
box off
xlabel('Observed scores','FontWeight','bold','FontName','Arial')
ylabel('Predicted scores','FontWeight','bold','FontName','Arial')
ax = gca; % current axes
ax.FontSize = 10;
ax.FontName = 'Arial';
ax.FontWeight = 'bold';
ax.TickDir = 'out';
ax.TickLength = [0.024 0.024];