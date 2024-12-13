
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

num_graph = size(all_graph,1);                                          % number of graph measures

%% ---------------REPEATED 10-FOLD CROSS-VALIDATION -------------------- 
% In each iteration, the data is divided into a training set and a testing set. Model is built on the training set and prediction is done for the testing set.
% 10-fold CV is repeated to obtain a more robust measure of predictive performance. In each repetition, there is a different random data splitting into folds

for rep = 1:1000                                    % 10-fold CV is repeated 1000 for stability
                fprintf('\n rep %d', rep); 
        
                k = 10;                                          % number of folds
                n = length(y);
                c = cvpartition(n,'KFold',k);          % partitioning into folds
        
                for i = 1:k
                            fprintf('\n fold %d out of 10', i); 
                
                            idxTrain = training(c,i);
                            idxTest = ~idxTrain; 
                
                            XTrain_null = X_null(idxTrain,:);
                            XTest_null = X_null(idxTest,:);
                        
                            yTrain = y(idxTrain);
                            yTest = y(idxTest);                                               
                             
                           train_graph = all_graph(:,idxTrain);
                           test_graph = all_graph(:,idxTest);
                        
                           %% ---------------(1) WORKING MODEL (GPM)---------------------   
                           %---------------BUILDING MODEL ON TRAINING DATA-------
                           % step 1: (internal nested loop) multiple regression model for each graph measure. Model can include interactions between graph metric and covariate (e.g., sex, group).
                           for j = 1:num_graph
                                    XTrain = [train_graph(j,:)' XTrain_null];
                                    mdl{j} = fitlm(XTrain,yTrain);                 % can include here interactions; defining categorical variables (e.g., group, sex)
                                    anv{j} = anova(mdl{j,1}, 'summary');
                                    p(j,1) = anv{j,1}.pValue(2);                    % model's p value               
                                    %MSE(j,1) = mdl{j,1}.MSE;                   % alternative criterion to lowest p value: selecting model with lowest MSE. Equivalent.
                           end
                        
                           % step 2: selecting model with lowest p value
                            [~, best_graph_index] = min(p);
                        
                           % step 3: saving model coefficients (betas)
                           beta_gpm = mdl{best_graph_index,1}.Coefficients.Estimate;   % model coefficients.
                        
                           %---------------------TEST DATA: PREDICTION ----------------  
                           XTest = [test_graph(best_graph_index,:)' XTest_null];
                           XTest0 = [ones(size(XTest,1),1) XTest];                                      % including a first col with ones for the intercept (which is included within beta_mat)
                           yhat_gpm = XTest0*beta_gpm'; 

                           err = yTest-yhat_gpm;
                           MSE_gpm(i,rep) = 1/length(yTest)*(err'*err);                               % MSE per fold and repetition

                           % yhat_all, ytest_all: for computing correlation across all folds (per repetition)
                           if i==1
                               yhat_all=yhat{i};
                               yTest_all = yTest;
                           else
                              yhat_all = [yhat_all; yhat{i}];     
                              yTest_all = [yTest_all;  yTest];
                           end 
                        
                           %% ---------------------(2) NULL MODEL---------------------     
                            %---------------BUILDING MODEL ON TRAINING DATA-------                        
                           mdl_null = fitlm(XTrain_null,yTrain);
                           beta_null = mdl_null.Coefficients.Estimate;
                        
                           %---------------------TEST DATA: PREDICTION ----------------
                           XTest0_null = [ones(size(XTest_null,1),1) XTest_null];                
                           yhat_null = XTest0_null*beta_null';

                           err_null = yTest-yhat_null;
                           MSE_null(i,rep) = 1/length(yTest)*(err_null'*err_null);                % MSE per fold and repetition

                           % yhat_all_null: for computing correlation across all folds (per repetition)
                           if i==1
                                yhat_all_null=yhat_null{i};
                           else
                                yhat_all_null = [yhat_all_null; yhat_null{i}];     
                           end
        
                end    % end of 10-fold CV for a specific repetition
                
                %% ---------------------Predictive Performance Metrics---------------------
                % correlation: per repetition
                [r_work(rep),p_work(rep)]=corr(yTest_all, yhat_all,'rows','complete');          % working model (GPM). Correlation value can be used, p value should be computed through permutation testing
                [r_null(rep),p_null(rep)]=corr(yTest_all, yhat_all_null,'rows','complete');       % null model.  Correlation value can be used, p value should be computed through permutation testing

                %MSE
                %saved above within the loop, for every fold and repetition

end         % end of repetitions
