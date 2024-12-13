
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        BRAIN-BASED GRAPH-THEORETICAL PREDICTIVE MODELING (GPM) - UTILIZING ELASTIC NET
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
%   * Note: this version of the GPM uses elastic-net regularization to define brain predictors out of the total set of graph-theoretical measures.
%           Glmnet package was used to implement elastic net. 
%           Download: https://hastie.su.domains/glmnet_matlab/intro.html. Details: https://glmnet.stanford.edu/index.html
%           CVGLMNET: this function was used to run elastic net.
%   
%   References:
%           If you use this code, please cite:
%           Dan, R., Whitton, A.E., Treadway, M.T. et al. Brain-based graph-theoretical predictive modeling to map the trajectory of anhedonia, impulsivity, and hypomania from the human functional connectome. 
%               Neuropsychopharmacol. 49, 1162–1170 (2024). https://doi.org/10.1038/s41386-024-01842-1
%
%           If using Glmnet, please cite:
%           Friedman J, Tibshirani R, Hastie T (2010). “Regularization Paths for Generalized Linear Models via Coordinate Descent.” Journal of Statistical Software, 33(1), 1–22. doi:10.18637/jss.v033.i01.
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

addpath(genpath('path/to/Glmnet'));           % adding Glmnet folder to run elastic net

for rep = 1:1000                                            % 10-fold CV is repeated 1000 for stability
                fprintf('\n rep %d', rep); 
        
                k =10;                                           % number of folds
                n = length(y);
                c = cvpartition(n,'KFold',k);          % partitioning into 10 folds
        
                for i = 1:k
                            fprintf('\n fold %d out of 10', i); 
                
                            %% ---------------------(1) WORKING MODEL (GPM)---------------------   
                            idxTrain = training(c,i);
                            idxTest = ~idxTrain;
                            XTrain = X(idxTrain,:);
                            yTrain = y(idxTrain);
                            XTest = X(idxTest,:);
                            yTest = y(idxTest);   
                
                            %---------------------ELASTIC NET: OPTIMIZING HYPER-PARAMETERS---------------------------                           
                            fprintf('\n working model');
                
                            % LOOPING ON ALPHA, 100 LAMBDA ARE TESTED FOR EACH ALPHA
                            % lambda is selected by cv, for each alpha
                            run = 1;
                            alpha_vec = 0:0.05:1;                                                                                         % 20 alpha values
                
                            %foldid =
                            %crossvalind('Kfold',length(yTrain),10);                                                             % crossvalind: uses bioinformatics toolbox. if not available, use the code below instead
                            nTrain = length(yTrain);
                            fold_indices = repmat(1:k,1,ceil(nTrain/k));        
                            foldid = fold_indices(randperm(nTrain));
                
                            for alpha = alpha_vec
                                            fprintf('\n alpha %d', alpha);
                            
                                            opts=struct; opts.alpha=alpha; options=glmnetSet(opts);               
                            
                                            CVerr = cvglmnet(XTrain,yTrain,'gaussian',options,[],[],foldid);          % runs elastic net
                                            %run several times, each time with a different alpha (sent in the options above). foldid: the fold partitioning, to keep the same partitioning for every alpha value tested.
                                                                     
                                            Lambda_vec(run) = CVerr.lambda_min;                                               % lambda that gives minimum cvm
                                            Lind = find(CVerr.lambda==CVerr.lambda_min);
                                            cvm_vec(run) = CVerr.cvm(Lind);                                                         % the mean cross-validated error
                                            coef_mat(run,:) = cvglmnetCoef(CVerr, 'lambda_min');                       % coefficients for minimum lambda
                                            run = run+1; 
                            end
                
                            %---------------------SELECT MODEL WITH BEST ALPHA-LAMBDA PAIR---------------------------                            
                            [best_cvm, best_ind] = min(cvm_vec);
                            best_coef = coef_mat(best_ind,:);    
                            %best_alpha(i,rep) = alpha_vec(best_ind);                                                        % optional: saving chosen alpha and lambda for every fold and repetition
                            %best_lambda(i,rep) = Lambda_vec(best_ind);
                        
                           %---------------------TEST DATA: PREDICTION ----------------  
                            XTest0 = [ones(size(XTest,1),1) XTest];                                                               % including a first col with ones for the intercept (which is included within beta_mat)
                            yhat_gpm = XTest0*best_coef';

                           err = yTest-yhat_gpm;
                           MSE_gpm(i,rep) = 1/length(yTest)*(err'*err);                                                       % MSE per fold and repetition

                           % yhat_all, ytest_all: for computing correlation across all folds (per repetition)
                           if i==1
                               yhat_all=yhat{i};
                               yTest_all = yTest;
                           else
                              yhat_all = [yhat_all; yhat{i}];     
                              yTest_all = [yTest_all;  yTest];
                           end 
                        
                           %% ---------------------(2) NULL MODEL---------------------     
                            fprintf('\n null model');
                            
                            XTrain_null = X_null(idxTrain,:);
                            XTest_null = X_null(idxTest,:);
                
                            %---------------------ELASTIC NET: OPTIMIZING HYPER-PARAMETERS---------------------------
                            %%
                            run = 1;    
                            for alpha = alpha_vec
                                        fprintf('\n alpha %d', alpha);
                        
                                        opts=struct; opts.alpha=alpha; options=glmnetSet(opts);          
                        
                                        CVerr_null = cvglmnet(XTrain_null,yTrain,'gaussian',options,[],[],foldid); 
                        
                                        Lambda_vec_null(run) = CVerr_null.lambda_min;                      
                                        Lind_null = find(CVerr_null.lambda==CVerr_null.lambda_min);
                                        cvm_vec_null(run) = CVerr_null.cvm(Lind_null);                      
                                        coef_mat_null(run,:) = cvglmnetCoef(CVerr_null, 'lambda_min');     
                                        run = run+1;
                            end
                
                            %---------------------SELECT MODEL WITH BEST ALPHA-LAMBDA PAIR---------------------------
                            [best_cvm_null, best_ind_null] = min(cvm_vec_null);
                            best_coef_null = coef_mat_null(best_ind_null,:);    
                            %best_alpha_null(i,rep) = alpha_vec(best_ind_null);
                            %best_lambda_null(i,rep) = Lambda_vec_null(best_ind_null);

                           %---------------------TEST DATA: PREDICTION ----------------
                            XTest0_null = [ones(size(XTest_null,1),1) XTest_null];                                        
                            yhat_null = XTest0_null*best_coef_null';

                           err_null = yTest-yhat_null;
                           MSE_null(i,rep) = 1/length(yTest)*(err_null'*err_null);                                            % MSE per fold and repetition

                           % yhat_all_null: for computing correlation across all folds (per repetition)
                           if i==1
                                yhat_all_null=yhat_null{i};
                           else
                                yhat_all_null = [yhat_all_null; yhat_null{i}];     
                           end
        
                end    %end of 10-fold CV for a specific repetition

                %% ---------------------Predictive Performance Metrics---------------------
                % correlation: per repetition
                [r_work(rep),p_work(rep)]=corr(yTest_all, yhat_all,'rows','complete');           % working model (GPM). Correlation value can be used, p value should be computed through permutation testing
                [r_null(rep),p_null(rep)]=corr(yTest_all, yhat_all_null,'rows','complete');       % null model.  Correlation value can be used, p value should be computed through permutation testing

                %MSE
                %saved above within the loop, for every fold and repetition

end         % end of repetitions
