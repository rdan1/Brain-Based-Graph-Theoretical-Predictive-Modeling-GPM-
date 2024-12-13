function [r_work, r_null] = GPM_predict(all_phenotype, all_graph, all_covariate1, all_covariate2)

%% ---------------------PREDICTORS AND OUTCOME--------------------------------
% null model: includes only covariates (i.e., all predictors except for graph-theoretical measures)

y = all_phenotype;                                                              % outcome. dimensions: subj X 1
X_null = [all_covariate1 all_covariate2];                              % covariates. dimensions: subj X variables
        
%% ---------------LEAVE ONE OUT CROSS VALIDATION (LOOCV)-------------------- 
% In each iteration, the data is divided into a training set and a testing set. Model is built on training set and prediction is done for testing set.
% see alternative script for 10-fold CV (recommended).
n = length(y);
num_graph = size(all_graph,1);

for i = 1:n
            fprintf ('\n Leaving out subj #%4.0f', i);     
        
            XTrain_null = X_null;
            XTrain_null(i,:) = [];                 % Train set: N-1 (excluding one participant)
            XTest_null = X_null(i,:);           % Test set: the participant who was excluded from model training
        
            train_graph = all_graph;
            train_graph(:,i) = [];                % Train set: N-1 (excluding one participant)
            test_graph = all_graph(:,i);     % Test set: the participant who was excluded from model training
        
            yTrain = y;
            yTrain(i,:) = [];                          % Train set: N-1 (excluding one participant)
            yTest = y(i,:);                            % Test set: the participant who was excluded from model training
        
           %% ---------------(1) WORKING MODEL---------------------   
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
            best_graph_vec_p(i) = best_graph_index;
        
           % step 3: saving model coefficients (betas)
           beta_mat(i,:) = mdl{best_graph_index,1}.Coefficients.Estimate;    % model coefficients.
           p_coef(i,:) = mdl{best_graph_index,1}.Coefficients.pValue;           % use this to examine which predictors were significant
        
           %---------------------TEST DATA: PREDICTION ----------------  
           XTest = [test_graph(best_graph_index,:)' XTest_null];
           XTest0 = [ones(size(XTest,1),1) XTest];                                          % including a first col with ones for the intercept (which is included within beta_mat)
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

%% ---------------------(1) WORKING MODEL--------------------- 
[r_work,p_work] = corr(y, yhat,'rows','complete');                                        %only r is needed, p value is not used (this is the one without permutations)

%% ---------------------(2) NULL MODEL--------------------- 
[r_null,p_null] = corr(y, yhat_null,'rows','complete');                                     %only r is needed, p value is not used (this is the one without permutations)

end