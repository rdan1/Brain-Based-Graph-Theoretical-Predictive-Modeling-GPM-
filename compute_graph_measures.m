%%-----------------------------------------------------------
%% Computing graph theoretical measures: from weighted and positive FC matrices
%%-----------------------------------------------------------
% Most graph theoretical measures can be computed only for positive matrices. Thus, first step is thresholding the functional connectivity matrix at 0, to remove negative weights.

data_path = 'path/to/FC/matrix';  %path to weighted and signed functional connectivity matrix - Z
load(fullfile(data_path, 'Z'));
ind = find(Z<0);                            %indices of negative connections
Z_pos = Z;
Z_pos(ind) = 0;                             %positive FC matrix (Z_pos): after setting negative FC to zeros

save(fullfile(data_path,'Z_pos.mat'),'Z_pos');

%% Global and local graph measures
% Graph theoretical measures are computed using the Brain Connectivity Toolbox (BCT). 
% Download the BCT from here: https://sites.google.com/site/bctnet/
% Citation: Complex network measures of brain connectivity: Uses and interpretations. Rubinov M, Sporns O (2010) NeuroImage 52:1059-69.

addpath(genpath('path/to/BCT'))
load(fullfile(data_path,'Z_pos.mat'));

results_path = 'path/to/results';
mkdir(results_path);

nsub = size(Z_pos,3);     %number of subjects
nROI = size(Z_pos,1);     %number of ROIs

Z_nrm=zeros(nROI,nROI,nsub); Cl=zeros(nROI,nsub); T=zeros(1,nsub); Eglob=zeros(1,nsub); Eloc=zeros(nROI,nsub); CPath=zeros(1,nsub); BC=zeros(nROI,nsub); strength=zeros(nROI,nsub);

for i = 1:nsub
      
   % (1) clustering coefficient
   % All weights must be between 0 and 1.
   Z_nrm(:,:,i) = weight_conversion(Z_pos(:,:,i), 'normalize'); %normalizes the weights
   Cl(:,i) = clustering_coef_wu(Z_nrm(:,:,i));
   
   % (2) transitivity
   % All weights must be between 0 and 1.
   T(i) = transitivity_wu(Z_nrm(:,:,i));
   
   % (3) global efficiency
    Eglob(i) = efficiency_wei(Z_pos(:,:,i));
   
   % (4) local efficiency
    Eloc(:,i) = efficiency_wei(Z_pos(:,:,i),2);

    % (5) characteristic path length
    %computing connection-length matrix L: defined as L_ij = 1/W_ij for all nonzero L_ij. W=weight matrix 
    %(computation and definition taken from efficiency_wei.m)
    W = Z_pos(:,:,i);
    L = W;                                                         % connection-length matrix
    A = W > 0;                                                  % adjacency matrix-0 or 1
    L(A) = 1 ./ L(A);                                           % computing L for all nonzero elements
    
    %computing distance matrix (D). The input distance matrix (D) may be obtained with any of the distance functions, e.g. distance_wei
    D = distance_wei(L);        
    CPath(i) = charpath(D);
    
    %(6) betweenes centrality
    %connection-length matrix L: computed above
    BC(:,i) = betweenness_wei(L);

    % (7) Node strength (weighted degree)
    % node strength is the weighted version of node degree
    strength(:,i) = strengths_und(Z_pos(:,:,i));
       
end

%% saving variables
save(fullfile(results_path,'Cl.mat'),'Cl'); save(fullfile(results_path,'T.mat'),'T'); save(fullfile(results_path,'Eglob.mat'),'Eglob'); save(fullfile(results_path,'Eloc.mat'),'Eloc'); save(fullfile(results_path,'CPath.mat'),'CPath'); save(fullfile(results_path,'BC.mat'),'BC'); save(fullfile(results_path,'strength.mat'),'strength');

%% average across nodes: mean clustering, mean local efficiency, mean node centrality, mean nodal strength
% additional possible global measures created by averaging across nodes.
% Note that transitivity is almost identical to the averaged clustering coefficient, thus should not include both measures

Mean_Cl = mean(Cl,1);
Mean_Eloc = mean(Eloc,1);
Mean_BC = mean(BC,1);
Mean_strength = mean(strength,1);

save(fullfile(results_path,'Mean_Cl.mat'),'Mean_Cl'); save(fullfile(results_path,'Mean_Eloc.mat'),'Mean_Eloc'); save(fullfile(results_path,'Mean_BC.mat'),'Mean_BC'); save(fullfile(results_path,'Mean_strength.mat'),'Mean_strength');

%% Constructing a matrix of graph measures to use in the predictive model
% Select measures to include within the predictive model. Matrix can include global measures as well as local ones. 
% For local measures, specific nodes can be selected for analysis (e.g., BC(82,:)).

all_graph = [measure1; measure2; ...];    % matrix including all graph measures for predictive modeling
save(fullfile(results_path,'all_graph.mat'),'all_graph');
