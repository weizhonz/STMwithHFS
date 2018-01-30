clear;clc;
addpath(genpath('./MLFre/DPC'));
addpath(genpath('./SLEP_package_4.1'));

%% Generate the synthetic data
SyntheticDatagenerator();


%% ---------------------- load the tree----------------------

groupInfo = load('SyntheticData/index_tree.mat');
nameVec = [1,2,3,4,5,6,7,8];

%% ----------------------- Set optional items -----------------------
opts=[];

% Starting point
opts.init=1;        % 2: starting from a zero point
% 1: warm start

% Termination
opts.tFlag=5;       % run .maxIter iterations
opts.maxIter=1000;   % maximum number of iterations

% regularization
opts.rFlag=1;       % 1: use ratio
% 0: use the true value
% Normalization
opts.nFlag=0;       % without normalization

% Group Property
newInd = groupInfo.index_tree;
opts.ind = newInd;

%% ----------------- set the parameter values ------------------
ub = 1;
lb = 0.1;
npar = 100;
scale = 'log';
Lambda = get_lambda(lb, ub, npar, scale);
d = 4;  



%% --------------------- experiments ----------------
ntrial = 8;
nSubsample = 100;
run_solver = 0;

npar = length(Lambda);
rej_ratio = zeros(d,npar);
run_time = [];
fold = 2;
X_all  = [];
y_all  = [];
for trial = 1:ntrial
    fileName = sprintf('SyntheticData/SyntheticData%d.mat',nameVec(trial));
    data = load(fileName);
    Xs{trial} = data.norm_input;
    ys{trial} = data.norm_res;
end
clear('data');

Xsize = length(ys{1});

for idx_subsample = 1:nSubsample
    indices = crossvalind('Kfold', Xsize, fold);
    for idx_t = 1: length(ys)
        subXs{idx_t} = zscore(Xs{idx_t}(indices == 1, :));
        subys{idx_t} = zscore(ys{idx_t}(indices ==1));
    end
    %% STM with Screening Method HFS
    fprintf('training STM in %d-th sampling ...\n', idx_subsample);
    [Sol_STM_HFS{idx_subsample}] = STM_HFS(subXs, subys, Lambda, opts);
end

save('results.mat','Sol_STM_HFS');






