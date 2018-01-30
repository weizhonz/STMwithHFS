clear;
addpath(genpath([pwd '/DPC']));

% This is an example for running the function tree_LeastR
%
%  Problem:
%
%  min  1/2 || A x - y||^2 + z * sum_j w_j ||x_{G_j}||
%
%  G_j's are nodes with tree structure
%
%  The tree structured group information is contained in
%  opts.ind, which is a 3 x nodes matrix, where nodes denotes the number of
%  nodes of the tree.
%
%  opts.ind(1,:) contains the starting index
%  opts.ind(2,:) contains the ending index
%  opts.ind(3,:) contains the corresponding weight (w_j)
%
%  Note: 
%  1) If each element of x is a leaf node of the tree and the weight for
%  this leaf node are the same, we provide an alternative "efficient" input
%  for this kind of node, by creating a "super node" with 
%  opts.ind(1,1)=-1; opts.ind(2,1)=-1; and opts.ind(3,1)=the common weight.
%
%  2) If the features are well ordered in that, the features of the left
%  tree is always less than those of the right tree, opts.ind(1,:) and
%  opts.ind(2,:) contain the "real" starting and ending indices. That is to
%  say, x( opts.ind(1,j):opts.ind(2,j) ) denotes x_{G_j}. In this case,
%  the entries in opts.ind(1:2,:) are within 1 and n.
%
%
%  If the features are not well ordered, please use the input opts.G for
%  specifying the index so that  
%   x( opts.G ( opts.ind(1,j):opts.ind(2,j) ) ) denotes x_{G_j}.
%  In this case, the entries of opts.G are within 1 and n, and the entries of
%  opts.ind(1:2,:) are within 1 and length(opts.G).
%
%% Related papers
%
% [1] Jun Liu and Jieping Ye, Moreau-Yosida Regularization for 
%     Grouped Tree Structure Learning, NIPS 2010
%
%% load data

%profile on; 

rng(1)
% ---------------------- generate random data ----------------------

N = 250;
p = 10000;
d = 3; % the depth of the tree excluding the root node
nns = [1,10,50,p]; % the size of each node at different levels
ratio = [1,0.2,0.5];

% data_name = ['syn1_' num2str(N) '_' num2str(p)];
% [ X, y, ind ] = gen_sync1( p, N, d, nns, ratio );

data_name = ['syn2_' num2str(N) '_' num2str(p)];
[ X, y, beta, ind ] = gen_sync2( p, N, d, nns, ratio );


%% In this example, the tree is set as:
%
% root, 1:100, with weight 0
% its children nodes, 1:50, and 51:100
%
% For 1:50, its children are 1:20, 21:40, and 41:50
%
% For 51:100, its children are 51:70, and 71:100
%
% These nodes in addition have each individual features (they contain) as
% children nodes.
%
%%

%% One efficient way
% We make use of the fact that the indices of the left nodes of the tree
% are smaller than the right nodes.

%----------------------- Set optional items -----------------------
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
opts.ind = ind;
% opts.ind=[[-1, -1, 1]',... % leave nodes (each node contains one feature)
%     [1, 20, sqrt(20)]', [21, 40, sqrt(20)]',... % the layer above the leaf
%     [41, 50, sqrt(10)]', [51, 70, sqrt(20)]', [71,90, sqrt(20)]', [91,100, sqrt(10)]',...
%     [1, 50, sqrt(50)]', [51, 100, sqrt(50)]']; % the higher layer
% 
% opts.depth = d;

% ----------------- set the parameter values ------------------
ub = 1;
lb = 0.05;
npar = 100;
scale = 'log';
Lambda = get_lambda(lb, ub, npar, scale);    

%----------------------- Run the code tree_LeastR -----------------------

% --------------------- experiments ----------------
npar = length(Lambda);
rej_ratio = zeros(d,npar);
run_solver = 1;
run_time = [];
ntrial = 1;

fprintf('data: %s:\n\n',data_name);
for trial = 1:ntrial
    fprintf('\ntrial %d:\n\n',trial);
    %X = data_matrix;
    %y = response;
    % run the solver without screening
    if run_solver == 1
        if trial == 1
            run_time.solver = 0;
            run_time.solver_step = zeros(1,npar);
        end
        fprintf('computing the ground truth by solver...\n');
        starts = tic;
        [Sol_GT,funVal,tsol]=pathSolutionTGL(X, y, Lambda, opts); % ground truth solution
        tsolver = toc(starts);
        run_time.solver=run_time.solver + tsolver;
        run_time.solver_step = run_time.solver_step + tsol;
        fprintf('Running Time of solver without MLFre: %f\n',tsolver);
        fprintf('Effective number of parameter values: %f\n',nnz(sum(Sol_GT.*Sol_GT)>=1e-20));
    end

    if opts.tFlag == 2
        opts.funVal = funVal;
    end
    
    % Compute the solution with screening MLFre
    if trial == 1
        run_time.MLFre_solver = 0;
        run_time.solver_MLFre_step = zeros(1,npar);
        run_time.MLFre = 0;
    end
    fprintf('computing the solution with screening MLFre...\n');
    starts = tic;
    [Sol_MLFre, ind_zf_MLFre, ts_MLFre] = MLFre_TGL(X, y, Lambda, opts);
    tMLFre_solver = toc(starts);
    tMLFre = tMLFre_solver - sum(ts_MLFre);
    run_time.MLFre_solver = run_time.MLFre_solver + tMLFre_solver;
    run_time.solver_MLFre_step = run_time.solver_MLFre_step + ts_MLFre;
    run_time.MLFre = run_time.MLFre + tMLFre;
    fprintf('Running time of MLFre: %f\n',tMLFre);
    fprintf('Running Time of solver with MLFre: %f\n',tMLFre_solver);

    % --------------------- compute the rejection ratio --------------------- %
    if run_solver == 1
        Sol = Sol_GT;
        fprintf('numerical error by MLFre: %f\n', ...
            norm(Sol_GT-Sol_MLFre,'fro'));
    else
        Sol = Sol_MLFre;
    end
    ind_zf = abs(Sol)<1e-12;
    
    rej_ratio = rej_ratio + reshape(sum(ind_zf_MLFre),d,npar)./repmat(sum(ind_zf),d,1);
    
end

if run_solver == 1
    run_time.solver = run_time.solver / ntrial;
    run_time.solver_step = run_time.solver_step / ntrial;
end
run_time.MLFre_solver = run_time.MLFre_solver / ntrial;
run_time.solver_MLFre_step = run_time.solver_MLFre_step / ntrial;
run_time.MLFre = run_time.MLFre / ntrial;
rej_ratio = rej_ratio / ntrial;

if run_solver == 1
    speedup = run_time.solver/run_time.MLFre_solver;
    fprintf('speedup: %f\n',speedup);
end

% ---------------------------- save the result ----------------------------
result_path = ['result/' data_name];
if ~exist(result_path,'dir')
    mkdir(result_path);
end
result_path = [result_path '/' scale];
if ~exist(result_path,'dir')
    mkdir(result_path);
end
result_path = [result_path '/'];
result_name = [data_name '_result_' scale];
if run_solver == 1
    save([result_path result_name],'data_name','Lambda','rej_ratio','run_time','speedup');
else
    save([result_path result_name],'data_name','Lambda','rej_ratio','run_time');
end
%% plot the rejection ratio

plot_rej_ratio(Lambda,rej_ratio,scale,data_name,result_path)
 
%profile viewer; 


