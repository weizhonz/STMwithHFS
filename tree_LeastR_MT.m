function [x_MT, funVal, ValueL_MT]=tree_LeastR_MT(As, ys, z, opts)
%
%%
% Function tree_LeastR
%      Least Squares Loss with the
%           tree structured group Lasso Regularization in Multi-Task
%           Learning
%
%% Problem
%
%  min  1/2 sum_t || A_t x_t - y_t||^2 + z * sum_j w_j ||x_{G_j}||
%
%  G_j's are nodes with tree structure
%
%  The tree overlapping group information is contained in
%  opts.ind_MT, which is a 3 x nodes matrix, where nodes denotes the number of
%  nodes of the tree.
%
%  opts.ind_MT(1,:) contains the starting index
%  opts.ind_MT(2,:) contains the ending index
%  opts.ind_MT(3,:) contains the corresponding weight (w_j)
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
%  say, x( opts.ind_MT(1,j):opts.ind_MT(2,j) ) denotes x_{G_j}. In this case,
%  the entries in opts.ind(1:2,:) are within 1 and n.
%
%
%  If the features are not well ordered, please use the input opts.G for
%  specifying the index so that
%   x( opts.G_MT ( opts.ind_MT(1,j):opts.ind_MT(2,j) ) ) denotes x_{G_j}.
%  In this case, the entries of opts.G are within 1 and n, and the entries of
%  opts.ind_MT(1:2,:) are within 1 and length(opts.G_MT).
%
% The following example shows how G and ind works:
%
% G={ {1, 2}, {4, 5}, {3, 6}, {7, 8},
%     {1, 2, 3, 6}, {4, 5, 7, 8},
%     {1, 2, 3, 4, 5, 6, 7, 8} }.
%
% ind={ [1, 2, 100]', [3, 4, 100]', [5, 6, 100]', [7, 8, 100]',
%       [9, 12, 100]', [13, 16, 100]', [17, 24, 100]' },
%
% where each node has a weight of 100.
%
%
%% Input parameters:
%
%  As-         A set of Matrix of size m_t x n
%                At can be a dense matrix
%                         a sparse matrix
%                         or a DCT matrix
%  ys -        A set of Response vector (of size m_tx1)
%  z -        Tree Structure regularization parameter (z >=0)
%  opts-      optional inputs (default value: opts=[])
%
%% Output parameters:
%
%  xs-         A set of solution
%  funVal-    Function value during iterations
%
%% Copyright (C) 2010-2011 Jun Liu, and Jieping Ye
%
% You are suggested to first read the Manual.
%
% For any problem, please contact with Jun Liu via j.liu@asu.edu
%
% Last modified on October 3, 2010.
%
%% Related papers
%
% [1] Jun Liu and Jieping Ye, Moreau-Yosida Regularization for
%     Grouped Tree Structure Learning, NIPS 2010
%
%% Related functions:
%
%  sll_opts
%
%%

%% Verify and initialize the parameters
%%
if (nargin <4)
    error('\n Inputs: As, ys, z, and opts.ind_MT should be specified!\n');
end

[m_MT,n_MT]=size(As{1});
T_num = length(ys);

if (length(ys{1}) ~=m_MT)
    error('\n Check the length of ys!\n');
end

if (z<0)
    error('\n z should be nonnegative!\n');
end

opts=sll_opts(opts); % run sll_opts to set default values (flags) 

% restart the program for better efficiency
%  this is a newly added function
if (~isfield(opts,'rStartNum'))
    opts.rStartNum=opts.maxIter;
else
    if (opts.rStartNum<=0)
        opts.rStartNum=opts.maxIter;
    end
end

%% Detailed initialization
%% Normalization

% Please refer to sll_opts for the definitions of mu, nu and nFlag
%
% If .nFlag =1, the input matrix A is normalized to
%                     A= ( A- repmat(mu, m,1) ) * diag(nu)^{-1}
%
% If .nFlag =2, the input matrix A is normalized to
%                     A= diag(nu)^{-1} * ( A- repmat(mu, m,1) )
%
% Such normalization is done implicitly
%     This implicit normalization is suggested for the sparse matrix
%                                    but not for the dense matrix
%



if (~issparse(As{1})) && (opts.nFlag~=0)
    fprintf('\n -----------------------------------------------------');
    fprintf('\n The data is not sparse or not stored in sparse format');
    fprintf('\n The code still works.');
    fprintf('\n But we suggest you to normalize the data directly,');
    fprintf('\n for achieving better efficiency.');
    fprintf('\n -----------------------------------------------------');
end

%% Group & Others

% Initialize ind
if (~isfield(opts,'ind_MT'))
    error('\n In tree_LeastR, the field .ind should be specified');
else
    ind_MT=opts.ind_MT;
    
    if (size(ind_MT,1)~=3)
        error('\n Check opts.ind');
    end
end

GFlag_MT=0;
% if GFlag=1, we shall apply general_altra
if (isfield(opts,'G_MT'))
    GFlag_MT=1;
    
    G_MT=opts.G_MT;
    if (max(G_MT) >n_MT*T_num|| max(G_MT) <1)
        error('\n The input G is incorrect. It should be within %d and %d',1,n);
    end
end

%% Starting point initialization
% compute AT y
if (opts.nFlag==0)
    %ATy =A'*y;
    AsTys_MT = Xsys_MT_cal(As, ys);
end

% process the regularization parameter
if (opts.rFlag==0)
    lambda=z;
else % z here is the scaling factor lying in [0,1]
    %     if (z<0 || z>1)
    %         error('\n opts.rFlag=1, and z should be in [0,1]');
    %     end
    
    %     computedFlag=0;
    %     if (isfield(opts,'lambdaMax'))
    %         if (opts.lambdaMax~=-1)
    %             lambda=z*opts.lambdaMax;
    %             computedFlag=1;
    %         end
    %     end
    %
    %     if (~computedFlag)
    %         if (GFlag==0)
    %             lambda_max=findLambdaMax(ATy, n, ind, size(ind,2));
    %         else
    %             lambda_max=general_findLambdaMax(ATy, n, G, ind, size(ind,2));
    %         end
    %
    %         % As .rFlag=1, we set lambda as a ratio of lambda_max
    %         lambda=z*lambda_max;
    %     end
end


% The following is for computing lambdaMax
% we use opts.lambdaMax=-1 to show that we need the computation.
%
% One can use this for setting up opts.lambdaMax
% if (isfield(opts,'lambdaMax'))
%
%     if (opts.lambdaMax==-1)
%         if (GFlag==0)
%             lambda_max=findLambdaMax(ATy, n, ind, size(ind,2));
%         else
%             lambda_max=general_findLambdaMax(ATy, n, G, ind, size(ind,2));
%         end
%
%         x=lambda_max;
%         funVal=lambda_max;
%         ValueL=lambda_max;
%
%         return;
%     end
% end


% initialize a starting point
if opts.init==2
    x_MT=zeros(n_MT*T_num,1);
else
    if isfield(opts,'x0')
        x_MT=opts.x0;
        if (length(x_MT)~=n_MT*T_num)
            error('\n Check the input .x0');
        end
    else
        x_MT=AsTys_MT;  % if .x0 is not specified, we use ratio*ATy,
        % where ratio is a positive value
    end
end

% compute A x
if (opts.nFlag==0)
    Ax_MT = [];
    idx_row = [1:T_num:length(x_MT)];
    for idx_t = 1:T_num
        Ax_MT = [Ax_MT;As{idx_t}*x_MT(idx_row)];
        idx_row = idx_row +1;
    end
    %Ax=A* x;
    % elseif (opts.nFlag==1)
    %     invNu=x./nu; mu_invNu=mu * invNu;
    %     Ax=A*invNu -repmat(mu_invNu, m, 1);
    % else
    %     Ax=A*x-repmat(mu*x, m, 1);     Ax=Ax./nu;
end

if (opts.init==0)
    % ------  This function is not available
    %
    % If .init=0, we set x=ratio*x by "initFactor"
    % Please refer to the function initFactor for detail
    %
    
    % Here, we only support starting from zero, due to the complex tree
    % structure
    
    x_MT=zeros(n_MT*T_num,1);
end

%% The main program
% The Armijo Goldstein line search schemes + accelearted gradient descent

bFlag_MT=0; % this flag tests whether the gradient step only changes a little

if (opts.mFlag==0 && opts.lFlag==0)
    L=1;
    % We assume that the maximum eigenvalue of A'A is over 1
    
    % assign xp with x, and Axp with Ax
    xp_MT=x_MT; Axp_MT=Ax_MT; xxp_MT=zeros(n_MT*T_num,1);
    
    alphap=0; alpha=1;
    
    for iterStep=1:opts.maxIter
        % --------------------------- step 1 ---------------------------
        % compute search point s based on xp and x (with beta)
        beta=(alphap-1)/alpha;    s_MT=x_MT + beta* xxp_MT;
        
        % --------------------------- step 2 ---------------------------
        % line search for L and compute the new approximate solution x
        
        % compute the gradient (g) at s
        As_MT=Ax_MT + beta* (Ax_MT-Axp_MT);
        
        % compute AT As
        if (opts.nFlag==0)
            
            ATAs_MT = zeros(n_MT*T_num,1);
            idx_row = [0:n_MT-1]*T_num+1;
            for idx_t = 1:T_num
                ATAs_MT(idx_row) = As{idx_t}'*As_MT((idx_t-1)*length(ys{1})+1:idx_t*length(ys{1}));
                idx_row = idx_row+1;
            end
            %ATAs=A'*As;
            %         elseif (opts.nFlag==1)
            %             ATAs=A'*As - sum(As) * mu';  ATAs=ATAs./nu;
            %         else
            %             invNu=As./nu;                ATAs=A'*invNu-sum(invNu)*mu';
        end
        
        % obtain the gradient g
        g_MT=ATAs_MT-AsTys_MT;
        
        % copy x and Ax to xp and Axp
        xp_MT=x_MT;    Axp_MT=Ax_MT;
        
        while (1)
            % let s walk in a step in the antigradient of s to get v
            % and then do the L1/Lq-norm regularized projection
            v_MT=s_MT-g_MT/L;
            
            % tree overlapping group Lasso projection
            ind_work_MT(1:2,:)=ind_MT(1:2,:);
            ind_work_MT(3,:)=ind_MT(3,:) * (lambda / L);
            
            if (GFlag_MT==0)
                x_MT=altra(v_MT, n_MT*T_num, ind_work_MT, size(ind_work_MT,2));
            else
                x_MT=general_altra(v_MT, n_MT*T_num, G_MT, ind_work_MT, size(ind_work_MT,2));
            end
            
            v_MT=x_MT-s_MT;  % the difference between the new approximate solution x
            % and the search point s
            
            % compute A x
            if (opts.nFlag==0)
                %Ax=A* x;
                Ax_MT = [];
                idx_row = [1:T_num:n_MT*T_num];
                for idx_t = 1:T_num
                    Ax_MT = [Ax_MT;As{idx_t}*x_MT(idx_row)];
                    idx_row = idx_row +1;
                end
%             elseif (opts.nFlag==1)
%                 invNu=x./nu; mu_invNu=mu * invNu;
%                 Ax=A*invNu -repmat(mu_invNu, m, 1);
%             else
%                 Ax=A*x-repmat(mu*x, m, 1);     Ax=Ax./nu;
            end
            
            Av_MT=Ax_MT -As_MT;
            r_sum_MT=v_MT'*v_MT; l_sum_MT=Av_MT'*Av_MT;
            
            if (r_sum_MT <=1e-20)
                bFlag_MT=1; % this shows that, the gradient step makes little improvement
                break;
            end
            
            % the condition is ||Av||_2^2 <= L * ||v||_2^2
            if(l_sum_MT <= r_sum_MT * L)
                break;
            else
                L=max(2*L, l_sum_MT/r_sum_MT);
                %fprintf('\n L=%5.6f',L);
            end
        end
        
        % --------------------------- step 3 ---------------------------
        % update alpha and alphap, and check whether converge
        alphap=alpha; alpha= (1+ sqrt(4*alpha*alpha +1))/2;
        
        xxp_MT=x_MT-xp_MT;   
        y_all = [];
        for idx_t = 1:T_num
        y_all = [y_all;ys{idx_t}];
        end
        Axy_MT=Ax_MT-y_all;
        
        ValueL_MT(iterStep)=L;
        
        % compute the regularization part
        if (GFlag_MT==0)
            tree_norm_MT=treeNorm(x_MT, n_MT*T_num, ind_MT, size(ind_MT,2));
        else
            tree_norm_MT=general_treeNorm(x_MT, n_MT*T_num, G_MT, ind_MT, size(ind_MT,2));
        end
        
        % function value = loss + regularizatioin
        funVal(iterStep)=Axy_MT'* Axy_MT/2 + lambda * tree_norm_MT;
        
        if (bFlag_MT)
            % fprintf('\n The program terminates as the gradient step changes the solution very small.');
            break;
        end
        
        switch(opts.tFlag)
            case 0
                if iterStep>=2
                    if (abs( funVal(iterStep) - funVal(iterStep-1) ) <= opts.tol)
                        break;
                    end
                end
            case 1
                if iterStep>=2
                    if (abs( funVal(iterStep) - funVal(iterStep-1) ) <=...
                            opts.tol* funVal(iterStep-1))
                        break;
                    end
                end
            case 2
                if ( funVal(iterStep)<= opts.tol)
                    break;
                end
            case 3
                norm_xxp_MT=sqrt(xxp_MT'*xxp_MT);
                if ( norm_xxp_MT <=opts.tol)
                    break;
                end
            case 4
                norm_xp_MT=sqrt(xp_MT'*xp_MT);    norm_xxp_MT=sqrt(xxp_MT'*xxp_MT);
                if ( norm_xxp_MT <=opts.tol * max(norm_xp_MT,1))
                    break;
                end
            case 5
                if iterStep>=opts.maxIter
                    break;
                end
        end
        
        % restart the program every opts.rStartNum
        if (~mod(iterStep, opts.rStartNum))
            alphap=0; alpha=1;
            xp_MT=x_MT; Axp_MT=Ax_MT; xxp_MT=zeros(n_MT*T_num,1); L =L/2;
        end
    end
else
    error('\n The function does not support opts.mFlag neq 0 & opts.lFlag neq 0!');
    endfunction [ output_args ] = tree_LeastR_MT( input_args )
    %TREE_LEASTR_MT Summary of this function goes here
    %   Detailed explanation goes here
    
    
end

