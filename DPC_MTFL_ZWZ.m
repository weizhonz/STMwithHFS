function [ Sol, ind_zf, tsol ] = DPC_MTFL_ZWZ( Xs, ys, Lambda, opts )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Changed from the function DPC_MTFL, the difference is that the input Xs and ys are cells instead of matrix and vector
%% Implementation of the sequential DPC rule proposed in
%  Safe Screening for Multi-Task Learning with Multiple Data Matrices
%  Jie Wang, Peter Wonka, and Jieping Ye. 
%
%% input: 
%         X: 
%            stacked data matrix, each column corresponds to a feature 
%            each row corresponds to a data instance
%
%         y: 
%            the response vector
%       
%         Lambda: 
%            the parameters sequence
%
%         opts: 
%            settings for the solver
%% output:
%         Sol: 
%              the solution; the ith column corresponds to the solution
%              with the ith parameter in Lambda
%
%         ind_zf: 
%              the index of the discarded features; the ith column
%              refers to the solution of the ith value in Lambda
%
%         t_sol: 
%              the ith entry of ts_sol records the running time of the
%              solver after screening
%% For any problem, please contact Jie Wang (jiewangustc@gmail.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = [];y = [];
for idx_t = 1:length(ys)
X = [X;Xs{idx_t}];
Xs{idx_t} = [];
y = [y;ys{idx_t}];
ys{idx_t}=[];
end
[N,d] = size(X);
npar = length(Lambda); % number of parameters

% ---------------------------- pass parameters -------------------------- %
opts.init = 0; % set .init for warm start
task_ind = opts.ind;
stask = diff(task_ind); % size of each task
T = length(stask); % number of tasks

% --------------------- initialize the output --------------------------- %
Sol = zeros(d,T,npar);
ind_zf = false(d,npar);
%rej_ratio = zeros(1,npar);

% ------- construct sparse matrix to vectorize the computation ---------- %
ft_ind = zeros(1,N);
for i = 1:T
    ft_ind(1,task_ind(i)+1:task_ind(i+1))=i; % fg_ind(j) is the index of the group 
                                     % which contains the jth feature
end
tS = sparse(ft_ind,1:N,ones(1,N),T,N,N); % ith row refers to the ith task, if the ith row
                   % refers to the i_s to i_e features, then
                   % tS(i,i_s:i_e) = 1.

% ------ compute the norm of each feature of each task ||x_l^t||_2 ---------- %  
Xtnorm = sqrt(tS*(X.*X));
H = -2*Xtnorm;
[xtnmx,xtnmxind] = max(Xtnorm);

% --------------------------- compute lambda_max ------------------------ %
Xtty = tS*(X.*repmat(y,1,d)); % Xtty is of T*d and Xtty(i,j) = <x_j^i,y_i>

[lambda_max2, indmx] = max(sum(Xtty.*Xtty,1));
lambda_max = sqrt(lambda_max2);
opts.lambda_max = lambda_max;

if opts.rFlag == 1
    opts.rFlag = 0; % the parameter value passing to the solver is its 
                       % absolute value rather than a ratio
    Lambda = Lambda*lambda_max;
end

% ----------------- sort the parameters in descend order ---------------- %
[Lambdav,Lambda_ind] = sort(Lambda,'descend');
rLambdav = 1./Lambdav;

% ----------------- solve MTFL sequentially with DPC ------------ %
lambdap = lambda_max;
tsol = zeros(1,npar);
for i = 1:npar
    fprintf('in DPC step: %d\n',i);
    lambdac = Lambdav(1,i);
    if lambdac>=lambda_max
        %Sol(:,Lambda_ind(i)) = 0; % because Sol is initalized to be 0
        ind_zf(:,Lambda_ind(i)) = true;     
        %rej_ratio(Lambda_ind(i)) = 1;
    else
        if lambdap==lambda_max
            theta = y/lambda_max;
            indmx = indmx(1);
            nv = (tS'*Xtty(:,indmx)).*X(:,indmx);
        else
            theta = (y - sum(X.*(Sol(:,:,Lambda_ind(i-1))*tS)',2))*rlambdap;
            nv = y*rlambdap-theta;
        end
        
        rlambdac = rLambdav(1,i);
        
        nv = nv/norm(nv);
        rv = y*rlambdac-theta;
        Prv = rv - (nv'*rv)*nv;
        o = theta + 0.5*Prv;
        r = 0.5*norm(Prv);

        % ------- screening by DPC, remove the ith feature if T(i)=1 ----- %
        Xtto = tS*(X.*repmat(o,1,d));
        q = -2*(Xtto.*Xtnorm);
        qpmx = -newton_qp(H,q,r,2*xtnmx,xtnmxind);
        Tmt = 1 > (sum(Xtto.*Xtto,1))' + qpmx + 1e-8;      
        ind_zf(:,Lambda_ind(i)) = Tmt;
        nTmt = ~Tmt;
        Xr = X(:,nTmt);
        
        if lambdap == lambda_max
            opts.x0 = zeros(size(Xr,2),T);
        else
            opts.x0 = Sol(nTmt,:,Lambda_ind(i-1));
        end
        
		% --- solve the group Lasso problem on the reduced data matrix -- %
        starts = tic;
        [x1, ~, ~]= mtLeastR(Xr, y, lambdac, opts);   
        tsol(Lambda_ind(i)) = toc(starts);
        
        Sol(nTmt,:,Lambda_ind(i)) = x1;
        
        lambdap = lambdac;
        rlambdap = rlambdac;
    end
end

end

