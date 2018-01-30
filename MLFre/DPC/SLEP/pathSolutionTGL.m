function [Sol,funVal,tsolver]=pathSolutionTGL(X, y, Lambda, opts)
%% solve the TGL problem for a sequence of parameter values

% -------------------------- pass parameters ---------------------------- %
npar = length(Lambda);
p=size(X,2);
tsolver = zeros(1,npar);
Sol = zeros(p,npar);
funVal = zeros(1,npar);
ind = opts.ind;

% ind1=[
%     [1, 2, sqrt(2)]', [1, 3, sqrt(3)]'
%     ]; % the higher layer
% 
% ind2=[
%     [1, 2, sqrt(2)]', [3,3,0]', [1, 3, sqrt(3)]'
%     ]; % the higher layer

% -------------------------- compute lambda ----------------------------- %
lambda_max=findLambdaMax(X'*y, p, ind, size(ind,2));
if opts.rFlag == 1
    Lambda = Lambda * lambda_max;
    opts.rFlag = 0;
end
[Lambdav,Lambda_ind] = sort(Lambda,'descend');

% -------------------- solve the problems sequentially ------------------ %
for i = 1:npar
    fprintf('in solver step: %d\n',i);
    if opts.init~=2
        if i == 1
            opts.x0=zeros(p,1);
        else
            opts.x0=x1;
        end
    end
  
    starts = tic;
    %[x1, funVal1, ValueL1]= tree_LeastR(X, y, Lambdav(i), opts);
    [x1, funVal1, ~]= tree_LeastR(X, y, Lambdav(i), opts);
    tsolver(Lambda_ind(i)) = toc(starts);
    funVal(Lambda_ind(i)) = funVal1(end);
    Sol(:,Lambda_ind(i))=x1;
end

end

