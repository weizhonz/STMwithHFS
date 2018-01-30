function [Sol, ind_zf, tsolver] = MLFre_TGL(X, y, Lambda, opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Implementation of the sequential TLFre rule proposed in
%  Jie Wang and Jieping Ye,
%  Multi-Layer Feature Reduction for Grouped
%  Tree Structure Learning via Hierarchical Projection,
%  
%
%% input: 
%         X: 
%            the data matrix, each column corresponds to a feature 
%            each row corresponds to a data instance
%
%         y: 
%            the response vector
%
%         Lambda: 
%            the parameter values of lambda
%
%         opts: 
%            settings for the solver
%% output:
%         Sol: 
%              the solution; Sol(:,i) stores the the 
%              solution for the ith values in Lambda 
%
%         ind_zf: 
%              a 3D tensor that stores the index of the discarded features 
%              by MLFre; ind_zf1(:,i,k) the index of 
%              discarded features corresponding to the ith layer at the kth 
%              values in Lambda 
%
%% For any problem, please contact Jie Wang (jiewangustc@gmail.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%

% -------------------------- pass parameters ---------------------------- %
p = size(X,2);
npar = length(Lambda); % number of parameter values of lambda
ind = opts.ind;
opts.init=1;        % starting from a zero point
if opts.tFlag==2
    funVal = opts.funVal;
end

% --------------------- recover tree structure -------------------------- %
eind = find(ind(2,:) == p);
if ind(1,1) == -1 % find the depth of the tree
    d = length(eind);
    nnl = [p,eind(1)-1,diff(eind)]; % number of nodes per layer
    nnind = [0,1,eind]; % ind(1,nnind1(i)+1:nnind(i+1)) stores the node from the ith layer
else
    d = length(eind)-1;
    nnl = [eind(1),diff(eind)];
    nnind = [0,eind];
end

% --------------------- initialize the output --------------------------- %
Sol = zeros(p,npar);
ind_zf = false(p,d,npar);
tsolver = zeros(1,npar);

% ------------------- compute the effective region of lambda ------------ %
lambda_max=findLambdaMax(X'*y, p, ind, size(ind,2));
if opts.rFlag == 1
    Lambda = Lambda * lambda_max;
    opts.rFlag = 0;
end
[Lambdav,Lambda_ind] = sort(Lambda,'descend');

% --------- compute the norm of each feature and each submatrix ---------
Xnorm = (sqrt(sum(X.^2,1)))';
ng = size(ind,2);
Xgnorm = zeros(1,ng);
gind = zeros(d,p);
if ind(1,1)==-1
    j = 2;
    l = 2;
else
    l = 1;
    j = 1;
end
k = 1;
for i = j : ng-1
    Xgnorm(i) = norm(X(:,ind(1,i):ind(2,i)));
    gind(l,ind(1,i):ind(2,i)) = k;
    k = k + 1;
    if ind(2,i)==p
        k = 1;
        l = l + 1;
    end
end

% ------- construct sparse matrix to vectorize the computation ----------
Gind = cell(1,d);
if ind(1,1) == -1
    j = 2;
else
    j = 1;
end
for i = j:d
    Gind{1,i} = sparse(gind(i,:),1:p,ones(1,p),nnl(i),p);
end

% --------------- put Xgnorm and weights in tree structure ---------------
XgnormTree = zeros(p,d);
weightTree = zeros(p,d);
for i = 1:d
    if ind(1,1)==-1&&i==1
        XgnormTree(:,i) = Xnorm;
        weightTree(:,i) = ind(3,1);
    else
        G = Gind{1,i};
        XgnormTree(:,i) = G'*(Xgnorm(nnind(i)+1:nnind(i+1)))';
        weightTree(:,i) = G'*(ind(3,nnind(i)+1:nnind(i+1)))';
    end
end

% ----------- solve SGL sequentially via TLFre ------------------
Xty = X'*y;
opts.rFlag = 0; % the input parameters are their true values

s = zeros(p,1);
c2 = zeros(p,1);
minn = zeros(p,1);

rLambdav = 1./Lambdav;
lambdap = Lambdav(1);
rlambdap = rLambdav(1);
vnormTree = zeros(p,d);
tol0 = 1e-12;
for i = 1:npar
    fprintf('in MLFre step: %d\n',i);
    lambdac = Lambdav(1,i);
    rlambdac = rLambdav(1,i);
    if lambdac>=lambda_max
        %Sol(:,Lambda_ind(i)) = 0; % because Sol is initalized to be 0
        ind_zf(:,:,Lambda_ind(i)) = true;     
    else
        if lambdap==lambda_max
            theta = y*rlambdap;
            z = Xty*rlambdap;
            [u, v] = Hierarchical_Projection( z, ind, nnind, Gind );
            if ind(3,end)==0
                weightd = (ind(3,nnind(d)+1:nnind(d+1)))';
                [~,Xmxind] = min(abs(Gind{1,d}*(v(:,d).*v(:,d))-weightd.*weightd));
                nv = X(:,ind(1,nnind(d)+Xmxind):ind(2,nnind(d)+Xmxind))*v(ind(1,nnind(d)+Xmxind):ind(2,nnind(d)+Xmxind),d);
            else
                nv = X*v(:,end);
            end           
        else
            theta = (y - X*Sol(:,Lambda_ind(i-1)))*rlambdap;
            nv = y*rlambdap-theta;
        end
       
        % ----- estimate the possible region of the dual optimum at lambdac
        nv = nv/norm(nv);
        rv = y*rlambdac-theta;
        Prv = rv - (nv'*rv)*nv;
        o = theta + 0.5*Prv;
        r = 0.5*norm(Prv);

        % ----- screening by MLFre, remove the ith feature if T(i)=1 ---- %
        c = X'*o;
        [u, v] = Hierarchical_Projection( c, ind, nnind, Gind );
        v2 = v.*v;
        for l = 1:d % compute norm of v for each node and arrange them based on the tree structure
            if l==1&&ind(1,1)==-1
                vnormTree(:,l) = abs(v(:,l));
            else
                G = Gind{1,l};
                vnormTree(:,l)=G'*sqrt(G*v2(:,l));
            end
        end
        csDifwv = cumsum(weightTree-vnormTree);
        
        T = false(p,1); % identify non-leaf inactive nodes
        for l = d:-1:2
            Tl = ~T; % find the indices of the remaining features
            % case 1
            Tc = false(p,1);
            Tc(Tl) = vnormTree(Tl,l)>tol0;
            if nnz(Tc)>0
                s(Tc) = vnormTree(Tc,l)+r*XgnormTree(Tc,l);
            end
            
            % case 2 & 3
            if nnz(Tc)<nnz(Tl) % if not all remaining nodes in level l fall in case 1
                Tcc = false(p,1);
                Tcc(Tl) = ~Tc(Tl);
                lind = nnind(l)+1:nnind(l+1);
                G = Gind{1,l};
                Tn = G*Tcc==(ind(2,lind)-ind(1,lind)+1)';
                indl = ind(:,lind);
                indlr = indl(:,Tn);
                for n = 1:nnz(Tn)
                    minn(n)=min(csDifwv(indlr(1,n):indlr(2,n),l-1));
                end
                sdist = (G(Tn,:))'*minn(1:nnz(Tn));
                s(Tcc) = max(0,r*XgnormTree(Tcc,l)-sdist(Tcc));
            end
            
            ind_zf(Tl,l,Lambda_ind(i))=s(Tl)<weightTree(Tl,l);
            T = T|ind_zf(:,l,Lambda_ind(i));
        end
        
        Tl = ~T; % identify inactive leaf nodes
        if ind(1,1)==-1           
            s(Tl) = abs(c(Tl))+r*Xnorm(Tl);
            ind_zf(Tl,1,Lambda_ind(i))=s(Tl)<ind(3,1);
        else
            G = Gind{1,1};
            lind = nnind(1)+1:nnind(2);
            Tn = G*Tl == (ind(2,lind)-ind(1,lind)+1)';
            c2(Tl) = c(Tl).*c(Tl);
            cnorm = (G(Tn,:))'*sqrt(G(Tn,:)*c2);
            s(Tl) = cnorm(Tl)+r*XgnormTree(Tl,1);
            ind_zf(Tl,1,Lambda_ind(i))=s(Tl)<weightTree(Tl,1);
        end      
        T = T|ind_zf(:,1,Lambda_ind(i));
        
        nT = ~T;
        Xr = X(:,nT);
        
        if lambdap == lambda_max
            opts.x0 = zeros(nnz(nT),1);
        else
            opts.x0 = Sol(nT,Lambda_ind(i-1));
        end
        
        % ------------ construct the reduced tree ---------------
        Tind = false(ng,1);
        nnlr = zeros(1,d+1);
        nnlr(end) = 1;
        if ind(1,1)==-1
            j=2;
            nnlr(1)=nnz(nT);
        else
            j=1;
        end
        for l = j:d
            lind = nnind(l)+1:nnind(l+1);
            Tind(lind) = Gind{1,l}*T==(ind(2,lind)-ind(1,lind)+1)';
            nnlr(l)=nnz(~Tind(lind));
        end
        if ind(1,1)==-1
            nnindr=[0,1,cumsum(nnlr(2:end))+1];
        else
            nnindr=[0,cumsum(nnlr)];
        end        
        indr = ind(:,~Tind);
        mapinde = cumsum(nT);
        mapinds = nnz(nT)+1-cumsum(nT,'reverse');
        for l=j:d+1
            lind = nnindr(l)+1:nnindr(l+1);
            oind1 = indr(1,lind);
            oind2 = indr(2,lind);
            indr(1,lind) = mapinds(oind1);
            indr(2,lind) = mapinde(oind2);
        end
        
        opts.ind = indr;
		% --- solve the TGL problem on the reduced data matrix -- %
        if opts.tFlag == 2
            opts.tol = funVal(Lambda_ind(i));
        end
        
        starts = tic;
        [x1, ~, ~]= tree_LeastR(Xr, y, lambdac, opts);   
        tsolver(Lambda_ind(i)) = toc(starts);
        
        Sol(nT,Lambda_ind(i)) = x1;
    end
    lambdap = lambdac;
    rlambdap = rlambdac;
end

end

