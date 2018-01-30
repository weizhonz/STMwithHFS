function [Sol_MT] = STM_HFS(Xs, ys, Lambda, opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Implementation of the sequential HFS rule for STM
%% input:
%         Xs:
%            Xs{i} stores the data matrix of the i-th task, each column corresponds to a feature
%            each row corresponds to a data instance
%
%         ys:
%            ys{i} strores the response vector of the i-th task
%
%         Lambda:
%            the parameter values of lambda
%
%         opts:
%            settings for the solver
%% output:
%         Sol:
%              the solution; Sol(:,:,i) stores the the
%              solution for the ith values in Lambda
%
%% For any problem, please contact Weizhong Zhang (zhangweizhongzju@gmail.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%

% -------------------------- pass parameters ---------------------------- %
p = size(Xs{1}, 2);
T_num = length(Xs); % number of tasks
npar = length(Lambda); % number of parameter values of lambda
ind = opts.ind;
ind_MT = TreeTransform(ind, T_num);
%clear('ind');
opts.init=1;        % starting from a zero point
if opts.tFlag==2
    funVal = opts.funVal;
end



% --------------------- recover tree structure for Multi Task-------------------------- %
eind_MT = find(ind_MT(2,:) == p*T_num);
if ind_MT(1,1) == -1 % find the depth of the tree
    d_MT = length(eind_MT);
    nnl_MT = [p*T_num,eind_MT(1)-1,diff(eind_MT)]; % number of nodes per layer
    nnind_MT = [0,1,eind_MT]; % ind(1,nnind1(i)+1:nnind(i+1)) stores the node from the ith layer
else
    d_MT = length(eind_MT)-1;
    nnl_MT = [eind_MT(1),diff(eind_MT)];
    nnind_MT = [0,eind_MT];
end


% --------------------- initialize the output --------------------------- %
Sol_MT = zeros(p,T_num,npar);
ind_zf_MT = false(p*T_num,d_MT,npar);
tsolver_MT = zeros(1,npar);%should be changed
tscreen_MT = zeros(1,npar);

% ------------------- compute the effective region of lambda ------------ %
Xsys_MT = Xsys_MT_cal(Xs, ys);
%lambda_max=findLambdaMax(X'*y, p, ind, size(ind,2)); % why?
lambda_max=findLambdaMax(Xsys_MT, p*T_num, ind_MT, size(ind_MT,2));
%lambda_max1=findLambdaMax(Xsys_MT, p*T_num, ind, size(ind,2));
if opts.rFlag == 1
    Lambda = Lambda * lambda_max;
    opts.rFlag = 0;
end
[Lambdav,Lambda_ind] = sort(Lambda,'descend');




% --------- compute the norm of each feature and each submatrix ---------
Xnorm_MT = zeros(size(Xs{1},2)*T_num,1);
idx_row = [0:size(Xs{1},2)-1]*T_num+1;
for idx_t = 1:T_num
    Xnorm_MT(idx_row) = (sqrt(sum(Xs{idx_t}.^2,1)))';
    idx_row = idx_row +1;
end
ng_MT = size(ind_MT,2);
Xgnorm_MT = zeros(1,ng_MT);
gind_MT = zeros(d_MT,p*T_num);
if ind_MT(1,1)==-1
    j = 2;
    l = 2;
else
    l = 1;
    j = 1;
end
k = 1;
for i = j : ng_MT-1
    for idx_t = 1: length(Xs)
        if(idx_t ==1)
            Xgnorm_MT(i) = norm(Xs{idx_t}(:,((ind_MT(1,i)-1)/T_num+1):(ind_MT(2,i)/T_num)));
        else
            Xgnorm_MT(i)= max(Xgnorm_MT(i),norm(Xs{idx_t}(:,((ind_MT(1,i)-1)/T_num+1):(ind_MT(2,i)/T_num))));
        end
    end
    gind_MT(l,ind_MT(1,i):ind_MT(2,i)) = k;
    k = k + 1;
    if ind_MT(2,i)==p*T_num
        k = 1;
        l = l + 1;
    end
end

% ------- construct sparse matrix to vectorize the computation ----------
Gind_MT = cell(1,d_MT);
if ind_MT(1,1) == -1
    j = 2;
else
    j = 1;
end
for i = j:d_MT
    Gind_MT{1,i} = sparse(gind_MT(i,:),1:p*T_num,ones(1,p*T_num),nnl_MT(i),p*T_num);
end



% --------------- put Xgnorm and weights in tree structure ---------------
XgnormTree_MT = zeros(p*T_num,d_MT);
weightTree_MT = zeros(p*T_num,d_MT);
for i = 1:d_MT
    if ind_MT(1,1)==-1&&i==1
        XgnormTree_MT(:,i) = Xnorm_MT;
        weightTree_MT(:,i) = ind_MT(3,1);
    else
        G_MT = Gind_MT{1,i};
        XgnormTree_MT(:,i) = G_MT'*(Xgnorm_MT(nnind_MT(i)+1:nnind_MT(i+1)))';
        weightTree_MT(:,i) = G_MT'*(ind_MT(3,nnind_MT(i)+1:nnind_MT(i+1)))';
    end
end





% ----------- solve STM sequentially via HFS ------------------
opts.rFlag = 0; % the input parameters are their true values

s_MT = zeros(p*T_num,1);
c2_MT = zeros(p*T_num,1);
minn_MT = zeros(p*T_num,1);

rLambdav = 1./Lambdav;
lambdap = Lambdav(1);
rlambdap = rLambdav(1);
vnormTree_MT = zeros(p*T_num,d_MT);
tol0 = 1e-12;
for i = 1:npar
    
    %fprintf('in HFS step: %d\n',i);
    lambdac = Lambdav(1,i);
    rlambdac = rLambdav(1,i);
    if lambdac>=lambda_max
        ind_zf_MT(:,:,Lambda_ind(i)) = true;
    else
        starts_screening =tic;
        if lambdap==lambda_max
            theta_MT = [];
            for idx_t = 1: length(ys)
                theta_MT = [theta_MT; ys{idx_t}*rlambdap];
            end
            
            z_MT = Xsys_MT*rlambdap;
            [u_MT, v_MT] = Hierarchical_Projection( z_MT, ind_MT, nnind_MT, Gind_MT );
            
            if ind_MT(3,end)==0
                weightd_MT = (ind_MT(3,nnind_MT(d_MT)+1:nnind_MT(d_MT+1)))';
                [~,Xmxind_MT] = min(abs(Gind_MT{1,d_MT}*(v_MT(:,d_MT).*v_MT(:,d_MT))-weightd_MT.*weightd_MT));
                idx_column_MT  = [ind_MT(1,nnind_MT(d_MT)+Xmxind_MT):ind_MT(2,nnind_MT(d_MT)+Xmxind_MT)];
                nv_MT = [];
                idx_column_X = [(idx_column_MT(end)/T_num)-(length(idx_column_MT)/T_num)+1:(idx_column_MT(end)/T_num)];
                for idx_t = 1:length(ys)
                    idx_column = idx_column_MT(idx_t:T_num:end);
                    nv_MT = [nv_MT; Xs{idx_t}(:,idx_column_X)*v_MT(idx_column,d_MT)];
                end
            else
                nv_MT = [];
                for idx_t = 1:length(ys)
                    idx_column = [idx_t:T_num:lenght(v_MT)];
                    nv_MT = [nv_MT; Xs{idx_t}*v_MT(idx_column,end)];
                end
            end
        else
            theta_MT = [];
            y_all = [];
            for idx_t = 1:length(ys)
                theta_MT = [theta_MT;(ys{idx_t} - Xs{idx_t}*Sol_MT(:,idx_t,Lambda_ind(i-1)))*rlambdap];
                y_all = [y_all; ys{idx_t}];
            end
            nv_MT = y_all*rlambdap-theta_MT;
        end
        
        % ----- estimate the possible region of the dual optimum at lambdac
        nv_MT = nv_MT/norm(nv_MT);
        %rv = y*rlambdac-theta;
        rv_MT = [];
        for idx_t = 1:length(ys)
            rv_MT = [rv_MT;ys{idx_t}*rlambdac];
        end
        rv_MT = rv_MT-theta_MT;
        
        Prv_MT = rv_MT - (nv_MT'*rv_MT)*nv_MT;
        o_MT = theta_MT + 0.5*Prv_MT;
        r_MT = 0.5*norm(Prv_MT);
        
        
        
        % ----- screening by MLFre, remove the ith feature if T(i)=1 ---- %
        c_MT = zeros(p*T_num,1);
        idx_row = (0:size(Xs{1},2)-1)*length(Xs)+1;
        for idx_t = 1:length(ys)
            c_MT(idx_row) = Xs{idx_t}'*o_MT((idx_t-1)*length(ys{1})+1:idx_t*length(ys{1}));
            idx_row = idx_row+1;
        end
        [u_MT, v_MT] = Hierarchical_Projection( c_MT, ind_MT, nnind_MT, Gind_MT );
        v2_MT = v_MT.*v_MT;
        for l = 1:d_MT % compute norm of v for each node and arrange them based on the tree structure
            if l==1&&ind_MT(1,1)==-1
                vnormTree_MT(:,l) = abs(v_MT(:,l));
            else
                G_MT = Gind_MT{1,l};
                vnormTree_MT(:,l)=G_MT'*sqrt(G_MT*v2_MT(:,l));
            end
        end
        csDifwv_MT = cumsum(weightTree_MT-vnormTree_MT);
        
        T_MT = false(p*T_num,1); % identify non-leaf inactive nodes
        for l = d_MT:-1:2
            Tl_MT = ~T_MT; % find the indices of the remaining features
            % case 1
            Tc_MT = false(p*T_num,1);
            Tc_MT(Tl_MT) = vnormTree_MT(Tl_MT,l)>tol0;
            if nnz(Tc_MT)>0
                s_MT(Tc_MT) = vnormTree_MT(Tc_MT,l)+r_MT*XgnormTree_MT(Tc_MT,l);
            end
            
            % case 2 & 3
            if nnz(Tc_MT)<nnz(Tl_MT) % if not all remaining nodes in level l fall in case 1
                Tcc_MT = false(p*T_num,1);
                Tcc_MT(Tl_MT) = ~Tc_MT(Tl_MT);
                lind_MT = nnind_MT(l)+1:nnind_MT(l+1);
                G_MT = Gind_MT{1,l};
                Tn_MT = G_MT*Tcc_MT==(ind_MT(2,lind_MT)-ind_MT(1,lind_MT)+1)';
                indl_MT = ind_MT(:,lind_MT);
                indlr_MT = indl_MT(:,Tn_MT);
                for n = 1:nnz(Tn_MT)
                    minn_MT(n)=min(csDifwv_MT(indlr_MT(1,n):indlr_MT(2,n),l-1));
                end
                sdist_MT = (G_MT(Tn_MT,:))'*minn_MT(1:nnz(Tn_MT));
                s_MT(Tcc_MT) = max(0,r_MT*XgnormTree_MT(Tcc_MT,l)-sdist_MT(Tcc_MT));
            end
            
            ind_zf_MT(Tl_MT,l,Lambda_ind(i))=s_MT(Tl_MT)<weightTree_MT(Tl_MT,l);
            T_MT = T_MT|ind_zf_MT(:,l,Lambda_ind(i));
        end
        
        Tl_MT = ~T_MT; % identify inactive leaf nodes
        if ind_MT(1,1)==-1
            s_MT(Tl_MT) = abs(c_MT(Tl_MT))+r_MT*Xnorm_MT(Tl_MT);
            ind_zf_MT(Tl_MT,1,Lambda_ind(i))=s_MT(Tl_MT)<ind_MT(3,1);
        else
            G_MT = Gind_MT{1,1};
            lind_MT = nnind_MT(1)+1:nnind_MT(2);
            Tn_MT = G_MT*Tl_MT == (ind_MT(2,lind_MT)-ind_MT(1,lind_MT)+1)';
            c2_MT(Tl_MT) = c_MT(Tl_MT).*c_MT(Tl_MT);
            cnorm_MT = (G_MT(Tn_MT,:))'*sqrt(G_MT(Tn_MT,:)*c2_MT);
            s_MT(Tl_MT) = cnorm_MT(Tl_MT)+r_MT*XgnormTree_MT(Tl_MT,1);
            ind_zf_MT(Tl_MT,1,Lambda_ind(i))=s_MT(Tl_MT)<weightTree_MT(Tl_MT,1);
        end
        T_MT = T_MT|ind_zf_MT(:,1,Lambda_ind(i));
        
        nT_MT = ~T_MT;
        
        
        %Xr = X(:,nT);
        nT_MT_X = nT_MT(1:T_num:end);
        for idx_t = 1:T_num
            Xrs{idx_t} = Xs{idx_t}(:,nT_MT_X);
        end
        
        
        
        if lambdap == lambda_max
            opts.x0 = zeros(nnz(nT_MT_X)*T_num,1);
        else
            x0_Matrix = Sol_MT(nT_MT_X,:,Lambda_ind(i-1));
            x0_temp =zeros(size(x0_Matrix,1)*T_num,1);
            idx_row = [0:size(x0_Matrix,1)-1]*T_num+1;
            for idx_t = 1:T_num
                x0_temp(idx_row) = x0_Matrix(:,idx_t);
                idx_row = idx_row+1;
            end
            opts.x0 = x0_temp;
        end
        
        % ------------ construct the reduced tree ---------------
        Tind_MT = false(ng_MT,1);
        nnlr_MT = zeros(1,d_MT+1);
        nnlr_MT(end) = 1;
        if ind_MT(1,1)==-1
            j=2;
            nnlr_MT(1)=nnz(nT_MT);
        else
            j=1;
        end
        for l = j:d_MT
            lind_MT = nnind_MT(l)+1:nnind_MT(l+1);
            Tind_MT(lind_MT) = Gind_MT{1,l}*T_MT==(ind_MT(2,lind_MT)-ind_MT(1,lind_MT)+1)';
            nnlr_MT(l)=nnz(~Tind_MT(lind_MT));
        end
        if ind_MT(1,1)==-1
            nnindr_MT=[0,1,cumsum(nnlr_MT(2:end))+1];
        else
            nnindr_MT=[0,cumsum(nnlr_MT)];
        end
        indr_MT = ind_MT(:,~Tind_MT);
        mapinde_MT = cumsum(nT_MT);
        mapinds_MT = nnz(nT_MT)+1-cumsum(nT_MT,'reverse');
        for l=j:d_MT+1
            lind_MT = nnindr_MT(l)+1:nnindr_MT(l+1);
            oind1_MT = indr_MT(1,lind_MT);
            oind2_MT = indr_MT(2,lind_MT);
            indr_MT(1,lind_MT) = mapinds_MT(oind1_MT);
            indr_MT(2,lind_MT) = mapinde_MT(oind2_MT);
        end
        
        
        opts.ind_MT = indr_MT;
        % --- solve the STM problem on the reduced data matrix -- %
        if opts.tFlag == 2
            opts.tol = funVal(Lambda_ind(i));
        end
        tscreen_MT(Lambda_ind(i)) = toc(starts_screening);
        
        starts = tic;
        [x1, ~, ~]= tree_LeastR_MT(Xrs, ys, lambdac, opts);
        tsolver_MT(Lambda_ind(i)) = toc(starts);
        
        nT_MT_temp = nT_MT(1:T_num:end);
        idx_row= [1:T_num:length(x1)];
        for idx_t = 1:T_num
            Sol_MT(nT_MT_temp,idx_t,Lambda_ind(i)) = x1(idx_row);
            idx_row = idx_row +1;
        end
    end
    lambdap = lambdac;
    rlambdap = rlambdac;
end

end

