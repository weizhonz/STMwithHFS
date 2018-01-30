function [ X, y, beta, ind ] = gen_sync2( p, N, d, nns, ratio )
%% generate synthetic data with zero pair-wise correlation

nl = p./nns; % the lenght of nl is d+1; n[end] = 1 as we only has 
                      % one root node; nl[1] is the number of nodes at d
                      % layer; nl[2] is the number of nodes at d-1 layer

% construct sparse matrix to vectorize the computation
gind = zeros(d,p);
for l = 1:d
    gind(l,nns(l):nns(l):p) = 1;
    gind(l,:) = nl(l)+1-cumsum(gind(l,:),'reverse');
end
Gind = cell(1,d);
if nl(1)==p
    j=2;
else
    j=1;
end
for l=j:d
    Gind{l}=sparse(gind(l,:),1:p,ones(1,p),nl(l),p);
end
                      
% generate the data matrix
mu = zeros(1,p);
if p<=20000
    Mrow = repmat((1:p)',1,p);
    Mcolumn = repmat(1:p,p,1);
    SIGMA = (0.5*ones(p,p)).^abs(Mrow-Mcolumn);
else
    power = 50;
    nz = p+2*(power*p-(power+1)*power/2);
    Mrow = zeros(1,nz);
    Mcolumn = zeros(1,nz);
    s = zeros(1,nz);
    %count = 0;
    Mrow(1:p) = 1:p;
    Mcolumn(1:p) = 1:p;
    s(1:p) = 1;
    count = nnz(s);
    for i = 1:power
        Mrow(count+1:count+p-i) = 1:(p-i);
        Mcolumn(count+1:count+p-i) = (i+1):p;
        s(count+1:count+p-i) = 0.5^i;
        count = nnz(s);
        Mrow(count+1:count+p-i) = (i+1):p;
        Mcolumn(count+1:count+p-i) = 1:(p-i);
        s(count+1:count+p-i) = 0.5^i;
        count = nnz(s);
    end
    SIGMA=sparse(Mrow,Mcolumn,s,p,p,nz);
end
X = mvnrnd(mu,SIGMA,N);

% generate the response
T = true(p,1);
for l = 1:d
    if l==1&&nl(1)==p
        Tf = unifrnd(0,1,[p,1])<=ratio(l);
    else
        Tl = unifrnd(0,1,[nl(l),1])<=ratio(l);
        Tf = logical((Gind{l})'*Tl);
    end
    T = T&Tf;
end


beta = zeros(p,1);
beta(T) = normrnd(0,1,[nnz(T),1]);

y = X*beta + 0.01*normrnd(0,1,[N,1]);  

% construct ind

if nl(1) == p % if the nodes at the first layer only has one feature
    cumnl = [0,1,cumsum(nl(2:end))+1];
    ind = zeros(3,cumnl(end));
    ind(:,1)=[-1, -1, 1]';
    ind(:,end)=[1,p,0]';
    for i = 2:d
        ind(1,cumnl(i)+1:cumnl(i+1)) = 1:nns(i):p;
        ind(2,cumnl(i)+1:cumnl(i+1)) = ind(1,cumnl(i)+1:cumnl(i+1))+nns(i)-1;
        ind(3,cumnl(i)+1:cumnl(i+1)) = sqrt(nns(i));
    end
else
    cumnl = [0,cumsum(nl)];
    ind = zeros(3,cumnl(end));
    ind(:,end)=[1,p,0]';
    for i = 1:d
        ind(1,cumnl(i)+1:cumnl(i+1)) = 1:nns(i):p;
        ind(2,cumnl(i)+1:cumnl(i+1)) = ind(1,cumnl(i)+1:cumnl(i+1)) + nns(i);
        ind(3,cumnl(i)+1:cumnl(i+1)) = sqrt(nns(i));
    end
end

end

