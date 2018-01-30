function [ u, v ] = Hierarchical_Projection( z, ind, nnind, Gind )
%% this function implements the Hierarchical Projection algorithm

p = max(ind(2,:));

d= length(Gind);

u = zeros(p,d+1);
v = zeros(p,d+1);

if ind(1,1) == -1
    v(:,1) = sign(z).*min(ind(3,1),abs(z));
else
    G = Gind{1,1};
    gnorm = sqrt(G*(z.*z));
    znode = gnorm < 1e-12;
    w = ind(3,nnind(1)+1:nnind(2))';
    Pgnorm = zeros(size(w));
    Pgnorm(znode) = 0;
    nznode = ~znode;
    Pgnorm(nznode) = min(1,w(nznode)./gnorm(nznode));
    v(:,1) = (G'*Pgnorm).*z;
end
u(:,1) = v(:,1);

for i = 2:d
    r = z - u(:,i-1); 
    G = Gind{1,i};
    gnorm = sqrt(G*(r.*r));
    znode = gnorm < 1e-12;
    w = ind(3,nnind(i)+1:nnind(i+1))';
    Pgnorm = zeros(size(w));
    Pgnorm(znode) = 0;
    nznode = ~znode;
    Pgnorm(nznode) = min(1,w(nznode)./gnorm(nznode));
    v(:,i) = (G'*Pgnorm).*r;
    u(:,i) = v(:,i) + u(:,i-1);
end

r = z - u(:,d);
gnorm = norm(r);
if gnorm < 1e-12
    v(:,d+1) = 0;
else
    v(:,d+1) = min(1,ind(3,end)/gnorm)*r;
end
u(:,d+1) = v(:,d+1) + u(:,d);
end

