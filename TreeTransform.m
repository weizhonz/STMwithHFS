function [ ind_MT ] = TreeTransform( ind, T_num )
%Transform the index of the model into multi-task scene

if ind(1,1) ==-1
    ind_MT_tail = zeros(size(ind,1), size(ind,2)-1);
    
    row1 = ind(2,2:(end-1))*T_num + 1;
    ind_MT_tail(1, 2:end) = row1;
    ind_MT_tail(2,:) = T_num*ind(2,2:end);
    ind_MT_tail(3,:) = ind(3,2:end);
    ind_MT_tail(1, find(ind(1,2:end)==1))=1;
    
    p = max(ind(2,:));
    ind_MT_head = zeros(size(ind,1), p);
    ind_MT_head(1,:) = [0:(p-1)]*T_num+1;
    ind_MT_head(2,:) = [1:p]*T_num;
    ind_MT_head(3,:) = ind(3,1);
    ind_MT = [ind_MT_head, ind_MT_tail];
else
    ind_MT = zeros(size(ind,1), size(ind,2));
    row1 = ind(2,1:(end-1))*T_num + 1;
    ind_MT(1, 2:end) = row1;
    ind_MT(2,:) = T_num*ind(2,:);
    ind_MT(3,:) = ind(3,:);
    ind_MT(1, find(ind(1,1:end)==1))=1;
end


end

