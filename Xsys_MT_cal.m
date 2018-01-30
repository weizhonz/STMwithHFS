function [Xsys_MT] = Xsys_MT_cal(Xs, ys)
%Calculate Xty for each task and arrange them according to both the orders
%of the features and the tasks


Xsys_MT = zeros(size(Xs{1},2)*length(Xs),1);
idx_row = (0:size(Xs{1},2)-1)*length(Xs)+1;
for i = 1 : length(Xs)
    Xty = (Xs{i})'*(ys{i});
    Xsys_MT(idx_row,1) = Xty;
    idx_row = idx_row + 1;
end

end