function [ output_args ] = SyntheticDatagenerator( input_args )
%SYNTHETICDATAGENERATOR Summary of this function goes here
%   Detailed explanation goes here

num_node = [10,5,1];
num_fea = [10,20,100];
dim_fea = 100;
num_task = 8;
num_sample = 20;
spars_ratio = 0.1;
%% generate the index tree
index_tree = zeros(3,1+sum(num_node));

%% 4-th layer
tree_column = [-1,-1,1];
index_tree(:,1) = tree_column;

start_column = 2;
%% 3-1th layer
for i = 1:3
    num_fea_temp = num_fea(i);
    num_node_temp = num_node(i);
    index_tree = layerConstructer(index_tree, start_column, num_fea_temp, num_node_temp);
    start_column = start_column+num_node_temp;
end
index_tree(3,end)=0;
save('SyntheticData/index_tree.mat', 'index_tree');

%% generator the coefficient matrix
active_node_idx = randperm(num_node(1),num_node(1)*spars_ratio)';
active_fea_idx = [];
for idx =1 :length(active_node_idx)
active_fea_idx = [active_fea_idx;((active_node_idx(idx)-1)*num_fea(1)+1:active_node_idx(idx)*num_fea(1))'];
end
coeff_matrix = zeros(dim_fea,num_task);
for idx_t = 1:num_task
coeff_matrix(active_fea_idx, idx_t) = randn(dim_fea*spars_ratio,1);
end

%% gengerator the data matrix and the response

for idx_t = 1:num_task
norm_input= randn(num_sample, dim_fea);
norm_res = norm_input*coeff_matrix(:,idx_t);
datasetname = ['SyntheticData/SyntheticData',num2str(idx_t),'.mat'];
coeff_vec = coeff_matrix(:,idx_t);
save(datasetname, 'norm_input', 'norm_res','coeff_vec');
end

save('SyntheticData/coeff_matrix.mat', 'coeff_matrix');








end

function [index_tree_new ]= layerConstructer(index_tree, start_column, num_fea_temp, num_node_temp)
for idx_column = start_column:start_column+num_node_temp-1
    if(idx_column==start_column)
        tree_column = [1;num_fea_temp;sqrt(num_fea_temp)];
        index_tree(:,idx_column) =tree_column;
    else
        tree_column(1:2) = tree_column(1:2)+num_fea_temp;
        index_tree(:,idx_column) =tree_column;
    end
end
index_tree_new = index_tree;

end




