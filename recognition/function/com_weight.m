
%feture: MFea
%A: the new matrix after propagating 
%B: the original matrix 
%NSClass: the number of sub_labels
%Index
function Query_i_p = com_weight( feature, A, B, NSClass,Index)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
a = cell2mat(feature);
b = B;
[row,col] = find(B == 0);
ind = sub2ind(size(b),row,col);
b(ind) = 1;
weight = A./b';
w = weight;
% [row,col] = find(weight == inf);
% ind = sub2ind(size(w),row,col);
% w(ind) = 1;
for i = 0 : length(Index)/2-1
    j = 2 * i;
    Query_i(:,Index(j+1):Index(j+2)) = bsxfun(@times,a(Index(j+1):Index(j+2),:)',w(:,i+1));
end

z = zeros(size(A));
z(ind) = 1/NSClass;
n_weight = A .* z;

for i = 0 : length(Index)/2-1
    j = 2 * i;
    Query_i_p(:,Index(j+1):Index(j+2)) = bsxfun(@plus,Query_i(:,Index(j+1):Index(j+2)),n_weight(:,i+1));
end
% Q = reshape(Query_i_p,size(n_weight,1),NSClass,size(n_weight,2));
% S = sum(Q,2);
% Q_s = reshape(S,size(n_weight,1),size(n_weight,2));
% max(max(abs(Q_s - A)));
end

