
%feature_set: the feature cell
%sim: the similarity of labels
%k: the number of kmeans cluster
function [a b] = get_refer_ID( feature_set, sim, k )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
select_stadd = var( sim );
for i = 1 : length( sim )
    select_stadd(2,i) = size(feature_set {i,2},2);
end

%%%%ADD BY TM
%%%%Cv = sortrows(select_stadd',-2);

%%%%[row col] = find ( select_stadd == Cv(1,2) ); 
[~, col] = sortrows(select_stadd',-2);col = col(1);
%%%%ADD BY TM


for i = 1 : length( feature_set )
    for j = 1 : k
        coun_t(i,j) = numel(find( feature_set{i,2}(1,:) == j));
    end
end
[~,little_l_ID] = max(coun_t');

a = col;
b = little_l_ID;

end

