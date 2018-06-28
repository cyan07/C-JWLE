function feature_set = get_feature_set(feature,k)
for j = 1 : length(feature)
    fea = feature{j,1};
    [feature_set{j,1}, feature_set{j,2}] =vl_kmeans(fea' ,k);
    fea = [];
    %[feature_set{j,1}, feature_set{j,2}] =normal_vl_kmeans(fea ,k);
end
% for j = 1 : length(feature_set)
%     feature_set{j,2} = [feature_set{j,2};feature{j,2}'];
% end
end