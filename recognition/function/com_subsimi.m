%n: from the dimension of feature_set , means the dimension of return
%k: from the dimension of feature in feature_set , means the number of each
    %sample in the feature_set, equal to k
%feature_set: contains each feature of labels ,cell:283*2


function sub_sim = com_subsimi( n, k, feature_set )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
for i = 1 : n
    for j = 1 : k
        for l = j : k
            sub_sim{i}(j,l) = sum( sum( EuDist2(feature_set{i,1}(:,j)',feature_set{i,1}(:,l)') ));
            sub_sim{i}(l,j) = sub_sim{i}(j,l);     %assignment diagonal
        end
    end
end

end

