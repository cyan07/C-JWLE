function [ map_org_idx aug_data_idx ] = map_idx_for_bird ( feature_set, Sort, index_21w_2_9w, k)
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here

 
for i = 1 : length( feature_set )
    for j = 1 : k
        f = Sort{i,1};
        fi_21w{i,j} = find( feature_set{f,2}(1,:) == Sort{i,2}(j) );
       for  l = 1 : length( fi_21w{i,j} )
            n = fi_21w{i,j}(l);
            s = feature_set{f,2}(2,n);
            idex_ll_9w{i,j}(l) = index_21w_2_9w( s );
            idex_ll_21w{i,j}(l) =  s;
       end
    end
end

part_idx = fi_21w;
map_org_idx = idex_ll_9w;
aug_data_idx = idex_ll_21w;
end