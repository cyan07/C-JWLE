
%feature : feature matrix

function n_aug_fea = normalfea( feature )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    s_sum = sum( feature ,1);
    n_aug_fea = bsxfun( @times, feature, 1./s_sum );    
    %[Km_center, Km_classify] = vl_kmeans( n_aug_fea', k );
end

