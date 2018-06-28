
%feature : feature vector as a row,matrix:n*4096
%k : the cluster number of Kmeans 
function [ Km_center, Km_classify] = normal_vl_kmeans( feature,k )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    sqrt_sum = sqrt( sum( feature.^2 ,2));
    n_aug_fea = bsxfun( @times, feature, 1./sqrt_sum );    
    [Km_center, Km_classify] = vl_kmeans( n_aug_fea', k );
end

