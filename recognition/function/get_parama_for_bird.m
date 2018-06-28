
function [sim, subsim, rsort, map_org_idx, aug_data_idx, sort_map_org] = get_parama(feature_set, NSClass, index_21w_2_9w)
%%computer similarity
n = length(feature_set);
sim = com_sim(n, NSClass, feature_set);
subsim = com_subsimi(n, NSClass, feature_set);
%save( fullfile( '/home/haichao/caffe/models/VGG/similarity', ['simi_' num2str(k) '_0' '.mat'] ), 'sim');
%save( fullfile( '/home/haichao/caffe/models/VGG/similarity', ['sub_simi_' num2str(k) '_0' '.mat'] ), 'sub_sim');

%%sort
[a b] = get_refer_ID(feature_set, sim, NSClass);
[rsort] = d_sort(sim, subsim, a, b);
[sim subsim] = sort_sim(rsort, sim, subsim);
%save( fullfile( '/home/haichao/caffe/models/VGG/sort', ['sortl_' num2str(k) '_0' '.mat'] ),'sortl');
%save( fullfile( '/home/haichao/caffe/models/VGG/sort', ['Sort_' num2str(k) '.mat']), 'rsort' );

%%get_map_idx1
[map_org_idx aug_data_idx] = map_idx_for_bird(feature_set, rsort, index_21w_2_9w, NSClass);
%save( fullfile( '/home/haichao/dog_query_cluster/click/reference', ['ll_img_' num2str(k) '.mat']), 'part_idx','map_org_idx','aug_data_idx');
for i = 1 : n
    for j = 1 : NSClass
        sort_map_org(i,j) = (rsort{i,1}-1) * NSClass + rsort{i,2}(j);
    end
end
end