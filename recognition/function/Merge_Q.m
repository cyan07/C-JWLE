%load('/home/haichao/181/image_click_Dog283_0_img_Fea_Clickcount.mat');
% load ( '/home/haichao/181/image_click_Dog283_0_database.mat' );
% load('/home/haichao/181/subLabel_kmeans/alter/train_feature_20_40_0.66667_rand_idx_33-271.mat');
% load ( '/home/haichao/181/Dog/index_by_hc/index_21w_2_9w.mat');
% for i= 1 :20
%     tf = train_feature{i,2}(1,:);
%     t = index_21w_2_9w(tf);
%     train_feature{i,2}(2,:) = database.label(t);
%end

function M_Q = Merge_Q(Query_datafea, query_label)
    M_Q = [];
    for i = 1 : max(query_label)
        ind = find(query_label==i);
        q = Query_datafea(:,ind);
        q = sum(q,2);
        M_Q = [M_Q,q];
    end
end