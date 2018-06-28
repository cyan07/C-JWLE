
function acc = get_NTt_acc(img_Fea_ClickCount,train_click_col,test_click_col,aug_train_id,sub_test_id,Q_dNUM,train_label,test_true_label)

query_col = union(train_click_col,test_click_col); 
img_col = [aug_train_id(:);sub_test_id(:)];
Que_Fea_Clickcount = img_Fea_ClickCount(img_col,query_col);
%%the IDX from method 1:K-means
x = sqrt( sum( Que_Fea_Clickcount ,1));
x_fea = bsxfun( @times, Que_Fea_Clickcount ,1./x ); 
%Q_NUM = 150;
[Fea IDX] = vl_kmeans(full(x_fea), Q_dNUM);%length(train_map_label{1,2})+Q_NUM);

IDX = [IDX; query_col]; 
ID =  IDX';
ID = sortrows(ID,2);
id = ID(:,2);
[~,train_ID,~] = intersect(id,train_click_col);
[~,test_ID,~] = intersect(id,test_click_col);
map = ID(:,1);
train_map = map(train_ID);
test_map = map(test_ID);
query_train_datafea = img_Fea_ClickCount(aug_train_id,train_click_col);
query_test_datafea = img_Fea_ClickCount(sub_test_id,test_click_col);
map_query_train = zeros(size(query_train_datafea,1),max(map));
for i = 1 : max(train_map)
    tr_ind = find(train_map == i);
    if ~isempty(tr_ind)
%         map_query_train(:,i) = 0;
%     else
        map_query_tr = query_train_datafea(:,tr_ind);
        map_query_train(:,i) = sum(map_query_tr,2);
    end
end
map_query_test = zeros(size(query_test_datafea,1),max(map));
for i = 1 : max(test_map)
    t_ind = find(test_map == i);
    if ~isempty(t_ind)
%         map_query_test(:,i) = 0;
%     else
        map_query_t = query_test_datafea(:,t_ind);
        map_query_test(:,i) = sum(map_query_t,2);
    end
end
m_query_train = map_query_train;
map_query_train = bsxfun( @times, m_query_train, 1./sum(m_query_train,2) );%normalization by Query   
m_query_test = map_query_test;
map_query_test = bsxfun( @times, m_query_test, 1./sum(m_query_test,2) );%normalization by Query   
map_t2t = EuDist2(map_query_train,map_query_test);
[min_r min_c] = min(map_t2t);

test_label = train_label(min_c);
acc = length(find(test_label(:) - test_true_label(:) == 0))/length(test_label);
%acc1 = 1 - nnz(test_label(:) - test_true_label(:)) / length(test_label);
end
%%1018

% %%top five to match query
% map_test_q5 = find_top_n(vf_m,map,i_t,i_tr,5);
% %save(fullfile( rootdir, 'subLabel_kmeans/alter', ['map_top1_' num2str(Nsub) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'map_test_q1', '-v7.3');
% 
% %[map_query_test,map_query_train] = get_Mquery_datafea(map,query_train_datafea,query_test_datafea,map_test_q5);
% %save(fullfile( rootdir, 'subLabel_kmeans/alter', ['map_top5_' num2str(Nsub) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'map_test_q5', 'map_query_test', 'map_query_train', '-v7.3');
% [min_c,min_ic,train_label,test_true_label] = get_test_label(map_query_train,map_query_test,train_feature,test_feature,NsubIndex);
% test_label = train_label(min_c);
% test_image_label = train_label(min_ic);
% %test_true_label = [];
% acc = length(find(test_label - test_true_label == 0))/length(test_label);
% save(fullfile( rootdir, 'subLabel_kmeans/alter', ['label_top_' num2str(Nsub) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'test_label', 'train_label', 'acc', '-v7.3');


%%% rewrite by hc on 2017/09/29

