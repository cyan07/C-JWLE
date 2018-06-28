function acc = get_NTt_struc_acc(aug_train_feature,data_fea,img_Fea_ClickCount,train_click_col,test_click_col,aug_train_id,sub_test,NsubIndex,Q_dNUM,train_label,test_true_label,k,perc,k_N,k_NS)
query_col = union(train_click_col,test_click_col); 
img_col = [aug_train_id(:);sub_test(:,1)];
Que_Fea_Clickcount = img_Fea_ClickCount(img_col,query_col);
%propagate
sub_feature = [aug_train_feature;data_fea(sub_test(:,1),:)];
sub_feature  = bsxfun( @times, sub_feature, 1./sqrt(sum(sub_feature.^2,2)) );%normalization  
label = [train_label(:);sub_test(:,2)];
l = 0;
for  i = 1 : length(NsubIndex)
    f = find(label == NsubIndex(i));
    feature{i,1} = sub_feature(f,:);
%     r = l+length(f);
%     feature{i,2} =[l+1:r];
%     l = r;
end
feature_set = get_feature_set(feature,k);
for  i = 1 : length(NsubIndex)
    f = find(label == NsubIndex(i));
    feature_set{i,2} = [feature_set{i,2};f'];
end
[sim, subsim, rsort, aug_img, sort_map_img] = get_parama(feature_set, k, img_col); 
MFea = get_MFea( aug_img, Que_Fea_Clickcount, length(NsubIndex), k);
[M_profea M_data M_mdata] = Aft_propagate( Que_Fea_Clickcount,sim,subsim,MFea,perc,k_N,k_NS);
[Fea_M IDX_M] = vl_kmeans(M_profea', Q_dNUM);

IDX = [IDX_M; query_col]; 
ID =  IDX';
ID = sortrows(ID,2);
id = ID(:,2);
[~,train_ID,~] = intersect(id,train_click_col);
[~,test_ID,~] = intersect(id,test_click_col);
map = ID(:,1);
train_map = map(train_ID);
test_map = map(test_ID);
query_train_datafea = img_Fea_ClickCount(aug_train_id,train_click_col);
query_test_datafea = img_Fea_ClickCount(sub_test(:,1),test_click_col);
map_query_train = zeros(size(query_train_datafea,1),max(map));
for i = 1 : max(train_map)
    tr_ind = find(train_map == i);
    if ~isempty(tr_ind)
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