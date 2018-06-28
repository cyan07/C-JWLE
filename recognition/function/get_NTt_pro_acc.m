function acc = get_NTt_pro_acc(data_fea,img_Fea_ClickCount,train_click_col,test_click_col,aug_train_id,sub_test,NsubIndex,Q_mNUM,train_label,test_true_label,perc,t)
query_col = union(train_click_col,test_click_col); 
img_col = [aug_train_id(:);sub_test(:,1)];
Que_Fea_Clickcount = img_Fea_ClickCount(img_col,query_col);
%propagate
sub_feature = data_fea(img_col,:);
label = [train_label(:);sub_test(:,2)];
for  i = 1 : length(NsubIndex)
    f = find(label == NsubIndex(i));
    feature{i} = sub_feature(f,:);
end
QSum = pro_fea(feature,Que_Fea_Clickcount,perc,t);
QSum = cell2mat(QSum);
%%the IDX from method 1:K-means
x = sqrt( sum( QSum ,1));
x_fea = bsxfun( @times, QSum ,1./x ); 

[Fea IDX] = vl_kmeans(x_fea', Q_mNUM);

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