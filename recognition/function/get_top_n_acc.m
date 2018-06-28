function acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,m)
map = ID(:,1);
%map_test_qm = find_top_n(vf_m,map,i_t,i_tr,m);
map_test_qm = query_id;
map_test_qm(i_t) = i_tr;
map_test_qm = map(map_test_qm);

[map_query_test,map_query_train] = get_Mquery_datafea(map,train_click_fea,test_click_fea,map_test_qm);

%[min_c,train_label] = get_test_label(map_query_train,map_query_test,train_feature,NsubIndex);
%normalization by Image 
m_query_train = map_query_train;
map_query_train = full(bsxfun( @times, m_query_train, 1./sum(m_query_train,2)));%normalization by Query   

m_query_test = map_query_test;
s = sum(m_query_test,2);
ind = find(s==0);
s(ind)=1;
map_query_test = bsxfun( @times, m_query_test, 1./s );%normalization by Query   

map_t2t = EuDist2(map_query_train,map_query_test);
[min_r min_c] = min(map_t2t);

test_label = train_label(min_c);
acc = length(find(test_label(:) - test_true_label(:) == 0))/length(test_label);
%acc1 = 1 - nnz(test_label(:) - test_true_label(:)) / length(test_label);
% %getACC
% add(genpath('/home/haichao/haichao/subLabel_Kmeans/acc'));
% load('/home/haichao/haichao/subLabel_kmeans/acc/image_click_Dog283_0_click_non1_ND_fdatabase.mat')
% idx = NsubIndex;
% image_index = find(ismember(fdatabase1.label, idx));
% acc2 = getACC(img_Fea_Clickcount,fdatabase1,image_index,IDX);

end
function label = find_top_n(vf_m,map,i_t,i_tr,K)

[~, idx] = sort(vf_m, 1);
    irow = idx([1:K],:);
    lab = [];
    for i = 1 : K
        la = map(irow(i,:));
        lab = [lab, la];
    end
    if K >1
        label = mode(lab');
    else
        label = lab;
    end
    label(i_t) = map(i_tr); 
end
function [map_query_test,map_query_train] = get_Mquery_datafea(map,query_train_datafea,query_test_datafea,map_test_qn)
map_query_test = zeros(size(query_test_datafea,1),max(map));
for i = 1 : max(map)
    tr_ind = find(map == i);
    map_query_tr = query_train_datafea(:,tr_ind);
    map_query_train(:,i) = sum(map_query_tr,2);
    
    t_ind = find(map_test_qn == i);
    if ~isempty(t_ind)
%         map_query_test(:,i) = 0;
%     else
        map_query_t = query_test_datafea(:,t_ind);
        map_query_test(:,i) = sum(map_query_t,2);
    end
end
end
% function [min_c,train_label] = get_test_label(map_query_train,map_query_test,train_feature,NsubIndex)
%%function [min_c,train_label] = get_test_label(map_query_train,map_query_test,train_feature,test_feature,NsubIndex)
% map_t2t = EuDist2(map_query_train,map_query_test);%% unnormalize
% [min_r min_c] = min(map_t2t);
% num_train = cellfun(@numel,train_feature(:,2),'Uniformoutput',false);
% num_tr = cell2mat(num_train);
% for i = 1 : length(num_tr)
%     if i == 1
%         num_tr(i,2) = 1;
%         num_tr(i,3) = num_tr(i,1);
%     else
%         num_tr(i,2) = 1+ num_tr(i-1,3);
%         num_tr(i,3) = num_tr(i-1,3) + num_tr(i,1);
%     end
% end
% num_tr = [num_tr,NsubIndex(:)];
% for i = 1 : length(num_tr)
%     left = num_tr(i,2);
%     right = num_tr(i,3);
%     la = num_tr(i,4);
%     train_label(left:right) = la;
% end
% end