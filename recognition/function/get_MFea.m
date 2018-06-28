function MFea = get_MFea( aug_data,query_datafea, NClass, NSClass)
%function [query_datafea MFea query_col] = get_MFea(index_21w_2_9w, aug_data, data_fea, NClass, NSClass)
aug_data  = aug_data';
aug_data = aug_data(:);% a label(contains 20 sub_label) connect the next label(contains 20 sub_label)

%%to Tindex for get MFea from TrnasData

Tindex = [];
l = 0;
for i = 1:length(aug_data)
    r = length(aug_data{i})+l;
    Tindex(l+1:r) = i;
    l = r;
end

%%to data_fea for MFea from TrnasData
% datafea= [];
%  for j = 1:length(aug_data)
%      datafe = data_fea(index_21w_2_9w(aug_data{j}),:);
%      datafea = cat(1,datafea,datafe);
%  end
%  
% %%to the query for train data
% query_col = find(sum(datafea)>0);
% query_datafea = datafea(:,query_col);
query_datafea = bsxfun(@times, query_datafea,...
        1./sum(query_datafea));
[NImg,NQue] = size(query_datafea);
MFea = TrnasData(NImg, NQue, NClass, NSClass, query_datafea, Tindex);
end