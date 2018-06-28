
%feature: a cell 1800*1 ,each element is a vector 1*17w
%NClass: the number of labels
%NSClass: the number of sub_labels
function [ss Index] = sum_click( feature, NClass, NSClass)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

a = cell2mat(feature);%original Mfea[1600*5671];
a = full(a);

a = normal_fea(a);

a = reshape(a, [NSClass, NClass, size(a,2)]);
ss= reshape(sum(a, 1), [NClass, size(a,3)]);
Index = [[0:NClass-1]*NSClass+1;[1:NClass]*NSClass ];
Index = Index(:);

%  a = permute(a, [2 1 3]);
%  a = reshape(a, [NSClass*NClass, size(a,3)]);
%  Index = [[0:NClass-1]*NSClass+1;[1:NClass]*NSClass ];
%  Index = Index(:);
%  for i = 0 : length(Index)/2-1
%      j = 2 * i;
%      s(i+1,:) = sum(a(Index(j+1):Index(j+2),:),1);
%  end
%  ss = full(s); 


end

