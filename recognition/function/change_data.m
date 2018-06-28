function data = change_data(data_fea,subsim,MFea)

NClass = length(subsim);
NSClass = length(subsim{1});    
Mfea = cell2mat(MFea);
[ln qn] = size(Mfea);
for i = 1 : qn
    pQsum{i} = full(reshape(Mfea(:,i),NSClass,NClass));
    rpQsum{i} = reshape(pQsum{i},1,NSClass*NClass);
end
rpQSum = cell2mat(rpQsum(:));

% for i = 1 : qn
%     rpQSum{i} = GetPropagate_W_v3(rpQsum{i},perc,s,rpIndex,k_NS);
% end
%rpQSum = GetPropagate_W_v3(rpQSum,perc,s,rpIndex,k_NS);
NQue = size(data_fea,2);
data = reshape(rpQSum',NSClass,NClass,1,NQue);
end