function [rpQSum data mdata] = Aft_propagate(data_fea,sim,subsim,MFea,perc,k_N,k_NS)

NClass = length(subsim);
NSClass = length(subsim{1});    
sub_sim = blkdiag(subsim{:});
s = sub_sim;
[row,col] = find(sub_sim == 0);
ind = sub2ind(size(s),row,col);
s(ind) = inf;
Mfea = cell2mat(MFea);
[ln qn] = size(Mfea);
for i = 1 : qn
    pQsum = full(reshape(Mfea(:,i),NSClass,NClass));
    pQSum{i} = GetPropagate_W_v3(pQsum,perc,sim,{[1:NClass]},k_N);
end
for i = 1 : qn
    rpQsum{i} = reshape(pQSum{i},1,NSClass*NClass);
    %rpQSum{i} = GetPropagate_W(pQsum,perc,sim,{[1:NClass]});
end
rpQSum = cell2mat(rpQsum(:));
for i = 1 : NClass
    rpIndex{i}= [(i-1)*NSClass+1:i*NSClass];
end
% for i = 1 : qn
%     rpQSum{i} = GetPropagate_W_v3(rpQsum{i},perc,s,rpIndex,k_NS);
% end
rpQSum = GetPropagate_W_v3(rpQSum,perc,s,rpIndex,k_NS);
NQue = size(data_fea,2);
data = reshape(rpQSum',NSClass,NClass,1,NQue);
for i = 1: NQue
    da = data(:,:,1,i);
    index = find(da(:));
    index = index(:);index = index';
    daN = mapminmax(da(index), 70, 255);
    da(index) = daN;
    mdata(:,:,1,i) = da;
end
end