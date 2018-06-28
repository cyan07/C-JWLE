% NImg =  10000;
% NQue = 2000;
% NClass = 283;
% NSClass = 20;
% MFea = rand([NImg, NQue]);
% Tindex = ceil(rand([NImg, 1])*NClass*NSClass);
% MFea_T = TrnasData(NImg, NQue, NClass, NSClass, MFea, Tindex)
function MFea = TrnasData(NImg, NQue, NClass, NSClass, MFea, Tindex)
SCNum = NClass*NSClass;
TindexC = mat2cell([1:SCNum]', ones(SCNum, 1), 1);
TindexC = cellfun(@(x) find(Tindex==x), TindexC,...
    'UniformOutput', false);
MFea = cellfun(@(x) sum(MFea(x,:), 1), TindexC,...
    'UniformOutput', 0);
end
%MFea = reshape(cell2mat(MFea), [NClass, NSClass, size(MFea{1}, 2)]);