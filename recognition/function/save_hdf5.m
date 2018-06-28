
%rootdir: the address to save hdf5 document
%data: the data to save
%NClusters: the number of labels
%NQue: the number of Query
function save_dir = save_hdf5(rootdir,data,NClusters, NQue  )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
rootdir = pwd;
list = dir(fullfile(rootdir,'*.mat'));
load(list.name);
dat = single(train_data);
labels = IDX(:,1);
%labels = single(1+round((NClusters-1)*rand(NQue,1)));

h5create(fullfile(rootdir,'data4torch.h5'),'/data',size(dat));
h5write(fullfile(rootdir,'data4torch.h5'),'/data',dat);
h5create(fullfile(rootdir,'data4torch.h5'),'/labels',length(labels), 'Datatype','single'); 
h5write(fullfile(rootdir,'data4torch.h5'),'/labels',labels,[1],[length(labels)]);
save_dir = fullfile(rootdir,'data4torch.h5');
end

