function [all_fea,feature] = arrange_fea(fea_dir,img_dir)
if nargin <1
    fea_dir = '/data/haichao/extract_feature/extract_feature/dog';
end
if nargin <2
    img_dir = '/data/haichao/to_augmentation/images';
end

all_fea = [];
%fdir = dir( fea_dir );
idir = dir( img_dir );
for i = 1 : length( idir )
    if( isequal( idir( i ).name, '.' )||...
            isequal( idir( i ).name, '..')||...
            ~idir( i ).isdir )              
            continue; 
    end
    img = dir( fullfile( img_dir, idir(i).name,'*.jpg'));
    na = cellfun(@(x) strsplit(x,'.'),{img.name},'Uniformoutput',false);
    nam = na{1}(1);
    for j = 2 : length(na)
        temp = na{j}(1);
        nam = [nam;temp];
    end
    [fea_name,l] = textread(fullfile( fea_dir, idir(i).name,'classify','classify_output.txt'),'%s%d');
    [b,c] = ismember(nam,fea_name);
    sub_dir = dir(fullfile( fea_dir, idir(i).name,'feature','*.mat'));
    sub_fea = load(fullfile( fea_dir, idir(i).name,'feature',sub_dir.name));
    f = sub_fea.feature;
    fea = f(c,:);
    all_fea{i-2} = fea;
end
feature = cell2mat(all_fea');
save('/data/haichao/buffer-memory/feature.mat','all_fea','feature','-v7.3');
end