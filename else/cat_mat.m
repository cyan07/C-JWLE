clear all;
mandir = '/data/haichao/to_augmentation/images';
sudir  = dir( mandir );
nu = 0;
for i = 1 : length( sudir )
    if( isequal( sudir( i ).name, '.' )||...
            isequal( sudir( i ).name, '..')||...
            ~sudir( i ).isdir )              
            continue; 
    end
    D = dir( fullfile( mandir, sudir(i).name));
    nu = nu + length(D)-2;
    if ( length( D ) > 6000 )%&& length( D ) <= 6000 )
        name{i,1} = sudir( i ).name;
        name{i,2} = length( D );
    end
end
name(cellfun(@isempty,name))=[];
maindir = '/data/haichao/extract_feature/extract_feature/dog';
% feature = [];
% for i = 3 : length(name)
%     feadir = dir(fullfile(maindir,name{i,1},'feature','*.mat'));
%     A = load(fullfile(maindir,name{i,1},'feature',feadir(1).name));
%     feature{i-2} = A.feature;
% end
%maindir = '/home/haichao/Dog/extract_feature';
% rootdir = '/home/haichao/caffe/models/VGG';
% maindir = '/home/haichao/caffe/models/VGG/extract_feature';
subdir  = dir( maindir );
l = length( name )/2;
for j = 1 : l
    for i = 1 : length( subdir )
        if( isequal( subdir( i ).name, '.' )||...
            isequal( subdir( i ).name, '..')||...
            ~subdir( i ).isdir )
            continue;                       % 如果不是目录则跳过
        end 
        if (strcmp( subdir( i ).name, name{1,j}) == 1 )
            feafir = dir( fullfile( maindir, subdir( i ).name, 'feature','*.mat' ));
            A = load( fullfile( maindir, subdir( i ).name, 'feature', feafir(1).name ));
            B = load( fullfile( maindir, subdir( i ).name, 'feature', feafir(2).name ));
            D = load( fullfile( maindir, subdir( i ).name, 'feature', feafir(3).name ));
            E = load( fullfile( maindir, subdir( i ).name, 'feature', feafir(4).name ));
            A1 = A.feature;
            B1 = B.feature;
            D1 = D.feature;
            E1 = E.feature;
            C = cat( 1,A1 ,B1 );
            C = cat( 1,C ,D1 );
            C = cat( 1,C ,E1 );
            feature = C;
            [m,n] = size(feature)
            save( fullfile( maindir, subdir( i ).name, 'feature', ['output' num2str(m) '.mat']), 'feature');
            delete( fullfile( maindir, subdir( i ).name, 'feature', feafir(1).name ));
            delete( fullfile( maindir, subdir( i ).name, 'feature', feafir(2).name ));
            delete( fullfile( maindir, subdir( i ).name, 'feature', feafir(3).name ));
            break;
        end
    end
end
