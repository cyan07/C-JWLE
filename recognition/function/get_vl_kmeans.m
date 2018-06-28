function get_vl_kmeans(k)
addpath( genpath( '/home/haichao/haichao/subLabel_kmeans' ));
load ( 'image_click_Dog283_0_database.mat','database' );
maindir = '/home/haichao/181/Dog/extract_feature';
subdir  = dir( maindir );
rootdir = '/home/haichao/haichao';
for j = 1 : length( database.cname )
    c = database.cname {j};
    c( find(c==' ')) = '_';
    for i = 1 : length( subdir )
        
        if( isequal( subdir( i ).name, '.' )||...
            isequal( subdir( i ).name, '..')||...
            ~subdir( i ).isdir )              
            continue;                       % ??????????????????????????????????
        end 
        if (strcmp( subdir( i ).name, c) == 1 )
            feafir = dir( fullfile( maindir, subdir( i ).name, 'feature','*.mat' ));
            sub_aug_data_fea = load( fullfile( maindir, subdir( i ).name, 'feature', feafir.name ));
            aug_feature{j} = sub_aug_data_fea.feature;
           

            feature = aug_feature{j};
     
            [sub_aug_clasf_center{j,1}, sub_aug_clasf_center{j,2}] =normal_vl_kmeans( feature ,k*2);
            
            break;
        end
    end
end
aug_feature = aug_feature(:);

j = 0;
 for i = 1 : length( sub_aug_clasf_center )
     for l = 1 : length( sub_aug_clasf_center{i,2} )
         j = j + 1;
         sub_aug_clasf_center{i,2}(2,l) = j;
     end
 end
 j = 0;
 for i = 1 : length( aug_feature )
     for l = 1 : size( aug_feature{i},1 )
         j = j + 1;
         aug_feature{i,2}(l) = j;
     end
 end

save( fullfile( rootdir, 'subLabel_kmeans', ['vl_kmeans_' num2str(k*2) '.mat']), 'sub_aug_clasf_center', 'aug_feature', '-v7.3' );
end