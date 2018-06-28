%clear all;
function save_dir = get_dataset(Nsub, k, perc, cl, k_N, k_NS)

if nargin <1
    Nsub = 129;
end
if nargin <2
    k = 30;
end
if nargin <3
    perc = 0.8;
end
if nargin <4
    cl = 3;
end
if nargin <5
    k_N = 129;
end
if nargin <6
    k_NS = 15;
end
% addpath( genpath( '/home/haichao/haichao/subLabel_kmeans' ));
addpath( genpath( '/home/haichao/experiment/vlfeat-0.9.20/toolbox' ));
addpath( genpath( '/home/haichao/experiment/function' ));
%addpath( genpath( 'new' ));
vl_setup;
rootdir = '/home/haichao/experiment/subLabel_kmeans';
%/home/haichao/haichao/haichao';
load('/home/haichao/experiment/subLabel_kmeans/alter/NsubIndex_129.mat')
str = ['new_rand_idx_' num2str(min(NsubIndex)) '-' num2str(max(NsubIndex))];
NClass = Nsub;NSClass = k;

load('/home/haichao/experiment/img_id_train_test_129.mat')%the id for train and test in 95041
load('/home/haichao/experiment/image_class_labels.mat');%the label for 95041
load('/home/haichao/experiment/feature.mat','feature');%the visual feature of aug_set,15w*4096    
load('/home/haichao/experiment/index_15w_2_9w.mat','index_15w_2_9w');%the id of aug_set in original set(95041)
load('/home/haichao/experiment/image_click_Dog283_0_img_Fea_Clickcount.mat','img_Fea_ClickCount');%9.5w*48w
load('/home/haichao/experiment/image_click_Dog283_0_CNN_Alex1_ND_S_S1_data_normal.mat','data_fea');%95041*4096
%load('image_id_train_test.mat')%the id for train and test in 95041
%load('data/haichao/subLabel_kmeans/new/image_class_labels.mat');%the label for 95041
%load('/data/haichao/subLabel_kmeans/283_feature.mat','feature');%the visual feature of aug_set,15w*4096    
% load ('/data/haichao/subLabel_kmeans/new/index_15w_2_9w.mat','index_15w_2_9w');%the id of aug_set in original set(95041)
% load('/data/haichao/subLabel_kmeans/new/image_click_Dog283_0_img_Fea_Clickcount.mat','img_Fea_ClickCount');%9.5w*48w
% load('/data/haichao/subLabel_kmeans/new/image_click_Dog283_0_CNN_Alex1_ND_S_S1_data_normal.mat','data_fea');%95041*4096
% load('image_click_Dog283_0_click_non1_ND_fdatabase.mat', 'fdatabase1')%
clear Valtset; 

load('/home/haichao/experiment/aug_train_dataset.mat')
%load('/home/haichao/experiment/img_id_train_test_129.mat')
load('/home/haichao/experiment/train_test_label_129.mat')
%load('/home/haichao/haichao/subLabel_kmeans/alter/img_id_train_test_129.mat')
%load('/home/haichao/haichao/subLabel_kmeans/alter/train_test_label_129.mat')

click_fea = img_Fea_ClickCount;

try
load( fullfile( rootdir, ['alter/train_fea_' num2str(Nsub) '_' num2str(cl) '_'  str '.mat']));%,'sub_train','sub_test','sub_test_id','sub_train_id','aug_train_id','train_click_col','test_click_col','-v7.3');
load(fullfile( rootdir,['alter/test_datafea_' num2str(Nsub) '_' num2str(cl) '_'  str '.mat']));%,'aug_train_feature','train_click_fea', 'tr_click_fea','test_click_fea','t_click_fea','sub_train_clickfea','-v7.3');
%load(fullfile( rootdir,['alter/sub_train_test_label_' num2str(Nsub) '_' num2str(cl) '_'  str '.mat']));%,'train_label','test_true_label');
load(fullfile( rootdir, 'alter', ['query_intersect_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']));%, 'i_q', 'i_tr', 'i_t', '-v7.3');
catch
sub_train = [trainfilenameindex,type_train];
sub_test = [testfilenameindex,type_test];
sub_test_id = testfilenameindex;
% train_index = cellfun(@(x) index_21w_2_9w(x),snumtrain,'UniformOutput',false);
% aug_train_id = [];% id in 9w(org)
% for i = 1 : length(train_index)
%     temp = train_index{i};
%     aug_train_id = [aug_train_id(:);temp(:)];
% end
train_id = cell2mat(snumtrain); % id in 15w(aug)

aug_train_feature = feature(train_id,:);
%aug_train_feature = bsxfun( @times, aug_train_feature, 1./sqrt(sum(aug_train_feature.^2,2)) );%normalization  
sub_train_feature = data_fea(trainfilenameindex,:);
%sub_train_feature  = bsxfun( @times, sub_train_feature, 1./sqrt(sum(sub_train_feature.^2,2)) );%normalization  
sub_test_feature = data_fea(sub_test_id,:);
sub_test_feature  = bsxfun( @times, sub_test_feature, 1./sqrt(sum(sub_test_feature.^2,2)) );%normalization  
test_true_label = type_test;
% train_label = label(aug_train_id,2);
sub_train_id = trainfilenameindex;

%50% in each query
st_click_fea = click_fea(sub_train_id,:);
temp = 1;

for i = 1 : Nsub
    ind = find(sub_train(:,2)==NsubIndex(i));
    %temp = sub_train(ind,1);
    click = st_click_fea(ind,:);
    click_col = find(sum(click)>cl);%2&sum(click)<cl);
    c = [click_col(:),full(sum(click(:,click_col)))'];
    [a,b] = sortrows(c,-2);
    click_f{i,1} = click(:,a([1:floor(length(click_col)/temp)],1));
    click_f{i,2} = a([1:floor(length(click_col)/temp)],1)';
end
train_click_col = unique(cell2mat(click_f(:,2)'));

% f10 =find(full(sum(st_click_fea))>cl);
% Q_dNUM
% tem = click_f(:,2);
% tm =sort(cell2mat(cellfun(@length,tem,'UniformOutput',false)));

%train_click_col = find(sum(st_click_fea)>cl);   %100%
st_click_fea = st_click_fea(:,train_click_col);
%sub_train_clickfea = bsxfun( @times, st_click_fea, 1./sum(st_click_fea,1) );    
tr_click_fea = click_fea(aug_train_id,:);
%train_click_col = find(sum(tr_click_fea)>cl);
tr_click_fea = tr_click_fea(:,train_click_col);
%train_click_fea = bsxfun( @times, tr_click_fea, 1./sum(tr_click_fea,1) );    

t_click_fea = click_fea(sub_test_id,:);
test_click_col = find(sum(t_click_fea)>0);%2&sum(t_click_fea)<cl);
t_click_fea = t_click_fea(:,test_click_col);
test_click_fea = bsxfun( @times, t_click_fea, 1./sum(t_click_fea,1) );%normalization by Query   

save( fullfile( rootdir, ['alter/train_fea_' num2str(Nsub) '_' num2str(cl) '_'  str '.mat']),'sub_train','sub_train_feature','sub_train_id','aug_train_feature','aug_train_id','train_click_col','st_click_fea','tr_click_fea','train_label','-v7.3');
save(fullfile( rootdir,['alter/test_datafea_' num2str(Nsub) '_' num2str(cl) '_'  str '.mat']),'sub_test_feature','sub_test','test_click_fea','test_click_col','sub_test_id','test_true_label','-v7.3');
%save(fullfile( rootdir,['alter/sub_train_test_label_' num2str(Nsub)  '_' num2str(cl) '_'  str '.mat']),'train_label','test_true_label');

%the same click col between train and test
[i_q, i_tr, i_t] = intersect(train_click_col,test_click_col);
save(fullfile( rootdir, 'alter', ['query_intersect_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']), 'i_q', 'i_tr', 'i_t', '-v7.3');
end

load('/home/haichao/experiment/new_snumtrain.mat');
load('/home/haichao/experiment/subLabel_kmeans/alter/new_image_id_col_129.mat');%,'new_aug_train_col','new_sub_train_col');
try 
    load(fullfile(  rootdir, 'alter', ['train_feature_' num2str(Nsub) '_' num2str(cl) '_'  str '.mat']))%, 'train_feature', 'sub_train_feature', '-v7.3');
    %load(fullfile(  rootdir, 'alter', ['test_feature_' num2str(Nsub) '_' num2str(cl) '_'  str '.mat']))%, 'test_feature', 'sub_test_feature', '-v7.3');
    load(fullfile(  rootdir, 'alter', ['train_click_feature_' num2str(Nsub) '_' num2str(cl) '_'  str '.mat']))%, 'sub_train_clickfea', 'train_click_fea', '-v7.3');
    %load(fullfile(  rootdir, 'alter', ['train_test_label_' num2str(Nsub) '_' num2str(cl) '_'  str '.mat']))%, 'sub_train', 'train_label')%, '-v7.3');
catch  
%aug_train
aug_train_id = aug_train_id(new_aug_train_col);
aug_train_feature = aug_train_feature(new_aug_train_col,:);
aug_train_feature = bsxfun( @times, aug_train_feature, 1./sqrt(sum(aug_train_feature.^2,2)) );%normalization  
train_click_fea = tr_click_fea(new_aug_train_col,:);
train_click_fea = bsxfun( @times, train_click_fea, 1./sum(train_click_fea,1) );    

train_label = train_label(new_aug_train_col);

%sub_train
sub_train = sub_train(new_sub_train_col,:);
sub_train_feature = sub_train_feature(new_sub_train_col,:);
sub_train_feature  = bsxfun( @times, sub_train_feature, 1./sqrt(sum(sub_train_feature.^2,2)) );%normalization  


sub_train_clickfea = st_click_fea(new_sub_train_col,:);
sub_train_clickfea = bsxfun( @times, sub_train_clickfea, 1./sum(sub_train_clickfea,1) );    

% %test
% sub_test = sub_test(new_test_col,:);
% sub_test_id = sub_test_id(new_test_col);
% sub_test_feature = sub_test_feature(new_test_col,:);

train_feature = get_fea(NsubIndex,sub_train,aug_train_id,aug_train_feature);
save( fullfile(  rootdir, 'alter', ['train_feature_' num2str(Nsub) '_' num2str(cl) '_'  str '.mat']), 'sub_train','aug_train_id','train_label','aug_train_feature','train_feature','test_feature', 'sub_train_feature', '-v7.3');
test_feature = get_fea(NsubIndex,sub_test,sub_test_id,sub_test_feature);
%save( fullfile(  rootdir, 'alter', ['test_feature_' num2str(Nsub) '_' num2str(cl) '_'  str '.mat']), 'test_feature', 'sub_test_feature', '-v7.3');
save( fullfile(  rootdir, 'alter', ['train_click_feature_' num2str(Nsub) '_' num2str(cl) '_'  str '.mat']), 'sub_train_clickfea', 'train_click_fea', '-v7.3');
%save( fullfile(  rootdir, 'alter', ['train_test_label_' num2str(Nsub) '_' num2str(cl) '_'  str '.mat']), 'sub_train', 'train_label')%, '-v7.3');
end

try
    load(fullfile( rootdir, 'alter/kmeans/perc', ['train_profea_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']),'train_data');%, 'train_mdata', 'train_profea', '-v7.3');
    load(fullfile( rootdir, 'alter/kmeans', ['train_datafea_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' str '.mat']));
    load(fullfile(  rootdir, 'alter/kmeans', ['train_feature_set_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_'  str '.mat']));
catch
% km=[10,20,30,40,60];
% %a_rate = [0.2:0.2:0.8,0.9];
 for i = 1 : length(km)
    NSClass = km(i);
    train_feature_set = get_feature_set(train_feature,NSClass);
    for j = 1 : length(train_feature_set)
        train_feature_set{j,2} = [train_feature_set{j,2};new_snumtrain{j}];
    end
    save( fullfile(  rootdir, 'alter/kmeans', ['train_feature_set_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_'  str '.mat']),'train_feature_set', '-v7.3');
end
% 
%  for i = 1 : length(km)
%     NSClass = km(i);
     load(['/home/haichao/experiment/subLabel_kmeans/alter/kmeans/train_feature_set_129_',num2str(cl),'_',num2str(NSClass),'_new_rand_idx_1-283.mat'])
% %test_feature_set = get_featureset(test_feature,NSClass);
% 
     [train_sim, train_subsim, train_rsort, aug_train, sort_map_org] = get_parama(train_feature_set, NSClass, index_15w_2_9w); 
% %[test_sim, test_subsim, test_rsort,~, sort_map_org] = get_param(train_feature_set, NSClass, aug_train_id); 
% 
% %[~, ~, ~, aug_test,~] = get_param(test_feature_set, NSClass, index_21w_2_9w); 
% 
%     %%get some parameter for train data
     train_MFea = get_MFea( aug_train, train_click_fea, NClass, NSClass);
%     save( fullfile( rootdir, 'alter/kmeans', ['train_datafea_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' str '.mat']), 'train_MFea', 'train_sim', 'train_subsim', 'train_rsort', 'aug_train', 'sort_map_org', '-v7.3');
% end
%  load(['/home/haichao/experiment/subLabel_kmeans/alter/kmeans/train_datafea_129_',num2str(cl),'_',num2str(NSClass),'_new_rand_idx_1-283.mat']);
%     BP_data = change_data(train_click_fea,train_subsim,train_MFea);
%  save( fullfile( rootdir, 'alter/kmeans', ['BP_data_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'BP_data');%train_MFea', 'train_sim', 'train_subsim', 'train_rsort', 'aug_train', 'sort_map_org', '-v7.3');

%%propagate train_datafea
% k_S = [5];%,10];%,15,20,30];
% a_rate = 0.5;%[0.2,0.4,0.6,0.8,0.9];
% for i = 1 : length(k_S)
%    for j = 1 : length(a_rate)
%  %NSClass = km(i);
%     % temp = 0.5;
%     %k_NS = NSClass*temp;
%        k_NS = k_S(i);
%        perc = a_rate(j);

    %load(['/home/haichao/experiment/subLabel_kmeans/alter/kmeans/train_datafea_129_',num2str(cl),'_',num2str(NSClass),'_new_rand_idx_1-283.mat']);
    [train_profea train_data ~] = Aft_propagate( train_click_fea,train_sim,train_subsim,train_MFea,perc,k_N,k_NS);
    save(fullfile( rootdir, 'alter/kmeans/perc', ['train_profea_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']),'train_data', 'train_profea', '-v7.3');
    %[train_query_label train_map_label train_num_label] = get_Qlabel(train_rsort,train_MFea,NSClass);
    %save(fullfile( rootdir, 'alter/kmeans/perc', ['train_data_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'train_query_label', 'train_map_label', 'train_num_label', '-v7.3');
% end
end
%torch_org
for i = 1 : size(train_data,4)
   m = train_data(:,:,1,i);
   nm = imresize(m',[112,92]);
   data(:,:,1,i)= nm;
end
save(fullfile( rootdir, 'alter/', ['torch_org_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']),'data', '-v7.3')

%vgg 
for i = 1 : size(train_data,4)
   m = train_data(:,:,1,i);
   nm = imresize(m,[224,224]);
   vgg_data(i,1,:,:) = nm;
end
save(fullfile( rootdir, 'alter/', ['vgg_data_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']),'vgg_data', '-v7.3')
% DEPICT_org
for i = 1 : size(train_data,4)
   m = train_data(:,:,1,i);
   nm = imresize(m,[32,32]);
   Do_data(:,:,1,i) = nm;
end
save(fullfile( rootdir, 'alter/', ['Do_data_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']),'Do_data', '-v7.3')

% load('/data/haichao/subLabel_kmeans/new/subLabel_kmeans/alter/query_intersect_129_50%_new_rand_idx_1-283.mat')
% load('/data/haichao/subLabel_kmeans/new/subLabel_kmeans/kmeans/train_datafea_129_80_new_rand_idx_1-283.mat')
% load('/data/haichao/subLabel_kmeans/new/subLabel_kmeans/kmeans/perc/train_data_129_80_0.8_129_40_new_rand_idx_1-283.mat')
Q_NUM = 500;
%Q_dNUM = length(train_map_label{1,2}) + Q_NUM;
Q_dNUM = Q_NUM;
%end
% for M2 map.
t = k_NS/NSClass;
x = sum( train_click_fea ,1);
x_fea = bsxfun( @times, train_click_fea, 1./x ); 
Pro_fea = pro_fea(train_feature(:,1),x_fea,perc,t);
Pro_fea = cell2mat(Pro_fea);
y = sqrt( sum( test_click_fea ,1));
y_fea = bsxfun( @times, test_click_fea, 1./y ); 
Pro_testfea = pro_fea(test_feature(:,1),y_fea,perc,t);
Pro_testfea = cell2mat(Pro_testfea);
save(fullfile( rootdir, 'alter', ['train_profea_M2_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' str '.mat']), 'Pro_fea', '-v7.3');
save(fullfile( rootdir, 'alter', ['test_profea_M2_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' str '.mat']), 'Pro_testfea', '-v7.3');

%%map
%Na map: not augmentation
vf_m = find_label_wvf(sub_test_feature,test_click_fea,sub_train_feature,sub_train_clickfea );
[vf_r vf_c] = min(vf_m(:,[1:length(vf_m)/2]));
query_id = diag(vf_m(vf_c,[length(vf_m)/2+1:end]));
save(fullfile( rootdir, 'alter/map', ['test_map_train_Na_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']), 'query_id', '-v7.3');
% %M1 map : augmetation
% vf_m = find_label_wvf(sub_test_feature,test_click_fea,aug_train_feature,train_click_fea );
% [vf_r vf_c] = min(vf_m(:,[1:length(vf_m)/2]));
% query_id = diag(vf_m(vf_c,[length(vf_m)/2+1:end]));
% save(fullfile( rootdir, 'alter/map', ['test_map_train_M1_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']), 'query_id', '-v7.3');
% %M2 map : augmentation and propagation
% vf_m= find_label_wvf(sub_test_feature,Pro_testfea',aug_train_feature,Pro_fea' );
% [vf_r vf_c] = min(vf_m(:,[1:length(vf_m)/2]));
% query_id = diag(vf_m(vf_c,[length(vf_m)/2+1:end]));
% save(fullfile( rootdir, 'alter/map', ['test_map_train_M2_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']), 'query_id' , '-v7.3');
% %M3 map : augmentation and propagation and stuctural

% %%acc
% %not separate train and test query, mix them together
% %Na method: not augmentation
% acc = get_NTt_acc(click_fea,train_click_col,test_click_col,sub_train(:,1),sub_test_id,Q_dNUM,sub_train(:,2),test_true_label)
% save(fullfile( rootdir, 'alter/acc', ['acc_top1_NTt_Na_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');
% %M1 method: augmentation
% acc = get_NTt_acc(click_fea,train_click_col,test_click_col,new_id,sub_test_id,Q_dNUM,train_label,test_true_label)
% save(fullfile( rootdir, 'alter/acc', ['acc_top1_NTt_M1_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');
% %M2 method: augmentation and propagate
% acc = get_NTt_pro_acc(data_fea,click_fea,train_click_col,test_click_col,new_id,sub_test,NsubIndex,Q_dNUM,train_label,test_true_label,perc,t)
% save(fullfile( rootdir, 'alter/acc', ['acc_top1_NTt_M2_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');
% %M3 method: augmentation and propagate and stuctural
% % for i = 1 : Nsub
% %     t = sub_test(test_feature{i,2},1);
% %     str_f{i} = [snumtrain{i},t'];
% % end
% acc = get_NTt_struc_acc(aug_train_feature,data_fea,click_fea,train_click_col,test_click_col,new_id,sub_test,NsubIndex,Q_dNUM,train_label,test_true_label,NSClass,perc,k_N,k_NS)
% save(fullfile( rootdir, 'alter/acc', ['acc_top1_NTt_M3_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');
% %M4 method: as input to torch
% % acc = get_NTt_acc(click_fea,train_click_col,test_click_col,aug_train_id,sub_test_id,train_map_label,train_label,test_true_label)
% % save(fullfile( rootdir, 'subLabel_kmeans/alter/acc', ['acc_top1_NTt_M2_' num2str(Nsub) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');

%%acc
%separate train and test query
%%not augmentation
x = sum(sub_train_clickfea ,1);
x_fea = bsxfun( @times, sub_train_clickfea, 1./x ); 
[Fea_s IDX_s ] = vl_kmeans(full(x_fea), Q_dNUM);

save(fullfile(rootdir,'alter/IDX',['IDX_Fea_kmeans_Na_',num2str(Nsub),'-',num2str(cl),'-',num2str(NSClass),'-',num2str(perc),'-',num2str(1),'-', str,'.mat']),'IDX_s','Fea_s','-v7.3');
IDX = [IDX_s;train_click_col];
ID =  IDX';
ID = sortrows(ID,2);
save(fullfile( rootdir, 'alter/IDX', ['query_IDX_Na_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'IDX','ID', '-v7.3');

%Na map:
load(fullfile(rootdir,'alter/map',['test_map_train_Na_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
acc = get_top_n_acc(test_true_label,sub_train(:,2),ID,query_id,i_t,i_tr,sub_train_clickfea,test_click_fea,1)
save(fullfile( rootdir, 'alter/acc', ['acc_top1_Na-Na_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');
% %M1 map
% load(fullfile(rootdir,'alter/map',['test_map_train_M1_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,sub_train(:,2),ID,query_id,i_t,i_tr,sub_train_clickfea,test_click_fea,1)
% save(fullfile( rootdir, 'alter/acc', ['acc_top1_M1-Na_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');
% %M2 map
% load(fullfile(rootdir,'alter/map',['test_map_train_M2_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,sub_train(:,2),ID,query_id,i_t,i_tr,sub_train_clickfea,test_click_fea,1)
% save(fullfile( rootdir, 'alter/acc', ['acc_top1_M2-Na_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');

%method 1,merge query straightly by kmeans
% a = train_map_label{1,1};
% Que_Fea_Clickcount_sing = train_click_fea(:,a);
F_ACC = [];
for i = 1 :5
Que_Fea_Clickcount_mult = train_click_fea;
% Que_Fea_Clickcount_mult(:,a) = [];
% x = sum( Que_Fea_Clickcount_sing ,1);
% x_fea = bsxfun( @times, Que_Fea_Clickcount_sing, 1./x ); 
% [Fea_sig IDX_sig ] = vl_kmeans(full(x_fea), Q_NUM/5);

y = sum( Que_Fea_Clickcount_mult ,1);
y_fea = bsxfun( @times, Que_Fea_Clickcount_mult, 1./y ); 
[Fea_mul IDX_mul ] = vl_kmeans(full(y_fea), Q_dNUM);
%save(fullfile(rootdir,'alter/IDX',['IDX_Fea_kmeans_M1_',num2str(Nsub),'-',num2str(cl),'-',num2str(NSClass),'-',num2str(perc),'-',num2str(1),'-', str,'.mat']),'Fea_mul','IDX_mul','-v7.3');%'IDX_mul','IDX_sig','Fea_mul','Fea_sig','-v7.3');
%[IDX ID] = cat_idx(IDX_sig, IDX_mul,train_map_label,train_click_col);
IDX = [IDX_mul;train_click_col];
ID =  IDX';
ID = sortrows(ID,2);
%save(fullfile( rootdir, 'alter/IDX', ['query_IDX_M1_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'IDX','ID', '-v7.3');

%Na map -M1
load(fullfile(rootdir,'alter/map',['test_map_train_Na_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))%;num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']))
acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
%save(fullfile( rootdir, 'alter/acc', ['acc_top1_Na-M1_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');
F_ACC = [F_ACC,acc];
end
% %M1 map -M1
% load(fullfile(rootdir,'alter/map',['test_map_train_M1_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))%num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'alter/acc', ['acc_top1_M1-M1_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');
% %M2 map -M1
% load(fullfile(rootdir,'alter/map',['test_map_train_M2_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))%'_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'alter/acc', ['acc_top1_M2-M1_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');
% %M3 map -M1
%%%

%method 2,propagating ,and then merge query by kmeans
F_ACC = [];
for i = 1 :5

[Fea_p IDX_p ] = vl_kmeans(Pro_fea', Q_dNUM);
%save(fullfile(rootdir,'alter/IDX',['IDX_Fea_kmeans_M2_',num2str(cl),'-',num2str(NSClass),'-',num2str(perc),'-',num2str(1),'-', str,'.mat']),'IDX_p','Fea_p','-v7.3');
IDX = [IDX_p;train_click_col];
ID =  IDX';
ID = sortrows(ID,2);
%save(fullfile( rootdir, 'alter/IDX', ['query_IDX_M2_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'IDX','ID', '-v7.3');
%train_vf =  aug_train_feature;

%Na map -M2
load(fullfile(rootdir,'alter/map',['test_map_train_Na_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))%num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']))
acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
%save(fullfile( rootdir, 'alter/acc', ['acc_top1_Na-M2_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');
F_ACC = [F_ACC,acc];
end
% %M1 map -M2
% load(fullfile(rootdir,'alter/map',['test_map_train_M1_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))%'_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'alter/acc', ['acc_top1_M1-M2_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');
% %M2 map -M2
% load(fullfile(rootdir,'alter/map',['test_map_train_M2_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'alter/acc', ['acc_top1_M2-M2_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');
%M3 map -M2
%%

% acc = getACC(click_fea(:, click_col),fdatabase1,image_index,IDX);
% structure without propagation 
F_ACC = [];
for i = 1 :5

data = BP_data;
    [m,n,l,o] = size(data);
    data = reshape(data, [m*n, o]);
    [Fea_d IDX_d ] = vl_kmeans(data, Q_dNUM);

    %save(fullfile(rootdir,['for_cluster_',num2str(Nsub),'_',num2str(cl),'-',num2str(NSClass),'-',num2str(perc),'-',num2str(k_NS),'-', str,'.mat']),'data','-v7.3');%,'IDX_md','Fea_md','-v7.3');
%save(fullfile(rootdir,'alter/IDX',['IDX_Fea_kmeans_M3_BP',num2str(Nsub),'_' num2str(cl),'-',num2str(NSClass),'-',num2str(perc),'-',num2str(k_NS),'-', str,'.mat']),'IDX_d','Fea_d','-v7.3');%,'IDX_md','Fea_md','-v7.3');

IDX = [IDX_d;train_click_col];
ID =  IDX';
ID = sortrows(ID,2);
%save(fullfile( rootdir, 'alter/IDX', ['query_IDX_M3_BP' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_NS) '_' str '.mat']), 'IDX','ID','-v7.3');%'m_ID','m_IDX', '-v7.3');
%Na map -M3
load(fullfile(rootdir,'alter/map',['test_map_train_Na_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
%save(fullfile( rootdir, 'alter/acc', ['acc_top1_Na-M3_BP' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_NS) '_' str '.mat']), 'acc')%,'m_acc');
F_ACC = [F_ACC,acc];
end
%%%
%method 3,structural data after propagation ,and then merge query by kmeans
%
F_ACC = [];
for i = 1 : 5
% for i = 1 : length(k_S)
%     for j = 1 : length(a_rate)
% %    NSClass = k_S(i);
%      temp = 0.5;
%      k_NS = NSClass*temp;
     %k_NS = k_S(i);
     %perc =a_rate(j);
    %load(fullfile('/home/haichao/experiment/subLabel_kmeans/alter/kmeans/perc',['train_profea_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_129_' num2str(k_NS) '_new_rand_idx_1-283.mat']), 'train_data')
    data = train_data;
    [m,n,l,o] = size(data);
    data = reshape(data, [m*n, o]);
    [Fea_d IDX_d ] = vl_kmeans(data, Q_dNUM);

    %save(fullfile(rootdir,['for_cluster_',num2str(Nsub),'_',num2str(cl),'-',num2str(NSClass),'-',num2str(perc),'-',num2str(k_NS),'-', str,'.mat']),'data','-v7.3');%,'IDX_md','Fea_md','-v7.3');
%save(fullfile(rootdir,'alter/IDX',['IDX_Fea_kmeans_M3_',num2str(Nsub),'_' num2str(cl),'-',num2str(NSClass),'-',num2str(perc),'-',num2str(k_NS),'-', str,'.mat']),'IDX_d','Fea_d','-v7.3');%,'IDX_md','Fea_md','-v7.3');

IDX = [IDX_d;train_click_col];
ID =  IDX';
ID = sortrows(ID,2);
%save(fullfile( rootdir, 'alter/IDX', ['query_IDX_M3_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_NS) '_' str '.mat']), 'IDX','ID','-v7.3');%'m_ID','m_IDX', '-v7.3');
%Na map -M3
load(fullfile(rootdir,'alter/map',['test_map_train_Na_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
%save(fullfile( rootdir, 'alter/acc', ['acc_top1_Na-M3_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_NS) '_' str '.mat']), 'acc')%,'m_acc');
% end
F_ACC = [F_ACC,acc];
    end
%end
save(fullfile( rootdir, 'alter/acc', ['F_ACC_top1_Na-M3_' num2str(Nsub) '_' num2str(cl) '_compare_' str '.mat']), 'F_ACC')%,'m_acc');

% %M1 map -M3
% load(fullfile(rootdir,'alter/map',['test_map_train_M1_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'alter/acc', ['acc_top1_M1-M3_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc','m_acc');
% %M2 map -M3
% load(fullfile(rootdir,'alter/map',['test_map_train_M2_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))%'_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'alter/acc', ['acc_top1_M2-M3_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc','m_acc');
% %M3 map -M3

%%
%method 4,input data to torch for merging query
% load('/home/haichao/experiment/subLabel_kmeans/torch/129_3_30_0.8_15/label_pre_0.2_0.01_0_1000_20_0.2_1.mat')
% learning_rate = power(10,-4);
% label = label_3;
ks = [0.2:0.2:1];
m_t= ks(5);
F_ACC = [];
IDX = [label;train_click_col];
ID =  IDX';
ID = sortrows(ID,2);
% load('/home/haichao/haichao/subLabel_kmeans/alter/IDX/label_mdata_1107.mat')
% m_IDX = [label;train_click_col];
% m_ID =  m_IDX';
% m_ID = sortrows(m_ID,2);
%save(fullfile( rootdir, 'torch/IDX', ['query_IDX_M4_128out_0.001_0.9_20_20_1' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'IDX','ID');%,'m_IDX','m_ID', '-v7.3');
%Na map -M4
load(fullfile(rootdir,'alter/map',['test_map_train_Na_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
F_ACC = [F_ACC,acc];
save(fullfile( rootdir, 'torch/acc', ['F_ACC_Na-M4_0.01_torch_label_1' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'F_ACC');%,'m_acc');
%save(fullfile( rootdir, 'torch/acc', ['acc_top1_Na-M4_128out_0.001_0.9_20_20_1' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');%,'m_acc');

% %M1 map -M4
% load(fullfile(rootdir,'alter/map',['test_map_train_M1_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'alter/acc', ['acc_top1_M1-M4_55_' num2str(Nsub) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');%,'m_acc');
% %M2 map -M4
% load(fullfile(rootdir,'alter/map',['test_map_train_M2_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'alter/acc', ['acc_top1_M2-M4_55_' num2str(Nsub) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');%,'m_acc');
% %M3 map -M4

%method 5,feature from torch for merging query
% load('/home/haichao/experiment/subLabel_kmeans/torch/129_3_30_0.8_15/feature/feature_pre_0.1_0_1000_20_0.2_1.mat');
% learning_rate = power(10,-1);
% ks = [20:20:100];
% Ks = ks(5);
% ks = [0.2:0.2:1];
% m_t= ks(3);
F_ACC = [];
% et = [0.2:0.2:0.8,0.9];
% eta = et(1);
for i = 1 : 5
%torch
[Fea_f IDX_f ] = vl_kmeans(feature, Q_dNUM);
% %DEPICT
% load(fullfile(rootdir,'DEPICT/FeaQuery.mat'));
%[Fea_f IDX_f ] = vl_kmeans(feature', Q_dNUM);
% %VGG
% vgg_fea = load(fullfile(rootdir,'VGG/haichao_feature.txt'));
%[Fea_f IDX_f ] = vl_kmeans(vgg_fea', Q_dNUM);
%save(fullfile(rootdir,'torch/IDX',['IDX_Fea_kmeans_M5_32-128Relu_',num2str(0.6),'-0.01-0.4-20-60',num2str(cl),'-',num2str(NSClass),'-',num2str(perc),'-' str,'.mat']),'IDX_f','Fea_f');%,'Fea_mf','IDX_mf','-v7.3');
%save(fullfile(rootdir,'torch',['IDX_Fea_kmeans_M5_cluster',num2str(cl),'-',num2str(NSClass),'-',num2str(perc),'-' str,'.mat']),'IDX_f');%,'Fea_f');%,'Fea_mf','IDX_mf','-v7.3');

IDX = [IDX_f;train_click_col];
ID =  IDX';
ID = sortrows(ID,2);
%save(fullfile( rootdir, 'torch/IDX', ['query_IDX_M5_W' num2str(0.6) '_0.01_0.4_20_60_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']),'ID', 'IDX');%,'m_IDX','m_ID', '-v7.3');
%Na map -M3
load(fullfile(rootdir,'alter/map',['test_map_train_Na_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
%load(fullfile(rootdir,'subLabel_kmeans/alter/map',['test_map_train_Na_' num2str(Nsub) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']))
acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
F_ACC = [F_ACC,acc];
end
save(fullfile( rootdir, 'torch/acc', ['F_ACC_Na-M5_0.01_torch_fea_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'F_ACC');%,'m_acc');
%save(fullfile( rootdir, 'torch/acc', ['acc_top1_Na-M5_32-128Relu_' num2str(0.6) '_0.01_0.4_20_60_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');%,'m_acc');

% %M1 map -M3
% load(fullfile(rootdir,'alter/map',['test_map_train_M1_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'alter/acc', ['acc_top1_M1-M5_55' num2str(Nsub) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');%,'m_acc');
% %M2 map -M3
% load(fullfile(rootdir,'alter/map',['test_map_train_M2_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'alter/acc', ['acc_top1_M2-M5_55' num2str(Nsub) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');%,'m_acc');
% %M3 map -M3

% %paint the cluster picture
% load('/home/haichao/experiment/subLabel_kmeans/torch/IDX_Fea_kmeans_M5_cluster3-30-0.8-new_rand_idx_1-283.mat');
% load('Dog_QQ_N.mat');
% Dog_name = Dog_QQ_N(train_click_col);
% ID = [1:500];
% for i = 1 : length(ID)
%     col = find(IDX_f==ID(i));
%     clu{i} = Dog_name(col);
% end
% save(fullfile( rootdir, ['cluster_name' num2str(Nsub) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_NS) '_' str '.mat']), 'clu');%,'m_acc');
% %Minibatchkmeans
% % load();
% IDX = [pre_clu;train_click_col];
% ID =  IDX';
% ID = sortrows(ID,2);
% save(fullfile( rootdir, 'minibatch-kmeans/IDX', ['query_IDX_MnBk_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'IDX','ID');%,'m_IDX','m_ID', '-v7.3');
% %Na map -MnBk
% load(fullfile(rootdir,'alter/map',['test_map_train_Na_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'minibatch-kmeans/acc', ['acc_top1_Na-MnBk_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');%,'m_acc');
% %M1 map -MnBk
% load(fullfile(rootdir,'alter/map',['test_map_train_M1_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'minibatch-kmeans/acc', ['acc_top1_M1-MnBk_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc')%,'m_acc');
% %M2 map -MnBk
% load(fullfile(rootdir,'alter/map',['test_map_train_M2_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))%'_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'minibatch-kmeans/acc', ['acc_top1_M2-MnBk_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc')%,'m_acc');
% %M3 map -M3
% 
% %Hierarchical
% % load();
% IDX = [c';train_click_col];
% ID =  IDX';
% ID = sortrows(ID,2);
% save(fullfile( rootdir, 'hierarchical-cluster/IDX', ['query_IDX_hie_clu_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'IDX','ID');%,'m_IDX','m_ID', '-v7.3');
% %Na map -hie_clu
% load(fullfile(rootdir,'alter/map',['test_map_train_Na_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'hierarchical-cluster/acc', ['acc_top1_Na-hie_clu_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');%,'m_acc');
% %M1 map -hie_clu
% load(fullfile(rootdir,'alter/map',['test_map_train_M1_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'hierarchical-cluster/acc', ['acc_top1_M1-hie_clu_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc')%,'m_acc');
% %M2 map -hie_clu
% load(fullfile(rootdir,'alter/map',['test_map_train_M2_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))%'_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'hierarchical-cluster/acc', ['acc_top1_M2-hie_clu_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc')%,'m_acc');

%compare by gj method
%load('/home/haichao/experiment/subLabel_kmeans/gj_compare/IDX.mat');
F_ACC = [];
IDX = [IDX';train_click_col];
ID =  IDX';
ID = sortrows(ID,2);
%save(fullfile( rootdir, 'hierarchical-cluster/IDX', ['query_IDX_by_gj_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'IDX','ID');%,'m_IDX','m_ID', '-v7.3');
%Na map -gj_clu
load(fullfile(rootdir,'alter/map',['test_map_train_Na_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
acc = get_top_n_acc(test_true_label,aug_train(:,2),ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
F_ACC = [F_ACC,acc];
save(fullfile( rootdir, 'hierarchical-cluster/acc', ['F_ACC_Na-by_gj_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'F_ACC');%,'m_acc');
%save(fullfile( rootdir, 'hierarchical-cluster/acc', ['acc_top1_Na-by_gj_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc');%,'m_acc');



% %M1 map -gj_clu
% load(fullfile(rootdir,'alter/map',['test_map_train_M1_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'hierarchical-cluster/acc', ['acc_top1_M1-hie_clu_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc')%,'m_acc');
% %M2 map -gj_clu
% load(fullfile(rootdir,'alter/map',['test_map_train_M2_' num2str(Nsub) '_' num2str(cl) '_' str '.mat']))%'_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']))
% acc = get_top_n_acc(test_true_label,train_label,ID,query_id,i_t,i_tr,train_click_fea,test_click_fea,1)
% save(fullfile( rootdir, 'hierarchical-cluster/acc', ['acc_top1_M2-hie_clu_' num2str(Nsub) '_' num2str(cl) '_' num2str(NSClass) '_' num2str(perc) '_' num2str(k_N) '_' num2str(k_NS) '_' str '.mat']), 'acc')%,'m_acc');
% 

%%%% rewrite by hc on 2017/11/02
end
% %%Kmeans clustering  
% try
%     load( fullfile( rootdir, 'subLabel_kmeans', ['vl_kmeans_' num2str(k*2) '.mat']));
% catch
%     
% for j = 1 : length( database.cname )
%     c = database.cname {j};
%     c( find(c==' ')) = '_';
%     for i = 1 : length( subdir )
%         
%         if( isequal( subdir( i ).name, '.' )||...
%             isequal( subdir( i ).name, '..')||...
%             ~subdir( i ).isdir )              
%             continue;                       % ??????????????????????????????????
%         end 
%         if (strcmp( subdir( i ).name, c) == 1 )
%             feafir = dir( fullfile( maindir, subdir( i ).name, 'feature','*.mat' ));
%             sub_aug_data_fea = load( fullfile( maindir, subdir( i ).name, 'feature', feafir.name ));
%             aug_feature{j} = sub_aug_data_fea.feature;
%            
% 
%             feature = aug_feature{j};
%      
%             [sub_aug_clasf_center{j,1}, sub_aug_clasf_center{j,2}] =normal_vl_kmeans( feature ,k*2);
%             
%             break;
%         end
%     end
% end
% aug_feature = aug_feature(:);
% 
% j = 0;
%  for i = 1 : length( sub_aug_clasf_center )
%      for l = 1 : length( sub_aug_clasf_center{i,2} )
%          j = j + 1;
%          sub_aug_clasf_center{i,2}(2,l) = j;
%      end
%  end
%  j = 0;
%  for i = 1 : length( aug_feature )
%      for l = 1 : size( aug_feature{i},1 )
%          j = j + 1;
%          aug_feature{i,2}(l) = j;
%      end
%  end
% 
% save( fullfile( rootdir, 'subLabel_kmeans', ['vl_kmeans_' num2str(k*2) '.mat']), 'sub_aug_clasf_center', 'aug_feature', '-v7.3' );
% end
%%%%ADD by TM    

function feature = get_fea(NsubIndex,sub,sub_id,sub_feature)
for i = 1 : length(NsubIndex)
    ins = find(sub(:,2)==NsubIndex(i));
    in_s = sub(ins,1);
    ind = get_id(sub_id,in_s);   
    feature{i,1} = sub_feature(ind,:);
    feature{i,2} = ind;
end
end

function a_id = get_id(x,y)
     a = ismember(x,y);
     a_id = find(a);   
end

% function feature_set = get_feature_set(feature,k)
% for j = 1 : length(feature)
%     fea = feature{j,1};
%     [feature_set{j,1}, feature_set{j,2}] =vl_kmeans(fea' ,k);
%     %[feature_set{j,1}, feature_set{j,2}] =normal_vl_kmeans(fea ,k);
% end
% for j = 1 : length(feature_set)
%     feature_set{j,2} = [feature_set{j,2};feature{j,2}];
% end
% end

function [IDX ID] = cat_idx(IDX_sig, IDX_mul,train_map_label,train_click_col)
IDX_S = [1:length(train_map_label{1,2})];
IDX_M = IDX_mul + length(IDX_S);%IDX_M = IDX_mul + 263;
a = train_map_label{1,1};
train_q = train_click_col;
a_sig = train_q(a);
train_q(a) = [];
IDX = [IDX_sig, IDX_M; a_sig, train_q]; 
ID =  IDX';
ID = sortrows(ID,2);
end


%%% rewrite by hc on 2017/09/24
function m = find_label_wvf(sub_test_feature,query_test_datafea,train_vf,query_train_datafea)
%function [m mrow mcol] = find_label_wvf(test_feature,query_test_datafea,train_vf,query_train_datafea)
% test_vf = test_feature(:,1);
% test_vf = cell2mat(test_vf);
test_vf = sub_test_feature;
m2 = multi_matrix(query_test_datafea,test_vf);
m1 = multi_matrix(query_train_datafea,train_vf);
%m =  m2 * m1';
f = floor(length(m1)/10);
m = [];
for i = 1 : 10
    m0 = EuDist2(m1([(i-1)*f+1:i*f],:), m2);
    [mrow mcol] = min(m0);
    m = [m;mrow,mcol+(i-1)*f];
end
end


% function [m mrow mcol] = find_label_wcf(test_feature,query_test_datafea,train_feature_set,train_cf)
% test_cf = test_feature(:,1);
% test_cf = cell2mat(test_cf);
% train = train_feature_set(:,1);
% train = cell2mat(train');
% 
% m2 = multi_matrix(query_test_datafea,test_cf);
% m1 = train * train_cf';
% m = EuDist2(m1', m2);
% [mrow mcol] = min(m);
% function mp = multi_matrix(query_datafea, vf)
%     mp = full(query_datafea') * vf;
% end
% 
% end

%%1019
% function label = find_top_n(vf_m,map,i_t,i_tr,K)
%     [~, idx] = sort(vf_m, 1);
%     irow = idx([1:K],:);
%     lab = [];
%     for i = 1 : K
%         la = map(irow(i,:));
%         lab = [lab, la];
%     end
%     if K >1
%         label = mode(lab');
%     else
%         label = lab;
%     end
%     label(i_t) = map(i_tr); 
% end


% function [min_c,train_label] = get_test_label(map_query_train,map_query_test,train_feature,NsubIndex)
% %function [min_c,train_label] = get_test_label(map_query_train,map_query_test,train_feature,test_feature,NsubIndex)
% map_t2t = EuDist2(map_query_train,map_query_test);
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
% num_test = cellfun(@numel,test_feature(:,2),'Uniformoutput',false);
% num_t = cell2mat(num_test);
% for i = 1 : length(num_t)
%     if i == 1
%         num_t(i,2) = 1;
%         num_t(i,3) = num_t(i,1);
%     else
%         num_t(i,2) = 1+ num_t(i-1,3);
%         num_t(i,3) = num_t(i-1,3) + num_t(i,1);
%     end
% end
% num_t = [num_t,NsubIndex(:)];
% for i = 1 : length(num_tr)
%     left = num_t(i,2);
%     right = num_t(i,3);
%     la = num_t(i,4);
%     test_label(left:right) = la;
% end
%end

%%% rewrite by hc on 2017/09/29
%%
% try
% %   %% 
%   load( fullfile( rootdir, 'subLabel_kmeans', ['data_fea_' num2str(k) '_' num2str(Nsub) '_' num2str(cl) '_maxidx' '.mat']), 'MFea', 'data_fea', 'sim', 'subsim' );
% % 
% catch
% %%computer similarity
% % %?????????kmeans???????????????????????????????????????????com_sim??????????????????????83???????????????????????????????????????????????????????cell??????n=283????????????????????m=20??????????????????????????????????????????????????83*283??????com_subsimi??????????????????????83????????????20???????????????????????????????????????com_sim???????????????????83*1??????cell??????????????cell????????????20*20??????????????
% % 
% feature_set = sub_aug_clasf_center(NsubIndex,:);
% 
% %%%%ADD by TM
% %%%%why 1:90
% %%%%ADD by TM
% 
% n = length(feature_set);
% m = k;
% sim = com_sim(n , m ,feature_set);
% subsim = com_subsimi(n , m ,feature_set);
% %save( fullfile( '/home/haichao/caffe/models/VGG/similarity', ['simi_' num2str(k) '_0' '.mat'] ), 'sim');
% %save( fullfile( '/home/haichao/caffe/models/VGG/similarity', ['sub_simi_' num2str(k) '_0' '.mat'] ), 'sub_sim');
% 
% %%sort
% %???????????????????????????????????????????d_sort????????????????????????283?????????????????????????????0???????????????????????????????????????????????????????????????????????????????????????sortl??????283????????????????????????????83*1??????sortll??????????????20????????????????????????????????????????83*20??????rsort????????????????????????????????????283*2cell????????????????????????????????????????????????????????????????????????
% 
% [a b] = get_refer_ID( feature_set, sim, k );
% [rsort] = d_sort( sim, subsim, a, b );
% % p = [5:5:20];
% % q = [20:20:80];
% % for i = 1 : length(n)
% %     [ksim ksubsim] = sort_sim(rsort, sim, subsim, p(i), q(i));
% % %save( fullfile( '/home/haichao/caffe/models/VGG/sort', ['sortl_' num2str(k) '_0' '.mat'] ),'sortl');
% % save( fullfile( 'subLabel_kmeans', ['sim_subsim_' num2str(p(i)) '_' num2str(q(i)) '_20*80_30_maxidx' '.mat']), 'ksim', 'ksubsim', '-v7.3' );
% % end
% [sim subsim] = sort_sim(rsort, sim, subsim, n, k);
% %save( fullfile( '/home/haichao/caffe/models/VGG/sort', ['sortl_' num2str(k) '_0' '.mat'] ),'sortl');
% %save( fullfile( '/home/haichao/caffe/models/VGG/sort', ['Sort_' num2str(k) '.mat']), 'rsort' );
% 
% %%get_map_idx1
% %index_21w_2_9w.mat??????????????????????????????????1w????????????????????????????9w?????????????????????????????????????????????????????????????????map_idx??????????????????
% %map_idx????????????????????????????????????????????cell??????283*2????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
% 
% [ map_org_idx aug_data_idx ] = map_idx( feature_set, rsort, index_21w_2_9w, k);
%save( fullfile( '/home/haichao/dog_query_cluster/click/reference', ['ll_img_' num2str(k) '.mat']), 'part_idx','map_org_idx','aug_data_idx');

%%to_Tindex
%?????????????????????????????????????????????????????????????????????????????????TrnasData????????????Tindex??????????????????????????????????????????????????????????????????????-283*20

% aug_data_idx = aug_data_idx';
% aug_data_idx = aug_data_idx(:);% a label(contains 20 sub_label) connect the next label(contains 20 sub_label)
% Tindex = [];
% l = 0;
% for i = 1:length(aug_data_idx)
%     r = length(aug_data_idx{i})+l;
%     Tindex(l+1:r) = i;
%     l = r;
% end
% 
% %%to_data_fea
% %??????????-Query????????????????????????????TrnasData????????????data_fea??????21w*48w??????????????
% 
% % load('/home/haichao/dog_query_cluster/click/image_click_Dog283_0_img_Fea_Clickcount.mat');
% % load('/home/haichao/dog_query_cluster/click_col.mat');
% % data_fea = img_Fea_ClickCount;
% datafea= [];
%  for j = 1:length(aug_data_idx)
%      datafe = data_fea(index_21w_2_9w(aug_data_idx{j}),:);
%      datafea = cat(1,datafea,datafe);
%  end
%  %click_col = find(sum(datafea)~=0);
%  click_col = find(sum(datafea)>cl);
%  data_fea = datafea(:,click_col);
% 
% 
% %%main
% %
% % NClass = length(NsubIndex);NSClass = k;
% 
% [NImg,NQue] = size(data_fea);
% 
% %MFea??????5660*1??????cell??????????????cell????????????1*48w??????query'_maxidx'
% MFea = TrnasData(NImg, NQue, NClass, NSClass,data_fea, Tindex);
% save( fullfile( rootdir, 'subLabel_kmeans', ['data_fea_' num2str(k) '_' num2str(Nsub) '_' num2str(cl) '_maxidx' '.mat']), 'MFea', 'data_fea', 'sim', 'subsim', '-v7.3');
% save( fullfile( rootdir, 'subLabel_kmeans', ['click_col_' num2str(k) '_' num2str(Nsub) '_' num2str(cl) '_maxidx' '.mat']),  'click_col');
% 
% end
% % NClass = Nsub;NSClass = k;
% % NClass = length(NsubIndex);NSClass = k;
% % [NImg,NQue] = size(data_fea);
% 
% % % try
% % %     load(fullfile( rootdir, 'subLabel_kmeans', ['PNC_PNSC_Pro_data_fea_' num2str(k) '_' num2str(Nsub) '_' num2str(perc) '_' num2str(cl) '.mat']));
% % % catch
% % % [NImg,NQue] = size(data_fea); 
% % % mfea = full(cell2mat(MFea));
% % % %mfea = normal_fea(mfea);
% % % mfea = reshape(mfea, [NSClass, NClass, size(mfea,2)]);
% % % mindex = {[1:NClass]};
% % % sub_sim = blkdiag(subsim{:});
% % % s = sub_sim;
% % % [row,col] = find(sub_sim == 0);
% % % ind = sub2ind(size(s),row,col);
% % % s(ind) = inf;
% % % Index = [[0:NClass-1]*NSClass+1;[1:NClass]*NSClass ];
% % % Index = Index(:);
% % % for i = 0 : length(Index)/2-1
% % %     j = 2 * i;
% % %     rIndex{1,i+1} = [Index(j+1):Index(j+2)];
% % % end
% % % for i = 1 : NQue
% % %     mQSum(:,:,i) = GetPropagate_W_v3(mfea(:,:,i), perc, sim, reindex);
% % % %     mQSum = permute(mQSum,[2 1 3]);
% % % %     mQSum = GetPropagate_W_v3(mfea(:,:,i), perc, sim, reindex);
% % % end
% % % mQSum = reshape(mQSum, [NSClass*NClass, NQue]);
% % % mQSum_i = GetPropagate_W_v3(mQSum, perc, s, rIndex);
% % % 
% % % mdata = reshape(mQSum_i',NSClass,NClass,1,NQue);
% % % for i = 1: NQue
% % %     mda = mdata(:,:,1,i);
% % %     mdd = mda;
% % %     [row,col] = find(mda);
% % %     %ind = sub2ind(size(s),row,col);'_PNC_PNSC'
% % %     [D,b] = mapminmax(mda(row,col), 128, 255);
% % %     mdd(row,col) = D; 
% % %     mdata(:,:,1,i) = dd;
% % % end
% % % save( fullfile( rootdir, 'subLabel_kmeans', [num2str(perc) '_' num2str(k) '_' num2str(Nsub) '_' num2str(cl) '_PNC_PNSC' '_data.mat']), 'mQSum', 'mQSum_i','-v7.3');
% % % save( fullfile( rootdir, 'subLabel_kmeans', ['PNC_PNSC_Pro_data_fea_' num2str(k) '_' num2str(Nsub) '_' num2str(perc) '_' num2str(cl) '.mat']), 'mdata' ,'-v7.3');
% % %   
% % % end
% 
% 
% try
%    load( fullfile( rootdir, 'subLabel_kmeans', ['Pro_data_fea_' num2str(k) '_' num2str(Nsub) '_' num2str(perc) '_' num2str(cl) '_maxidx' '.mat']));
% catch
% %sum_click ??????Mfeacell????????????????????????????????????????????????????????????NClass*NSClass=Mfea??????????????????????283*20=5660
% %????????????????????ss???????????????????????????????????????????????83*48w??????Index????????????????????????????????????????[1 20 21 40 41 60????????????282*20+1 283*20]????????????NClass??????NClass??????????????
% 
% 
% [NImg,NQue] = size(data_fea);
% [ss Index] = sum_click( MFea, NClass, NSClass);
% reindex = {[1:NClass]};
% %perc??????????????????????
% 
% %GetPropagate_W ????????????????????????????????????????????????????????????????????????????????????????????????????????????????
% QSum = GetPropagate_W_v3(ss', perc, sim, reindex);
% 
% Query_i_p = com_weight( MFea, QSum, ss, NSClass,Index);
% 
% for i = 0 : length(Index)/2-1
%     j = 2 * i;
%     rIndex{1,i+1} = [Index(j+1):Index(j+2)];
% end
% sub_sim = blkdiag(subsim{:});
% s = sub_sim;
% [row,col] = find(sub_sim == 0);
% ind = sub2ind(size(s),row,col);
% s(ind) = inf;
% % new = Query_i_p;
% % [ro,co] = find(new == NaN);
% % ind = sub2ind(size(new),ro,co);
% % new(ind) = 0;
%   QSum_i = GetPropagate_W_v3(Query_i_p, perc, s, rIndex);
% 
% data = reshape(QSum_i',NSClass,NClass,1,NQue);
% for i = 1: NQue
%     da = data(:,:,1,i);
%     dd = da;
%     [row,col] = find(da);
%     %ind = sub2ind(size(s),row,col);
%     [D,b] = mapminmax(da(row,col), 128, 255);
%     dd(row,col) = D; 
%     data(:,:,1,i) = dd;
% end
% save( fullfile( rootdir, 'subLabel_kmeans', [num2str(perc) '_' num2str(k) '_' num2str(Nsub) '_' num2str(cl) '_maxidx' '_data.mat']), 'ss', 'QSum','Query_i_p','QSum_i','-v7.3');
% save( fullfile( rootdir, 'subLabel_kmeans', ['Pro_data_fea_' num2str(k) '_' num2str(Nsub) '_' num2str(perc) '_' num2str(cl) '_maxidx' '.mat']), 'data' ,'-v7.3');
% % for i = 1: NQue
% imdata = data;
% imdata = imresize(imdata,[112 92]);
% save(fullfile( rootdir, 'subLabel_kmeans', ['imresize_data_fea_' num2str(k) '_' num2str(Nsub) '_' num2str(perc) '_' num2str(cl) '_maxidx' '.mat']), 'imdata' ,'-v7.3');
% % end
% end
% 
% 
% try
%     load( fullfile( rootdir, 'subLabel_kmeans', ['R_C_A_P_data_fea_' num2str(Nsub) '_' num2str(k) '_' num2str(perc) '_' num2str(cl) '_maxidx' '.mat']));
%     load(fullfile( rootdir, 'subLabel_kmeans', ['R_C_A_P_fea_' num2str(Nsub) '_' num2str(k) '_' num2str(perc) '_' num2str(cl) '_maxidx' '.mat']));
% catch
%     %%
% sub_sim = blkdiag(subsim{:});
% Mfea = full(cell2mat(MFea));
% [ln qn] = size(Mfea);
% for i = 1 : qn
%     pQsum = reshape(Mfea(:,i),NSClass,NClass);
%     pQSum{i} = GetPropagate_W_v3(pQsum,perc,sim,{[1:NClass]},20);
% end
% for i = 1 : qn
%     rpQsum{i} = reshape(pQSum{i},1,NSClass*NClass);
%     %rpQSum{i} = GetPropagate_W(pQsum,perc,sim,{[1:NClass]});
% end
% rpQSum = cell2mat(rpQsum(:));
% for i = 1 : NClass
%     rpIndex{i}= [(i-1)*NSClass+1:i*NSClass];
% end
% rpQSum = GetPropagate_W_v3(rpQSum, perc,sub_sim,rpIndex,50);
% save(fullfile( rootdir, 'subLabel_kmeans', ['R_C_A_P_fea_' num2str(Nsub) '_' num2str(k) '_' num2str(perc) '_' num2str(cl) '_maxidx' '.mat']), 'rpQSum' ,'-v7.3');
% [NImg,NQue] = size(data_fea);
% data = reshape(rpQSum',NSClass,NClass,1,NQue);
% mpmima = [30:20:110 128];
% for j = 1 : length(mpmima)
%     load('/home/haichao/haichao/caffe/models/VGG/subLabel_kmeans/R_C_A_P_fea_20_80_0.6_30_20_50_maxidx.mat');
%     data = reshape(rpQSum',NSClass,NClass,1,NQue);
%     %load(fullfile( rootdir, 'subLabel_kmeans', ['R_C_A_P_fea_' num2str(Nsub) '_' num2str(k) '_' num2str(perc) '_' num2str(cl) '_maxidx' '.mat']));
% for i = 1: NQue
%     da = data(:,:,1,i);
%     dd = da;
%     
%     index = find(da(:));
%     index = index(:);index = index';
%     daN = mapminmax(da(index), mpmima(j), 255);
%     da(index) = daN;
%     
% %     [row,col] = find(da);
% %     %ind = sub2ind(size(s),row,col);
% %     [D,b] = mapminmax(da(row,col), 128, 255);
% %     dd(row,col) = D; 
%      data(:,:,1,i) = da;
% end
% save( fullfile( rootdir, 'subLabel_kmeans', ['R_C_A_P_data_fea_' num2str(Nsub) '_' num2str(k) '_' num2str(perc) '_' num2str(cl) '_' num2str(mpmima(j)) '_mpmima_maxidx' '.mat']), 'data' ,'-v7.3');
% end
% end
% 
% load(fullfile( rootdir, 'subLabel_kmeans', [num2str(perc) '_' num2str(k) '_' num2str(Nsub) '_' num2str(cl) '_maxidx' '_data.mat']));
% try
%     load(fullfile( rootdir, 'subLabel_kmeans', ['Non255_data_fea_' num2str(Nsub) '_' num2str(k) '_' num2str(perc) '_' num2str(cl) '_maxidx' '.mat']));
% catch
% dat = reshape(QSum_i',NSClass,NClass,1,size(QSum_i,1));
% save( fullfile( rootdir, 'subLabel_kmeans', ['Non255_data_fea_' num2str(Nsub) '_' num2str(k) '_' num2str(perc) '_' num2str(cl) '_maxidx' '.mat']), 'dat' ,'-v7.3');
% end
% % for i = 1 : NQue
% % data(:,:,1,i) = mapminmax(data(:,:,1,i), 0, 255);
% % end
% %
% for i = 1:5
% [max(max(data(:, :, 1,i))), min(min(data(:, :, 1,i)))]
% imshow(data(:, :, 1,i)/255)
% 
% %imshow(data(:, :, 1,i))
% 
% pause
% end
% 
% 
% %save(fullfile('/home/haichao/caffe/models/VGG/dataset',['data' num2str(NClass) '*' num2str(NSClass) '*1*' num2str(NQue) '.mat']),'data','-v7.3');
% NClusters = NSClass*NClass;
% save_dir = save_hdf5(fullfile(rootdir,'subLabel_kmeans', [num2str(perc)]),data,NClusters, NQue );
% end
% %h5disp(save_dir);