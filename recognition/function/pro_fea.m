
function QSum = pro_fea(image_featureset,Que_Fea_Clickcount,perc,t)
if nargin <4
    t = 1;
end
%addpath( genpath( '/home/haichao/haichao/subLabel_Kmeans/function' ));
%feature_set = image_featureset(idx);
feature_set = image_featureset;
for i = 1 : length(feature_set)
    feature_set{i} = feature_set{i}./repmat(sqrt(sum(feature_set{i}.^2,2)),1,size(feature_set{i},2));
end
% % 
% try 
%     Nameres = ['sub_sim-',num2str(cluster_num),'-',num2str(length(idx)),'-',num2str(alpha),'.mat'];
%    %Nameres = ['kmean-ORG-',num2str(database_num),'-',num2str(cluster_num),'-',num2str(alpha),'-',str_num,'-',num2str(k_cluster_Q),'-',num2str(net_num),'.mat'];
%     load( Nameres );
% catch
for i = 1 : length(feature_set)
    for j = 1 : size(feature_set{i},1)
        for l = j : size(feature_set{i},1)
            sub_sim{i}(j,l) = sum(sum(EuDist2(feature_set{i}(j,:),feature_set{i}(l,:))));
            sub_sim{i}(l,j) = sub_sim{i}(j,l);     %assignment diagonal
        end
    end
end
% Nameres = ['sub_sim-',num2str(cluster_num),'-',num2str(length(idx)),'-',num2str(alpha),'.mat'];
% save(Nameres, 'sub_sim');
%end
% QSum = [];
% ss = full(Que_Fea_Clickcount');
% %qn = size(Que_Fea_Clickcount,2);
% qn = floor(size(Que_Fea_Clickcount,2)/10);

%sim = blkdiag(sub_sim{:});
% lr = 1;
cr = 0;
for i = 1 : length(feature_set) 
    lr = cr+1;
    cr = cr + size(feature_set{i},1);
    ss = full(Que_Fea_Clickcount([lr:cr],:)');
    s = sub_sim{i};
    reindex = {[1:size(feature_set{i},1)]};    
    QSum{i} = GetPropagate_W_v3(ss, perc, s, reindex,floor(length(reindex{1})*t));
end
% [row,col] = find(sim == 0);
% ind = sub2ind(size(s),row,col);
% s(ind) = inf;
% lr = 1;
% cr = length(sub_sim{1});
% for i = 1: length(feature_set)-1
%     reindex{i} = [lr:cr];
%     lr = lr + length(sub_sim{i});
%     cr = cr + length(sub_sim{i+1});
% end
% reindex{length(feature_set)} = [lr:cr];
% 
% 
% for i = 1: 10
%     QSum{i} = GetPropagate_W_v3(ss([(i-1)*qn+1:i*qn],:), perc, s, reindex);
% end
%QSum = GetPropagate_W_v3(ss, perc, s, reindex);
end
% 
