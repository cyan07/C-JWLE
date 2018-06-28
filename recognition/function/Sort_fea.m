function [rsort] = Sort_fea(feature_set, sim, subsim, k)
%feature_set: a cell with {n,2};each factor in {:,1} is feature such as a
%matrix[4096,k], it in {:,2}(1,:) is the sub-label for images in each class;
%just use these mentioned;
%sim: the distance of class;
%subsim: the distance of sub-class;
%k: the number of kmeans cluster
[a b] = get_refer_ID(feature_set, sim, k);
[rsort] = d_sort(sim, subsim, a, b);
[ sim subsim ] = sort_sim( rsort, sim, subsim);
end

function [a b] = get_refer_ID( feature_set, sim, k )
%select the var which is max in every class;and select the number which is max in
%every sub-class
select_stadd = var( sim );
for i = 1 : length( sim )
    select_stadd(2,i) = size(feature_set {i,1},2);
end
%%%%ADD BY TM
%%%%Cv = sortrows(select_stadd',-2);
%%%%[row col] = find ( select_stadd == Cv(1,2) ); 
[~, col] = sortrows(select_stadd',-2);col = col(1);
%%%%ADD BY TM
for i = 1 : length( feature_set )
    for j = 1 : k
        coun_t(i,j) = numel(find( feature_set{i,2}(1,:) == j));
    end
end
[~,little_l_ID] = max(coun_t');

a = col;
b = little_l_ID;
end

function [rsort] = d_sort( sim, subsim, a, b )
%get the order for sim and subsim by the reference of a and b 
 midsim = sim;
for i = 1 : length( sim )
    sortl (i) = a;
    midsim (a,:) = inf;
    [m,id] = min(midsim(:,a));
    a = id;
end

for i = 1 : length( subsim )
    sa = b(i);
    midsim = subsim{i};
    for j = 1: length(subsim{1})
        sortll (i,j) = sa;
        midsim (sa,:) = inf;
        [m,id] = min(midsim(:,sa));
        sa = id;
    end
end
for i = 1 : length( sim )
    j = sortl (i);
    rsort {i,1} = j;
    rsort {i,2} = sortll(j,:);
end
end

function [ sim subsim ] = sort_sim( rsort, sim, subsim)
%sort the sim and subsim according the rsort
m = size(rsort,1); 
n = length(subsim{1});
for i = 1 : m
    a(i,:) = sim(rsort{i,1},:);
    b{i} = subsim{rsort{i,1}};
    for j = 1 : n
        c{i}(j,:) = b{i}(rsort{i,2}(j),:);
    end
    for j= 1 : n
        d{i}(:,j) = c{i}(:,rsort{i,2}(j));
    end
end
for i = 1 : m
    e(:,i) = a(:,rsort{i,1});
end
sim = e;
subsim = d;
% sim = k_nn_com_sim(e,k_n);
% subsim = k_nn_com_sim(d,k_ns);
end
