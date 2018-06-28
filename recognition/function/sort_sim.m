%ridx = randperm(length(label_index),20);
function [ sim subsim ] = sort_sim( rsort, sim, subsim)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
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
% function [sim] = k_nn_com_sim(simi, n)
% left = ceil(n/2);%+1 
% right = floor(n/2);%-1
% if iscell(simi) == 0
%    sim = simi;
%    for j = 1 : length(simi)
%        for k = 1 : left+1
%           sim(k, 2+n:end) = inf; 
%        end
%        for k = left+2 : length(simi)-right-1
%            sim(k,1:k-left-1) = inf;
%            sim(k,k+right+1:end) = inf;
%        end
%        for k = length(simi)-right : length(simi)
%            sim(k,1:length(simi)-n-1) = inf;
%        end
%    end
% else
%     for i = 1: length(simi)
%         sim{i} =  k_nn_com_sim(simi{i}, n);
%     end
% end
% end


