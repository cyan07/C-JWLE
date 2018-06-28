function [count_label percent_map_label] = count_label(train_map_label)
for i = 1 : length(train_map_label)
    train_map_label{i,4} = size(train_map_label{i,3},1) /size(train_map_label{i,2},1);
end
for i = 1: 22
    e = [1:size(train_map_label{i,2},1)];
    ary = arrayfun(@(x) numel(find(train_map_label{i,3} == x)),e,'UniformOutput',false);
    ary = cell2mat(ary);
    ay{i,1} = ary;
    e = [];
    ay{i,2} = numel(find(ary >=2));
    ay{i,3} = numel(find(ary >=3));
    ay{i,4} = numel(find(ary >=4));
    ay{i,5} = numel(find(ary >=5));
end
count_label = ay;
percent_map_label = train_map_label;
end
