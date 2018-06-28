

function [Q_label map_label num_label] = get_Qlabel(rsort,MFea,k)
mfea = full(cell2mat(MFea));
[row,col] = find(mfea);
for i = 1 : length(mfea)
    c = find(col == i);
    num = numel(c);
    for j = 1 : num
         label_row = fix(row(c(j)) / k);
         label_col = mod(row(c(j)), k);
         if label_col == 0
             label = k * (rsort{label_row,1});
         else
             label = k * (rsort{label_row+1,1}-1) + rsort{label_row+1,2}(label_col);
         end
         Q_label{i}(j) = label;
    end
end
Q_label = cellfun(@unique,Q_label,'UniformOutput',false);
num_label = 0;
Q_lab = cellfun(@numel,Q_label);
num_Q = unique(Q_lab);
for i = 1 : length(num_Q)
    f = find(Q_lab == num_Q(i));
    Q = Q_label(f);
    [a, b, c] = unique(cell2mat(Q'),'rows');
    num_label = num_label + size(a,1);
    map_label{i,1} = f;
    map_label{i,2} = a;
    %map_label{i,3} = b;
    map_label{i,3} = c;
end
% Q_lab = cellfun(@unique,Q_label);
% Q_label = Q_lab;
end