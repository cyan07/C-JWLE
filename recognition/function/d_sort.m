function [rsort] = d_sort( sim, subsim, a, b )

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

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

