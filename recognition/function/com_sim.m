
%n: from the dimension of feature_set , means the dimension of return,283
%m: from the dimension of feature in feature_set , means the number of each
    %sample in the feature_set,20
%feature_set: a cell contains each feature of labels, cell:283*2,the first col
               %is the feature
function sim = com_sim( n, m, feature_set)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
for i = 1 : n 
    for j = i :  n
        
  %%       sim(i,j) = sum( sum( EuDist2(feature_set{i,1}',feature_set{j,1}' ))) / (m.^2);

 %%%%Add by TM
 Dist = EuDist2(feature_set{i,1}',feature_set{j,1}' );
 sim(i,j) = max(max(min(Dist, [], 1)), max(min(Dist, [], 2)));
 %%%%Add by TM
    
    
    
    
    end
end

for i = 1 : length( sim )
    for j = 1 : length( sim )
        sim(j,i) =  sim(i,j);   
    end
end

end

