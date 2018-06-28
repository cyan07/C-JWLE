


function [Img_vf Img_id] = find_Img(row_v,col_v,feature,feature_set,rsort)

rt = 0;
for i = 1 : length(feature)
    lf = rt+1;
    rt = rt+size(feature{i,1},1);
    feature{i,2}(2,:) = [lf:rt];
end

Img_vf = [];
Img_id = [];
for i = 1 : length(row_v)
    row = rsort{row_v(i),1};
    label = rsort{row_v(i),2}(col_v(i));
    [~,col] = find(feature_set{row,2}(1,:)==label);
    Imgcol = feature{row,2}(2,col);
    Img_fea = feature{row,1}(col,:);
    Img_vf = [Img_vf;Img_fea];
    Img_id = [Img_id,Imgcol];
end

end   