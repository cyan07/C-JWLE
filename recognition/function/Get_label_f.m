function Ig_num = Get_label_f(aug_train,Img_num,NsubIndex)
for i = 1 : size(aug_train,1)
    Img = [];
    for j = 1 : size(aug_train,2)
        I = aug_train{i,j};
        Img = [Img,I];
    end
    Im{i,1}= Img;
    if i >1
        Im{i,2} = length(Im{i,1})+Im{i-1,2};
    else
        Im{i,2} = length(Im{i,1});
    end
end
%cellfun(@numel,Im,'UniformOutput',false);
for i = 1 : length(NsubIndex)
    if i ==1
        cl = find(Img_num<=Im{i,2});
    else
        cl = find((Im{i-1,2}<Img_num)&(Img_num<=Im{i,2}));
    end
    Ig_num(cl) = NsubIndex(i);
    cl = [];
end
end
    