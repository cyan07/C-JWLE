function nmp = multi_matrix(query_datafea, vf)
    nmp = [];
    mp = [];
    p = [];
    [img que] = size(query_datafea);
%     if img > 10000
%         for i = 1 : floor(img/10000)+1
%             l = 1 + (i-1)*10000;
%             if i < floor(img/10000)+1
%                 r = i*10000;
%             else
%                 r = img;
%             end
%             mp = multi_matrix(query_datafea([l:r],:),vf([l:r],:));
%             nmp = [nmp;mp];
%          end 
%          mp = []; p = [];
%     else
        if que > 3000
            for i = 1 : floor(que/3000)+1
                l = 1 + (i-1)*3000;
                if i < floor(que/3000)+1
                    r = i*3000;
                else
                    r = que;
                end
                p = full(query_datafea(:,[l:r]));
                mp = p' * vf;
                nmp = [nmp;mp];
            end
            mp = []; p = [];
        else
            p = full(query_datafea);
            nmp = p' * vf;
        end
        p = [];
   % end
end