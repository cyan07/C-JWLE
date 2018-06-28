function res = EuDist2(X, B)
nbase = size(B, 1); 
nframe = size(X, 1);
% find k nearest neighbors
XX = sum(X.*X, 2);%A.*B ,a1*b1,a2*a2,
BB = sum(B.*B, 2);

res  = repmat(XX, 1, nbase)-2*X*B'+repmat(BB', nframe, 1);          %repmat(A,2,3)augment A  to 2*3 
                                                                    %X*X+B*B-2*X*B= the distance of X&B
%res = exp(-res);                                 %similarity & distance is opposit


