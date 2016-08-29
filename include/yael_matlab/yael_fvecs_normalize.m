% This function normalize a set of vectors 
% Parameters:
%   v     the set of vectors to be normalized (column stored)
%   nr    the norm for which the normalization is performed (Default: Euclidean)
%
% Output:
%   vout  the normalized vector
%   vnr   the norms of the input vectors
%
% Remark: the function return Nan for vectors of null norm
%function [vout, vnr] = yael_fvecs_normalize (v, nr)
function [v, vnr] = yael_fvecs_normalize (v, nr)

fprintf ('# Warning: Obsolete. Use yael_vecs_normalize instead\n');

if nargin < 2, nr = 2; end

% norm of each column
%vnr = (sum (v.^nr)) .^ (-1 / nr);  %out of memory
vnr = zeros(1,size(v,2));
for k1 = 1:size(v,2)
    vnr(k1) = sum(v(:,k1) .^ nr) .^ (-1 / nr);
end

% sparse multiplication to apply the norm
%vout = bsxfun (@times, v, vnr);     %out of memory
%vout = zeros(size(v,1), size(v,2));
for k1 = 1:size(v,2)
   v(:,k1) = v(:,k1) .* vnr(k1); 
end


