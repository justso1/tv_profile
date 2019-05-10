function [val,grad] = isotropicTotalVariation(x)
% Discretization of total variation on an image suggested in
% "Geometric properties of solutions to the total variation denoising
% problem" (Chambolle et al.)

LL = x(2:end,1:(end-1)); % lower left
LR = x(2:end,2:end); % lower right
UL = x(1:(end-1),1:(end-1)); % upper left
UR = x(1:(end-1),2:end); % upper right

% Four elements of gradient vector
diff1 = LR - LL; 
diff2 = UL - LL;
diff3 = UR - UL;
diff4 = UR - LR;

v = realsqrt(diff1.*diff1 + diff2.*diff2 + diff3.*diff3 + diff4.*diff4);
val = sum(v(:));

if nargout <= 1, return; end

grad = zeros(size(x));

v(abs(v)<1e-14) = inf;
grad(2:end,1:(end-1)) = grad(2:end,1:(end-1))-(diff1+diff2)./v;
grad(2:end,2:end) = grad(2:end,2:end)+(diff1-diff4)./v;
grad(1:(end-1),1:(end-1)) = grad(1:(end-1),1:(end-1))+(diff2-diff3)./v;
grad(1:(end-1),2:end) = grad(1:(end-1),2:end)+(diff3+diff4)./v;