function [result,val] = isoperimetricCVX(mask, percent)

% A few special cases
if percent == 0
    result = zeros(size(mask));
    val = isotropicTotalVariation(result);
    return
end

if percent == 1
    result = mask;
    val = isotropicTotalVariation(result);
    return;
end

% Number of pixels to be used in the isoperimetric profile
pixelSum = percent * sum(mask(:));

cvx_begin
    cvx_solver mosek
    cvx_precision best
    
    variable x(size(mask,1),size(mask,2))
  
    LL = x(2:end,1:(end-1)); % lower left
    LR = x(2:end,2:end); % lower right
    UL = x(1:(end-1),1:(end-1)); % upper left
    UR = x(1:(end-1),2:end); % upper right

    % Four elements of gradient vector
    diff1 = LR - LL; 
    diff2 = UL - LL;
    diff3 = UR - UL;
    diff4 = UR - LR;

    v = norms([diff1(:) diff2(:) diff3(:) diff4(:)],2,2);
    
    minimize sum(v)
    subject to
        sum(x(:)) == pixelSum
        0 <= x <= mask
cvx_end

result = x;
val = cvx_optval;