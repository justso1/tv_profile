function [result,val] = isoperimetricProjectionAlgorithm(mask, percent, convergence, initial, niter, nApproximateProjectionIter)

if nargin < 3, convergence = 1e-5; end
if nargin < 5, niter = 10000; end
if nargin < 6, nApproximateProjectionIter = 5000; end

% A few special cases
if percent == 0
    result = zeros(size(mask));
    val = isotropicTotalVariation(result);
    return;
end

if percent == 1
    result = mask;
    val = isotropicTotalVariation(result);
    return;
end

% Number of pixels to be used in the isoperimetric profile
pixelSum = percent * sum(mask(:));

% Random initializer seems to work well
if nargin < 4, result = mask;
else, result = initial; end

nz = find(mask);
proj = fastClampedProjection(result(nz),pixelSum);
result(nz) = proj;

[val,grad] = isotropicTotalVariation(result);

for iter=1:niter
    oldResult = result;
    oldVal = val;
    stepsize = iter^-1;
    
    % Remove irrelevant parts of gradient
    grad = grad.*mask;
    
    if iter < nApproximateProjectionIter % heuristic projection is faster and seems to help
        % Take a gradient step
        gradStep = result - stepsize*grad;
        
        % Projection onto the bound constraints
        proj1 = max(min(gradStep,1),0);

        % Projection onto the sum constraint
        curSum = sum(gradStep(:));
        proj2 = gradStep + mask/sum(mask(:))*(pixelSum-curSum);

        % Average the two projections
        result = .5*(proj1+proj2);
    else
        % Project onto both constraints
        proj = fastClampedProjection(result(nz) - stepsize*grad(nz),pixelSum);
        result(nz) = proj;
    end
    
    [val,grad] = isotropicTotalVariation(result);

    % Convergence measure
    change = sum(abs(oldResult(:)-result(:))/pixelSum);
    if iter > 1, changeObj = abs(oldVal-val)/val;
    else, changeObj = inf; end
    
    % Plot status info 
    if mod(iter,100) == 0
        fprintf('%d:\t%g\t\t%g\t\t%g\t\t%g\t\t%g\n',iter,val,change,changeObj,sum(result(:)),max(result(:)));
    end
    
    % Check for convergence
    if iter > nApproximateProjectionIter && change < convergence && changeObj < convergence, break; end
end
