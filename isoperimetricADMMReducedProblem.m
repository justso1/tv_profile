function [x,z,zPrime,timingInfo,groundTruthInfo] = isoperimetricADMMReducedProblem(G,c,k,tol,maxIter,groundTruth)

% Uses ADMM to solve the optimization problem:
%      min_x  sum_i ||P_i * x||_2
%      s.t.   x - Gz = 0
%             sum(z) = c, 0 <= z <= 1
% Assumes G is a km x n matrix.  The matrices P_i extract the length-k
% subvectors of a km-dimensional vector.

if nargin < 4 || isempty(tol), tol = 1e-7; end
if nargin < 5 || isempty(maxIter), maxIter = 10000; end

if nargout >= 5
    groundTruthInfo.zDist = Inf*ones(maxIter,1);
    groundTruthInfo.zPrimeDist = Inf*ones(maxIter,1);
end

tic; if nargout >= 4, timingInfo = Inf*ones(maxIter,1); end

% Primal variables
z = zeros(size(G,2),1); Gz = zeros(size(G,1),1);
zPrime = z;             x = zeros(size(G,1),1);

% Dual variables
y = zeros(size(G,1),1);
lambda = 0;
q = zeros(size(G,2),1);

% Augmented Lagrangian coefficients --- will be updated dynamically
tau = 5;
rho = .1;
beta = .01;

% Matrix of least-squares problem that appears in the ADMM formulation (it's big!)
mtx2 = rho*(G'*G) + beta*speye(size(G,2));
[R,~,P] = chol(mtx2);

% Want to invert R'*R + tau*1*1' but the latter is dense so we use the rank-1 update formula
applyInvMtx2 = @(x) P*(R\(R'\(P'*x)));
invU = applyInvMtx2(sqrt(tau)*ones(size(mtx2,1),1));
applyOp = @(x) invHelper(x,applyInvMtx2,invU,tau); % see implementation at the bottom 

% ADMM iterations
nChanged = 0; % number of changes to augmented Lagrangian
for i=1:maxIter
    % Primal update for z --- this is the bottleneck of the code
    oldZ = z; GoldZ = Gz;
    rhs = G'*(rho*x+y) + ones(size(G,2),1)*(c*tau-lambda) + beta*zPrime - q;
    z = applyOp(rhs); % this is faster as long as we keep the Cholesky factorization in memory
    Gz = G*z; % We need this product a lot

    % Primal update for zPrime
    zPrime = min(max(z+q/beta,0),1);
    
    % Primal update for x
    h = reshape(Gz - y/rho,k,[]);
    hNorm = sqrt(sum(h.^2,1));
    coeff = max(1-1./(rho*hNorm),0);
    x = reshape(coeff.*h,[],1);
    
    % Dual update
    y = y + rho*(x - Gz);
    lambda = lambda + tau*(sum(z)-c);
    q = q + beta*(z - zPrime);
    
    % Print status
    r1 = sum(z)-c; %fprintf('\tSum primal residual: %g\n', abs(r1));
    r2 = z-zPrime; %fprintf('\tEquality primal residual: %g\n', norm(r2));
    r3 = x-Gz;     %fprintf('\tPoisson primal residual: %g\n', norm(r3));
    s1 = tau*sum(z-oldZ); %fprintf('\tSum dual residual: %g\n', abs(s1));
    s2 = beta*(oldZ-z);   %fprintf('\tEquality dual residual: %g\n', norm(s2));
    s3 = rho*(GoldZ-Gz);  %fprintf('\tPoisson dual residual: %g\n', norm(s3));
    nr1 = abs(r1); nr2 = norm(r2); nr3 = norm(r3);
    ns1 = abs(s1); ns2 = norm(s2); ns3 = norm(s3);
    
    if mod(i,100) == 0, fprintf('%d\t%g\t%g\t%g\t%g\t%g\t%g\n',i,abs(r1),norm(r2),norm(r3),abs(s1),norm(s2),norm(s3)); end
    
    % Termination criteria
    pp = 1+length(z)+length(x); nn = length(x)+length(zPrime);
    epsPri1  = tol*(sqrt(pp) + max(abs(sum(z)),c)); epsDual1 = tol*sqrt(nn);
    epsPri2  = tol*(sqrt(pp) + max(norm(z),norm(zPrime))); epsDual2 = tol*(sqrt(nn) + norm(q));
    epsPri3  = tol*(sqrt(pp) + max(norm(x),norm(Gz))); epsDual3 = tol*(sqrt(nn) + norm(y));
    if nr1<epsPri1 && ns1<epsDual1 && nr2<epsPri2 && ns2<epsDual2 && nr3<epsPri3 && ns3<epsDual3
        fprintf('Converged!\n');
        break;
    end
    
    % Update augmented Lagrangian
    if nChanged < 50 && mod(i,20) == 0 % don't refactor too many times
        changed = false;

        % Checks criteria in Boyd survey
        if nr1 > 10*ns1, tau = tau*2; changed = true;
        elseif ns1 > 10*nr1, tau = tau/2; changed = true; end
        if nr2 > 10*ns2, beta = beta*2; changed = true;
        elseif ns2 > 10*nr2, beta = beta/2; changed = true; end
        if nr3 > 10*ns3, rho = rho*2; changed = true;
        elseif ns3 > 10*nr3, rho = rho/2; changed = true; end

        if changed
            mtx2 = rho*(G'*G) + beta*speye(size(G,2));
            [R,~,P] = chol(mtx2);
            
            % Annoyingly, Matlab makes us re-define all the function pointers
            applyInvMtx2 = @(x) P*(R\(R'\(P'*x)));
            invU = applyInvMtx2(sqrt(tau)*ones(size(mtx2,1),1));
            applyOp = @(x) invHelper(x,applyInvMtx2,invU,tau); 
            
            fprintf('\tChanged parameters: %g %g %g\n', tau, beta, rho);
            nChanged = nChanged + 1;
        end
    end
    
    if nargout >= 4, timingInfo(i) = toc; end
    
    if nargout >= 5
        groundTruthInfo.zDist(i) = norm(groundTruth-z)/norm(groundTruth);
        groundTruthInfo.zPrimeDist(i) = norm(groundTruth-zPrime)/norm(groundTruth);
    end
end

if nargout >= 4, timingInfo = timingInfo(~isinf(timingInfo)); end

if nargout >= 5
    groundTruthInfo.zDist = groundTruthInfo.zDist(~isinf(groundTruthInfo.zDist));
    groundTruthInfo.zPrimeDist = groundTruthInfo.zDist(~isinf(groundTruthInfo.zPrimeDist));
end

% Sherman-Morrison-Woodbury formula helper function
function result = invHelper(x,applyInv,invU,tau)
invX = applyInv(x); % factored the formula so that there's only one expensive linear solve
result = invX - invU * (sqrt(tau)*sum(invX)) / (1 + sum(invU)*sqrt(tau)); % parenthesized carefully...