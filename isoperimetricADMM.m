function [im,val,im2,val2,timingInfo,groundTruthDist] = isoperimetricADMM(image,frac,groundTruth)

tic

% Pad with zeros around the outside for convenience
image = padarray(padarray(image,1,0,'both')',1,0,'both')';

% Make a list of the nonzero pixels
nonzeroIndices = find(image~=0);
imageToIndex = zeros(size(image));
imageToIndex(nonzeroIndices) = (1:length(nonzeroIndices))';

% Gradient operator
upperLeft  = imageToIndex(1:(end-1),1:(end-1));
upperRight = imageToIndex(1:(end-1),2:end);
lowerLeft  = imageToIndex(2:end,1:(end-1));
lowerRight = imageToIndex(2:end,2:end);
idx = [ upperLeft(:) upperRight(:) lowerLeft(:) lowerRight(:) ];
nzIdx = (idx(:,1)~=0) | (idx(:,2)~=0) | (idx(:,3)~=0) | (idx(:,4) ~= 0);
idx = idx(nzIdx,:);
ng = size(idx,1);
UL = idx(:,1)'; UR = idx(:,2)'; LL = idx(:,3)'; LR = idx(:,4)';
R1 = 1:4:(4*ng); C1 = LR; V1 =  ones(1,ng);
R2 = 1:4:(4*ng); C2 = LL; V2 = -ones(1,ng);
R3 = 2:4:(4*ng); C3 = UL; V3 =  ones(1,ng);
R4 = 2:4:(4*ng); C4 = LL; V4 = -ones(1,ng);
R5 = 3:4:(4*ng); C5 = UR; V5 =  ones(1,ng);
R6 = 3:4:(4*ng); C6 = UL; V6 = -ones(1,ng);
R7 = 4:4:(4*ng); C7 = UR; V7 =  ones(1,ng);
R8 = 4:4:(4*ng); C8 = LR; V8 = -ones(1,ng);
R = [R1 R2 R3 R4 R5 R6 R7 R8];
C = [C1 C2 C3 C4 C5 C6 C7 C8];
V = [V1 V2 V3 V4 V5 V6 V7 V8];
idx = (C~=0); R = R(idx); C = C(idx); V = V(idx);
n = length(nonzeroIndices);
grad = sparse(R,C,V,4*ng,n);

% Deal with timing if needed
if nargout >= 5, timingInfo.setupTime = toc; end

% Call ADMM helper code
if nargout >= 6
    groundTruth = padarray(padarray(groundTruth,1,0,'both')',1,0,'both')';
    [~,z,zPrime,timingInfo.iterTime,groundTruthDist] = isoperimetricADMMReducedProblem(grad,frac*n,4,[],[],groundTruth(nonzeroIndices));
elseif nargout >= 5, [~,z,zPrime,timingInfo.iterTime] = isoperimetricADMMReducedProblem(grad,frac*n,4);
else, [~,z,zPrime] = isoperimetricADMMReducedProblem(grad,frac*n,4);
end

% Read off solution
val = sum(sqrt(sum(reshape(grad*zPrime,4,[]).^2))); 
im = zeros(size(image));
im(nonzeroIndices) = zPrime;
im = im(2:(end-1),2:(end-1));

val2 = sum(sqrt(sum(reshape(grad*z,4,[]).^2))); 
im2 = zeros(size(image));
im2(nonzeroIndices) = z; 
im2 = im2(2:(end-1),2:(end-1));
