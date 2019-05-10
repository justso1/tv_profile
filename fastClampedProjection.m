function x = fastClampedProjection(x0,c)
% Solves the optimization problem
%      min_x ||x0-x||_2 s.t. 0 <= x <= 1 and sum(x)=c
% using binary search on a shift of x0.

clamp = @(z) min(max(z,0),1); % useful function to clamp to [0,1]

sorted = sort(x0); % actually the slowest part of the algorithm
cdfarray = cumsum(sorted); % cumulative sum of the array --- used to sum along intervals quickly

% bounds on the shift that will be needed
l = -sorted(end);
u = 1-sorted(1);

% binary search
m = (u+l)/2;
v = shiftedSum(sorted,cdfarray,m);
while abs(v-c)/abs(v) > 1e-6
    if v < c, l = m;
    else, u = m; end
    m = (u+l)/2;
    v = shiftedSum(sorted,cdfarray,m); % third slowest line of code
end

% same shift can be used on unsorted array
x = clamp(x0+m); % second slowest line of code


function v = shiftedSum(sorted,cdfarray,shift)
% Returns sum(clamp(sorted+shift)) --- leverages sorting to make this
% log(n) time

n = length(sorted);

% Anything in sorted with val <= -shift is clamped at zero
% WANT:  Largest index in sorted so that sorted(lhs)<=-shift
l = 0; u = n+1;
while u-l > 1
    mid = floor((u+l)/2);
    if sorted(mid) > -shift, u = mid;
    else, l = mid; end
end
lhs = l;

% Anything in sorted with val >= 1-shift is clamped at one
% WANT:  Smallest index in sorted so that sorted(rhs)>=1-shift
l = 0; u = n+1;
while u-l > 1
    mid = floor((u+l)/2);
    if sorted(mid) < 1-shift, l = mid;
    else, u = mid; end
end
rhs = u; % might be beyond length of array

if rhs > 1, v = cdfarray(rhs-1) + shift*(rhs-1); % default to sum of list
else, v = n; return; end % everything clamped at 1
    
if lhs > 0, v = v - cdfarray(lhs) - shift*lhs; end % remove the 0's
if rhs <= n, v = v + (n-rhs+1); end % account for the 1's at the end