clear

nTests = 4; % number of x axis values to try
imfilename = 'IL_D_200_150.png'; % image file

im = imread(imfilename);
image = double(rgb2gray(im)<.9); % indicator of a shape

solver = 'cvx';
%solver = 'admm';
%solver = 'projection';

%% Carry out experiments

% Experiments we'll carry out
frac = linspace(0,1,nTests);

% Where we'll store the output
results = cell(nTests,1);
objectives = zeros(nTests,1);

for i=1:nTests
    fprintf('ITERATION %d OF %d...\n',i,nTests);
    
    if strcmp(solver,'cvx') % call Mosek through CVX
        [result,val] = isoperimetricCVX(image,frac(i));
    elseif strcmp(solver,'projection')
        [result,val] = isoperimetricProjectionAlgorithm(image,frac(i));
    elseif strcmp(solver,'admm')
        [result,val] = isoperimetricADMM(image,frac(i));
    else, error('Unrecognized solver.');
    end
    
    objectives(i) = val;
    results{i} = result;
end

%% Display

for i=1:nTests
    figure;
    imagesc(results{i});
end