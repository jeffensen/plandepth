close all;
clear all;

load('experimental_variables.mat')

p = ones(1,6)/6;
n_blocks = 100;
for i=1:n_blocks
 starts(i) = find(cumsum(p)>=rand,1);
end

outlike_d3 = readNPY('outlike_d3.npy');
[~, planetsT3] = max(outlike_d3, [], 3); 
outlike_d4 = readNPY('outlike_d4.npy');
[~, planetsT4] = max(outlike_d4, [], 3);

for i=1:n_blocks
    if i < 51
        conditions{i} = 'normal';
    else
        conditions{i} = 'noisy';
    end
end
    
save('experimental_variables.mat', ...
     'state_transition_matrix', ...
     'starts', ...
     'planetsT3', ...
     'planetsT4', ...
     'conditions')