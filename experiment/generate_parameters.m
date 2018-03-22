close all;
clear all;

load('experimental_variables.mat')

actionCost = [-2, -5];
Practise = [3, 3, 3, 3, 3, 3];
planetRewards = [-20, -10, 0, 10, 20];
Rew_Planets = [1,2,3,4,5];

confs = readNPY('confsExp1.npy');
startsExp = readNPY('startsExp1.npy')+1;
[~, planetsExp] = max(confs, [], 3);

confs= readNPY('confsPractise1.npy');
startsPractise = readNPY('startsPractise1.npy')+1;
[~, planetsPractise] = max(confs, [], 3);

conditionsPractise = struct;
conditionsPractise.noise = cell(4, 5);

conditionsExp = struct;
conditionsExp.noise = cell(4, 25);

conditionsPractise.notrials = ones(1,20)*2;
conditionsPractise.notrials(11:end) = 3;

conditionsExp.notrials = ones(1,100)*2;
conditionsExp.notrials(51:end) = 3;

for i = 1:4
    if i == 1 || i == 3
        conditionsPractise.noise(i,:) = {'low'};
        conditionsExp.noise(i,:) = {'low'};
    else
        conditionsPractise.noise(i,:) = {'high'};
        conditionsExp.noise(i,:) = {'high'};
    end
end

conditionsPractise.noise = reshape(conditionsPractise.noise', 1, []);
conditionsExp.noise = reshape(conditionsExp.noise', 1, []);
save('experimental_variables.mat', ...
     'actionCost', ...
     'planetRewards', ...
     'Rew_Planets', ...
     'Practise', ...
     'conditionsExp', ...
     'conditionsPractise', ...
     'startsExp', ...
     'startsPractise', ...
     'planetsExp', ...
     'planetsPractise', ...
     'state_transition_matrix');