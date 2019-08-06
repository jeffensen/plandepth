%% Prepare Matlab for experiment
sca;
close all;
clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng('shuffle');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%   Modify before experiment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = struct;
 
Pbn_ID = 1; % fill out
data.Age = 22; % fill out
data.Gender = 0 ; % 0 = male; 1 = female
%  
%  
% 
data.Responses.RT = NaN(100, 3);
data.Responses.Keys = NaN(100, 3);
data.States = NaN(100, 4);
data.Points = NaN(100, 3);
data.PlanetConf = NaN(100,6);
data.Conditions.notrials = NaN (100,1);
data.Conditions.noise = {};
%    
file_name = strcat('part_', int2str(Pbn_ID),'_', date,'_', datestr(now,'HH-MM'), '.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Here we call some default settings for setting up Psychtoolbox
PsychDefaultSetup(2);

PsychTweak('UseGPUIndex', 0);

%% Load everything needed for the experiment
load('experimental_variables_new.mat')

%makes screen transparent for debugging
%PsychDebugWindowConfiguration();

% Screen('Preference', 'SkipSyncTests', 1);

screen_number = max(Screen('Screens'));

% Define black and white (white will be 1 and black 0). This is because
% in general luminace values are defined between 0 and 1 with 255 steps in
% between. All values in Psychtoolbox are defined between 0 and 1
white = WhiteIndex(screen_number);

% Set a blakish background
blackish = white / 20;

% Open an on screen window using PsychImaging and color it grey.
[window, windowRect] = PsychImaging('OpenWindow', screen_number, blackish);

% Get the size of the on screen window
[screenXpixels, screenYpixels] = Screen('WindowSize', window);

% Query the frame duration
ifi = Screen('GetFlipInterval', window);

% Here we use to a waitframes number greater then 1 to flip at a rate not
% equal to the monitors refreash rate. For this example, once per second,
% to the nearest frame
flipSecs = 1;
waitframes = round(flipSecs / ifi);

% Setup the text type for the window
Screen('TextFont', window, 'Ariel');
Screen('TextSize', window, 30);

% Get the centre coordinate of the window
[xCenter, yCenter] = RectCenter(windowRect);

imagePos = [[xCenter-500, yCenter];
            [xCenter-200, yCenter-250];
            [xCenter+200, yCenter-250];
            [xCenter+500, yCenter];
            [xCenter+200, yCenter+250];
            [xCenter-200, yCenter+250]];

% Set up alpha-blending for smooth (anti-aliased) lines
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%INSTRUCTIONS START FROM HERE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Some introductory text
text = ['Hallo!' ... 
         '\n\n\n\n Dein Weltraumabenteuer kann nun beginnen.'];

% Some block transition text     
trans_text = ['In K�rze erreichst Du ein neues Planetensystem......'];

% Some brake text
break_text1 = ['Bitte nimm Dir etwas Zeit zum Ausruhen, falls Du dich m�de f�hlst.'...
                '\n\n\n\n Achtung, als n�chstes reist Du in Planetensysteme mit Asteroiden!'];

 
break_text2 = ['Bitte nimm Dir etwas Zeit zum Ausruhen, falls Du dich m�de f�hlst.'...
                '\n\n\n\n Achtung, die Anzahl deiner Reiseschritte ver�ndert sich!'];

anykey_text = ['Dr�cke eine Taste um fortzufahren.'];

% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, anykey_text, ...
                  'center', screenYpixels*0.8);

vbl = Screen('flip', window);

KbStrokeWait;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%EXPERIMENT STARTS FROM HERE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Variables

% Specify number of MiniBlocks
NoMiniBlocks = 100;
       
% Initial point and planet specific rewards
points = 1000;
planetRewards = [-20, -10, 0, 10, 20];
actionCost = [-2 -5];

% points bar
bar = [0, 0, 100, 1000];

rocket_img_location = 'rocket.png';
[rocket, ~, ralpha] = imread(rocket_img_location);
% Add transparency to the background
rocket(:,:,4) = ralpha;

planets_img_location = ['planet_1.png'; 'planet_2.png'; 'planet_3.png'; ...
                        'planet_4.png'; 'planet_5.png'];

PlanetsTexture = NaN(5);
for i=1:5
  [planet, ~, palpha] = imread(planets_img_location(i, :)); 
  % Add transparency to the background
  planet(:,:,4) = palpha;
  % Make the planets into a texture
  PlanetsTexture(i) = Screen('MakeTexture', window, planet);
end

% Get the size of the planet image
[p1, p2, p3] = size(planet);

planetRect = [0 0 p1 p2];

%generate planet locations
planetsPos = NaN(4, 6);
for i=1:6
    planetsPos(:,i) = CenterRectOnPointd(planetRect, imagePos(i,1), imagePos(i,2));
end

% Get the size of the rocket image
[s1, s2, s3] = size(rocket);

rocketRect = [0 0 s1 s2];

%generate rocket locations
rocketPos = NaN(4, 6);
for i=1:6
    rocketPos(:,i) = CenterRectOnPointd(rocketRect, imagePos(i,1), imagePos(i,2));
end

% Make the rocket into a texture
RocketTexture = Screen('MakeTexture', window, rocket);

debrisLoc = 'debris.png';

[debris, ~, dalpha] = imread(debrisLoc);

% Add transparency to the background
debris(:,:,4) = dalpha;

% Get the size of the debris image
[d1, d2, ~] = size(debris);

debrisRect = [0 0 d2 d1];

%generate debris locations
debrisPos = CenterRectOnPointd(debrisRect, xCenter, yCenter);

% Make the debris into a texture
DebrisTexture = Screen('MakeTexture', window, debris);

% Maximum priority level
topPriorityLevel = MaxPriority(window);
Priority(topPriorityLevel);

%%%% Switch the order of trials %%%%%%
conditionsExp.notrials = fliplr(conditionsExp.notrials);
planetsTmp = planetsExp(1:50,:);
planetsExp(1:50, :) = planetsExp(51:end, :);
planetsExp(51:end, :) = planetsTmp;

startsTmp = startsExp(1:50);
startsExp(1:50) = startsExp(51:end);
startsExp(51:end) = startsTmp;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Prevent spilling of keystrokes into console:
ListenChar(-1);

% Wait for the "s" and the "RightArrow" key with KbQueueWait.
deviceIndex=[];
keysOfInterest=zeros(1,256);
keysOfInterest(KbName('s'))=1;
keysOfInterest(KbName('RightArrow'))=1;
keysOfInterest(KbName('Escape'))=1;
KbQueueCreate(deviceIndex, keysOfInterest);

for n = 1:NoMiniBlocks
    
    % current experimental condition
    cond = conditionsExp.noise{n};
    NoTrials = conditionsExp.notrials(n);
    start = startsExp(n);
    planetList = planetsExp(n, :);

    %save data
    data.Conditions.notrials(n)= NoTrials;
    data.Conditions.noise{n} = cond;
    data.PlanetConf(n, :)= planetList;
    
    % mini block transition massage
    DrawFormattedText(window, trans_text,...
        'center', screenYpixels * 0.25, white);
    
    Screen('flip', window);
    % Wait for two seconds
    WaitSecs(1.5);
    
    if strcmp(cond, 'high')
        % Draw debris
        Screen('DrawTexture', window, DebrisTexture, [], debrisPos)
    end
    
    % draw point bar
    draw_point_bar(points, window, xCenter, yCenter);
    
    % draw remaining action counter
    draw_remaining_actions(window, 1, NoTrials, xCenter, yCenter);
    
    % plot planets for the given mini block
    draw_planets(planetList, window, PlanetsTexture, planetsPos);
    
    % Draw the rocket at the starting position
    Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,start)');
    
    vbl = Screen('flip', window);
    KbQueueStart(deviceIndex);
    
    for t = 1:NoTrials
       % Wait for a key press
        secs = GetSecs;
        while true
            [pressed, firstPress] = KbQueueCheck(deviceIndex);
            press_secs = firstPress(find(firstPress));
            if pressed
                Key = KbName(min(find(firstPress)));
                break
            end
        end
        KbQueueStop(deviceIndex);	% Stop delivering events to the queue
        
        % Save response and response time
        data.States(n,t) = start;
        data.Responses.RT(n, t) = press_secs-secs;
        
        if strcmp(Key, 'RightArrow')
            p = state_transition_matrix(1, start, :);
            next = find(cumsum(p)>=rand,1);
            ac = actionCost(1);
            points = points + ac;
            data.Responses.Keys(n,t)= 1;

        elseif strcmp(Key, 's')
            if strcmp(cond, 'low')
                p = state_transition_matrix(3, start, :);
            else
                p = state_transition_matrix(4, start, :);
            end
            next = find(cumsum(p)>=rand,1);
            ac = actionCost(2);
            points = points + ac;
            data.Responses.Keys(n,t)= 2;
        else
            %stop the experiment if escape was pressed
            points = -1;
        end
        if points < 0
            break
        end
        % move the rocket
        md = .5; %movement duration
        time = 0;
        locStart = imagePos(start, :);
        locEnd = imagePos(next, :);
        while time < md
            if strcmp(cond, 'high')
                % Draw debris
                Screen('DrawTexture', window, DebrisTexture, [], debrisPos)
            end
            draw_point_bar(points, window, xCenter, yCenter);
            draw_remaining_actions(window, t, NoTrials, xCenter, yCenter);
            draw_planets(planetList, window, PlanetsTexture, planetsPos);

            % Position of the square on this frame
            xpos = locStart(1) + time/md*(locEnd(1) - locStart(1));
            ypos = locStart(2) + time/md*(locEnd(2) - locStart(2));
        
            % Center the rectangle on the centre of the screen
            cRect = CenterRectOnPointd(rocketRect, xpos, ypos);

            % Draw the rect to the screen
            Screen('DrawTexture', window, RocketTexture, [], cRect);
            
            % Draw action cost
            DrawFormattedText(window, int2str(ac), 'center', yCenter - 100, white);


            % Flip to the screen
            vbl  = Screen('Flip', window, vbl + 0.5*ifi);

            % Increment the time
            time = time + ifi;
        end
        
        % set start to a new location
        start = next;
        reward = planetRewards(planetList(next));
        points = points + reward;
        
        if reward > 0
            s = strcat('+', int2str(reward));
        else
            s = int2str(reward);
        end
        
        data.Points(n, t) = points;
        
        if points < 0 
            break
        end
        
        if strcmp(cond, 'high')
            % Draw debris
            Screen('DrawTexture', window, DebrisTexture, [], debrisPos)
        end
        
        DrawFormattedText(window, s, 'center', yCenter - 100, white);
        draw_point_bar(points, window, xCenter, yCenter);
        draw_remaining_actions(window, t+1, NoTrials, xCenter, yCenter);
        draw_planets(planetList, window, PlanetsTexture, planetsPos);
        Screen('DrawTexture', window, RocketTexture, [], cRect);
        
        vbl = Screen('Flip', window);
        KbQueueStart(deviceIndex);
    end
    KbQueueStop(deviceIndex);
    data.States(n,t+1) = start;
    WaitSecs(.5);
    if points < 0
        break
    end
    save('tmpdata.mat', 'data');
    if mod(n, 25) == 0
       %make a brake
       if n < 100
        if n == 25 || n == 75
 
           DrawFormattedText(window, break_text1,... 
                 'center', screenYpixels * 0.25, white);
 
        else
 
           DrawFormattedText(window, break_text2,... 
                 'center', screenYpixels * 0.25, white);
 
        end
        % Press Key to continue  
        DrawFormattedText(window, anykey_text, ... 
                  'center', screenYpixels*0.8);
        Screen('flip', window);
        KbStrokeWait;
 
       end
    end
end

KbQueueRelease(deviceIndex);
ListenChar(0);

save(file_name, 'data');
delete('tmpdata.mat');

%% End screen
end_msg = ['Ende des Experiments.' ...
           '\n\n Danke f�r deine Teilnahme.'];

       
gameOver = ['Game over' ...
            '\n\n Deine Treibstoffreserven sind aufgebraucht.' ...
            '\n\n Danke f�r deine Teilnahme'];       

% Draw the text
if points < 0
    DrawFormattedText(window, gameOver, 'center', ...
                        screenYpixels * 0.25, white);     
else
    DrawFormattedText(window, end_msg, 'center', ...
                        screenYpixels * 0.25, white); 
end
       
Screen('Flip', window);

WaitSecs(2);

% clear the screen
sca;