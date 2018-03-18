
%% Prepare Matlab for experiment
sca;
close all;
clear all;


% Here we call some default settings for setting up Psychtoolbox
PsychDefaultSetup(2);

PsychTweak('UseGPUIndex', 0);

%% Load everything needed for the experiment
load('experimental_variables.mat')

%makes screen transparent for debugging
PsychDebugWindowConfiguration();

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
text = ['Hello space cadet!' ... 
         '\n Wellcome to the training grounds.' ... 
         '\n\n In what follows you will learn how to survive in space.' ...
         '\n\n Press any key to continue.'];

% Some block transition text     
trans_text = ['We shortly arrive to a new planetary system...'];

% Some brake text
break_text = ['Rest for some time...'];

% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.25, white);

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%EXPERIMENT STARTS FROM HERE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Variables

% Specify number of MiniBlocks

NoMiniBlocks = 100;
NoTrials = 3;

if NoTrials == 3
    planets = planetsT3;
    starts = startsT3;
elseif NoTrials == 4
    planets = planetsT4;
end
       
% Initial point and planet specific rewards
points = 995;
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

% buttons_img_location = ['button1.png'; 'button2.png'];
% 
% ButtonsTexture = NaN(2);
% for i=1:2
%   [button, ~, balpha] = imread(buttons_img_location(i, :)); 
%   % Add transparency to the background
%   button(:,:,4) = balpha;
%   % Make the planets into a texture
%   ButtonsTexture(i) = Screen('MakeTexture', window, button);
% end
% 
% % Get the size of the planet image
% [b1, b2, b3] = size(planet);
% 
% buttonRect = [0 0 b1 b2];
% 
% %generate planet locations
% buttonsPos = NaN(4, 2);
% xPos = [-800, 800];
% for i=1:2
%     buttonsPos(:,i) = CenterRectOnPointd(buttonRect, xCenter + xPos(i), yCenter+300);
% end


% Maximum priority level
topPriorityLevel = MaxPriority(window);
Priority(topPriorityLevel);

for n = 1:NoMiniBlocks
    % current experimental condition
    cond = conditions.noise{n};
    NoTrials = conditions.notrials{n};
    if n > 50
        loc = n - 50;
    else
        loc = n;
    end
    if NoTrials == 3
        start = startsT3(loc);
        planetList = planetsT3(loc,:);
    else
        start = startsT4(loc);
        planetList = planetsT4(loc,:);
    end
    
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
    
    % draw buttons
%     draw_buttons(window, ButtonsTexture, buttonsPos);
    
    % plot planets for the given mini block
    draw_planets(planetList, window, PlanetsTexture, planetsPos);
    
    % Draw the rocket at the starting position
    Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,start)');
    
    vbl = Screen('flip', window);
    
    for t = 1:NoTrials
        % Wait for a key press
        [secs, keyCode, deltaSecs] = KbPressWait;
        
        if strcmp(KbName(keyCode), 'LeftArrow')
            p = state_transition_matrix(1, start, :);
            next = find(cumsum(p)>=rand,1);
            points = points + actionCost(1);
        elseif strcmp(KbName(keyCode), 'RightArrow')
            if strcmp(cond, 'low')
                p = state_transition_matrix(3, start, :);
            else
                p = state_transition_matrix(4, start, :);
            end
            next = find(cumsum(p)>=rand,1);
            points = points + actionCost(2);
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
%             draw_buttons(window, ButtonsTexture, buttonsPos);
            draw_planets(planetList, window, PlanetsTexture, planetsPos);

            % Position of the square on this frame
            xpos = locStart(1) + time/md*(locEnd(1) - locStart(1));
            ypos = locStart(2) + time/md*(locEnd(2) - locStart(2));
        
            % Center the rectangle on the centre of the screen
            cRect = CenterRectOnPointd(rocketRect, xpos, ypos);

            % Draw the rect to the screen
            Screen('DrawTexture', window, RocketTexture, [], cRect);

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
%         draw_buttons(window, ButtonsTexture, buttonsPos);
        Screen('DrawTexture', window, RocketTexture, [], cRect);
        
        vbl = Screen('Flip', window);
    end
    WaitSecs(.5);
    if points < 0
        break
    end
end


%% End screen
end_msg = ['This is the end of experiment.' ...
           '\n\n Thank you for participating.'];

       
gameOver = ['Game over' ...
            '\n\n You have lost all your fuel.' ...
            '\n\n Thank you for participating'];       
% Draw all the text in one go
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

%%%%