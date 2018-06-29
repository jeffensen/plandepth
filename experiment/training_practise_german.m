%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%   TRAINING  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Prepare Matlab for experiment
sca;
close all;
clear all;

%%%%%%%% MODIFY BEFORE EXPERIMENT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data = struct;
 
Pbn_ID = 01; % fill out
No_Training= 1; % change if more than one training
data.Age = 22; % fill out
data.Gender = 0 ; % 0 = male; 1 = female
%  
%  
% 
data.Responses.RT = NaN(20, 3);
data.Responses.Keys = NaN(20, 3);
data.States = NaN(20, 4);
data.Points = NaN(20, 3);
data.PlanetConf = NaN(20,6);
data.Conditions.notrials = NaN (20,1);
data.Conditions.noise = {};
%    
file_name = strcat('Training_part_', int2str(Pbn_ID),'_', int2str(No_Training),'_',date,'_',datestr(now,'HH-MM'), '.mat');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initiate random number generator with a random seed
rng('shuffle');

% Here we call some default settings for setting up Psychtoolbox
PsychDefaultSetup(2);

% PsychTweak('UseGPUIndex', 0);

%% Load everything needed for the experiment
load('experimental_variables.mat')

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

% Set up alpha-blending for smooth (anti-aliased) lines
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');


% Get the centre coordinate of the window
[xCenter, yCenter] = RectCenter(windowRect);


imagePos = [[xCenter-500, yCenter];
            [xCenter-200, yCenter-250];
            [xCenter+200, yCenter-250];
            [xCenter+500, yCenter];
            [xCenter+200, yCenter+250];
            [xCenter-200, yCenter+250]];


        
md = .5; %movement duration
        
%%%%%%%%%%%% SHOW PLANETSYSTEM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      
planets_img_location = ['planet_1.png'; 'planet_2.png'; 'planet_3.png'; ...
                        'planet_4.png'; 'planet_5.png'];

PlanetsTexture = NaN(1,5);
for i=1:5
  [planet, ~, palpha] = imread(planets_img_location(i, :)); 
  % Add transparency to the background
  planet(:,:,4) = palpha;
  % Make the planets into a texture
  PlanetsTexture(i) = Screen('MakeTexture', window, planet);
end

% Get the size of the planet image
[p1, p2, p3] = size(planet);

planetRect = [0 0 p2 p1];

%generate planet locations
planetsPos = NaN(4, 6);
for i=1:6
    planetsPos(:,i) = CenterRectOnPointd(planetRect, ... 
                            imagePos(i,1), imagePos(i,2));
end
  
rocket_img_location = 'rocket.png';
[rocket, ~, ralpha] = imread(rocket_img_location);

% Add transparency to the background
rocket(:,:,4) = ralpha;

% Get the size of the rocket image
[s1, s2, ~] = size(rocket);

rocketRect = [0 0 s2 s1];

%generate rocket locations
rocketPos = NaN(4, 6);
for i=1:6
    rocketPos(:,i) = CenterRectOnPointd(rocketRect, ...
                                imagePos(i,1), imagePos(i,2));
end

% Make the rocket into a texture
RocketTexture = Screen('MakeTexture', window, rocket);

% Initial point and planet specific rewards
points = 350;
% points bar
bar = [0, 0, 100, 1000];

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

%%%%%%%%%%% INSTRUCTIONS 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% text
text = ['Du kannst die Aufgabe nun ein paar Mal üben'];


% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels*0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücke eine Taste um fortzufahren.', ...
                  'center', screenYpixels*0.8);

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;      


%%%%%%%%%%%% PRACTISE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Variables

% Specify number of MiniBlocks

NoMiniBlocks = 20;
planets = planetsPractise;
starts = startsPractise;

for n = 1:NoMiniBlocks
    if n == 1
        rng(123);
    elseif n == 6
        rng(456);
    elseif n == 11
        rng(789);
    elseif n == 16
        rng(111);
    end
    if points < 0 
        break
    end
   
    text = ['In Kürze erreichst Du ein neues Planetensystem...'];
    
    % Draw all the text in one go
    DrawFormattedText(window, text,...
                      'center', screenYpixels * 0.25, white);

    % Flip to the screen
    Screen('Flip', window);
    WaitSecs(1.5);
    
    cond = conditionsPractise.noise{n};
    NoTrials = conditionsPractise.notrials(n);
    
    if strcmp(cond, 'high')
        % Draw debris
        Screen('DrawTexture', window, DebrisTexture, [], debrisPos)
    end
    
    % draw point bar
    draw_point_bar(points, window, xCenter, yCenter);
    
    % draw remaining action counter
    draw_remaining_actions(window, 1, NoTrials, xCenter, yCenter);
    
    % draw buttons
    % draw_buttons(window, ButtonsTexture, buttonsPos);
    
    % plot planets for the given mini block
    planetList = planets(n,:);
    draw_planets(planetList, window, PlanetsTexture, planetsPos);
    
    start = starts(n);
    % Draw the rocket at the starting position 
    Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,start)');
    vbl = Screen('flip', window);
    
    %save data
    data.Conditions.notrials(n)= NoTrials;
    data.Conditions.noise{n} = cond;
    data.PlanetConf(n, :)= planetList;
    
    for t = 1:NoTrials
        % Wait for a key press
        while true
            [secs, keyCode, deltaSecs] = KbPressWait;
            Key = KbName(keyCode);
            if strcmp(Key, 'RightArrow') || strcmp(Key, 's')
                break;
            end
        end
        
           
        
        % Save response and response time
        data.States(n,t) = start;
        data.Responses.RT(n, t) = secs-vbl;
        
       
        if strcmp(Key, 'RightArrow')
            p = state_transition_matrix(1, start, :);
            next = find(cumsum(p)>=0.5,1);
            ac = actionCost(1);
            points = min(points + ac, 1000);
            data.Responses.Keys(n,t)=1;
            
        elseif strcmp(Key, 's')
            if strcmp(cond, 'low')
                p = state_transition_matrix(3, start, :);
            else
                p = state_transition_matrix(4, start, :);
            end
            next = find(cumsum(p)>=rand,1);
            ac = actionCost(2);
            points = min(points + ac, 1000);
            data.Responses.Keys(n,t)=2;
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
            
            DrawFormattedText(window, int2str(ac), 'center', yCenter-100, white);

            % Flip to the screen
            vbl  = Screen('Flip', window, vbl + 0.5*ifi);

            % Increment the time
            time = time + ifi;
        end
        
        % set start to a new location
        start = next;
        reward = planetRewards(planetList(next));
        points = min(points + reward, 1000);
        
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
    end
    WaitSecs(.5);
end

save(file_name, 'data');
delete('tmpdata.mat');




%%%%%%%%%%%% END INSTRUCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% End screen
end_msg = ['Glückwunsch!' ... 
         '\n\n '...
         '\n\n Du bist nun bereit dein Weltraumabenteuer zu beginnen' ...
         '\n\n '...
         '\n\n Bitte gib dem Versuchleiter Bescheid.'];

       
gameOver = ['Game over' ...
            '\n\n Bitte gib dem Versuchsleiter Bescheid.' ...
            '\n\n '];       

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

display(points)
