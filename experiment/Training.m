%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%   TRAINING  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Prepare Matlab for experiment
sca;
close all;
clear all;

%initiate random number generator with a random seed
rng('shuffle');

% Here we call some default settings for setting up Psychtoolbox
PsychDefaultSetup(2);

% PsychTweak('UseGPUIndex', 0);

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
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%  INSTRUCTIONS START FROM HERE   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% Introductory text
text = ['Hello space cadet!' ...
        '\n\n'...
        '\n\n Wellcome to the training grounds '... 
        '\n\n '...
        '\n\n Today you will take part in a space adventure.'...
        '\n\n'...         
        '\n\n In what follows you will learn how to survive in space.' ...
        '\n\n '];
     
% Draw all the text in one go
DrawFormattedText(window, text, 'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Press any key to continue', ...
                  'center', screenYpixels*0.9);

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;


%%%%%%%%%%%% INSTRUCTIONS 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%text 
text = ['You will travel through different planetary systems.' ...
        '\n Each system consists of six plannets.' ...
         '\n\n Press any key to see an example.'];


% Draw all the text in one go
DrawFormattedText(window, text, 'center', screenYpixels * 0.25, white);

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;

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
  
% plot planets for the given mini block
planetList = [1, 3, 2, 2, 4, 5];
draw_planets(planetList, window, PlanetsTexture, planetsPos);
    
% Press Key to continue  
DrawFormattedText(window, 'Press any key to continue', ...
                  'center', screenYpixels*0.9); 

Screen('flip', window);
 
KbStrokeWait;
% 
% 
% %%%%%%%%%%%% INSTRUCTIONS 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text = ['An image of a spaceship will always mark your current position'];
        

% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Press any key to continue', ...
                  'center', screenYpixels*0.9);

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;


%%%%%%%%%%%% SHOW PLANETSYSTEM + ROCKET %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

draw_planets(planetList, window, PlanetsTexture, planetsPos);
    
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

Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,1)');

% Press Key to continue  
DrawFormattedText(window, 'Press any key to continue', ...
                  'center', screenYpixels*0.9);  

Screen('flip', window);

KbStrokeWait;

 
%%%%%%%%%%%% INSTRUCTIONS 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text = ['To survive in space you have to collect as much fuel as possible.' ... 
        '\n\n To get fuel you have to jump from planet to planet.'...
        '\n\n '...
        '\n\n At any moment you can choose between commands (keys) LEFT or RIGHT.'...
        '\n\n '...
        '\n\n Selecting LEFT uses 2 units of fuel and selecting RIGHT uses 5 units of fuel.']; 


% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Press any key to continue', ...
                  'center', screenYpixels*0.9);

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;

%%%%%%%%%%%%% INTRODUCE FUEL TANK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%% INSTRUCTIONS 5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% text
text = ['At any moment you can see your current fuel level' ...
          '\n on the top of the screen.'];


% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Press any key to continue', ...
                  'center', screenYpixels*0.8); 

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;                

%%%%%%%%%%%% Bar in the middle %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

planets = Practise;
       
% Initial point and planet specific rewards
points = 1000;
% points bar
bar = [0, 0, 100, 1000];

% draw point bar
draw_point_bar(points, window, xCenter, screenYpixels);
    
% Draw planets
draw_planets(planets, window, PlanetsTexture, planetsPos);
    
start = 1;
% Draw the rocket at the starting position 
Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,start)');
    
% Press Key to continue  
DrawFormattedText(window, 'Press any key to continue', ...
                  'center', screenYpixels*0.8);

vbl = Screen('flip', window);
    
KbStrokeWait;       

 
% %%%%%%%%%%%% PRACTISE JUMPING LEFT  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
text = ['Let us first introduce you to effects of command LEFT.' ...
        '\n Selecting LEFT allows you to travel in a clockwise direction' ...
        '\n towards the nearest neighbouring planet.' ...
        '\n\n You can practise this a few times now.']; 


% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Press any key to continue', ...
                  'center', screenYpixels*0.9); 

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;

NoMiniBlocks = 1;
NoTrials = 12;
points = 1000;

% draw fuel tank
draw_point_bar(points, window, xCenter, screenYpixels);
   
% plot planets for the given mini block
draw_planets(Practise, window, PlanetsTexture, planetsPos);

start = 1;
% Draw the rocket at the starting position 
Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,start)');
vbl = Screen('flip', window);

for t = 1:NoTrials
    % Wait for a key press
    [secs, keyCode, deltaSecs] = KbPressWait;

    if strcmp(KbName(keyCode), 'LeftArrow')
        p = state_transition_matrix(1, start, :);
        next = find(cumsum(p)>=rand,1);

        % move the rocket
        time = 0;
        locStart = imagePos(start, :);
        locEnd = imagePos(next, :);
        while time < md

            % Position of the square on this frame
            xpos = locStart(1) + time/md*(locEnd(1) - locStart(1));
            ypos = locStart(2) + time/md*(locEnd(2) - locStart(2));

            % Center the rectangle on the centre of the screen
            cRect = CenterRectOnPointd(rocketRect, xpos, ypos);

            % draw fuel tank
            draw_point_bar(points, window, xCenter, screenYpixels);
                        
            % draw planets            
            draw_planets(Practise, window, PlanetsTexture, planetsPos);
            Screen('DrawTexture', window, RocketTexture, [], cRect);

            % Flip to the screen
            vbl  = Screen('Flip', window, vbl + 0.5*ifi);

            % Increment the time
            time = time + ifi;

        end
        points = points + actionCost(1);
        % set start to a new location
        start = next;
        
        % draw fuel tank
        draw_point_bar(points, window, xCenter, screenYpixels);
        
        % draw planets
        draw_planets(Practise, window, PlanetsTexture, planetsPos);

        % draw_buttons(window, ButtonsTexture, buttonsPos);
        Screen('DrawTexture', window, RocketTexture, [], cRect);

    else strcmp(KbName(keyCode), 'RightArrow');
          DrawFormattedText(window, 'Please Press LEFT',...
                                    'center', 'center', white);

    end
    Screen('Flip', window);
end

%%%%%%% SHORT BREAK%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DrawFormattedText(window, 'Now we will introduce you to command RIGHT.',...
                                    'center', 'center', white);
                                
Screen('flip', window);

WaitSecs(1.);

% %%%%%%%%%%%% PRACTISE JUMPING Right %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text = ['If you choose the command RIGHT your spaceship will jump to a non-neighbouring planet.' ... 
        '\n\n '...
        '\n\n Next you can try out what happens on each position when you select RIGHT.'...
        '\n\n '...
        '\n\n It is important that you remember the position dependent travel pattern.']; 


% Draw all the text in one go
DrawFormattedText(window, text,'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Press any key to continue', ...
                  'center', screenYpixels*0.9);

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;

points = 1000;

for n = 1:NoMiniBlocks
    for t = 1:NoTrials
        while( true)
            if t < 7
                start = t;
            else
                start = t-6;
            end
            
            % draw fuel tank
            draw_point_bar(points, window, xCenter, screenYpixels);
            
            % plot planets for the given mini block
            planetList = Practise;
            draw_planets(planetList, window, PlanetsTexture, planetsPos);

            % Draw the rocket at the starting position 
            Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,start)');
            vbl = Screen('flip', window);

            % Wait for a key press
            [secs, keyCode, deltaSecs] = KbPressWait;

            if strcmp(KbName(keyCode), 'RightArrow')
                p = state_transition_matrix(2, start, :);
                next = find(cumsum(p)>=rand,1);

                % move the rocket
                time = 0;
                locStart = imagePos(start, :);
                locEnd = imagePos(next, :);
                poinst = points + actionCost(2);
                while time < md

                    % Position of the square on this frame
                    xpos = locStart(1) + time/md*(locEnd(1) - locStart(1));
                    ypos = locStart(2) + time/md*(locEnd(2) - locStart(2));

                    % Center the rectangle on the centre of the screen
                    cRect = CenterRectOnPointd(rocketRect, xpos, ypos);
                    
                    % draw fuel tank
                    draw_point_bar(points, window, xCenter, screenYpixels);
                    
                    % draw planets
                    draw_planets(planetList, window, PlanetsTexture, planetsPos);

                    % Draw the rect to the screen
                    Screen('DrawTexture', window, RocketTexture, [], cRect);
                    % Flip to the screen

                    vbl  = Screen('Flip', window, vbl + 0.5*ifi);


                    % Increment the time
                    time = time + ifi;
               end
               points = points + actionCost(2);
                
               WaitSecs(1);
               Screen('Flip', window);
               WaitSecs(1);
               break;
            elseif strcmp(KbName(keyCode), 'LeftArrow')
                  DrawFormattedText(window, 'Please Press RIGHT', 'center', 'center', white);
                  Screen('Flip', window); 
                  WaitSecs(1);                   
            end
        end
    end
end

% %%%%%%%%%%%% PRACTISE RIGHT ACTION IN LOW NOISE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text_2 = ['In real space taking action RIGHT is sometimes unreliable as'...
          '\n the spaceship can miss its goal and land on a neighbouring planet.' ... 
           '\n\n To give you a feeling for this we will simulate it here.'...
           '\n\n Note that the travel pattern is the same, just less reliable.']; 


% Draw all the text in one go
DrawFormattedText(window, text_2, 'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Press any key to continue', ...
                  'center', screenYpixels*0.9);

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;

points = 1000;
for n = 1:NoMiniBlocks
    for t = 1:NoTrials
        while(true)
            if t < 7
                start = t;
            else
                start = t-6;
            end
            
            % draw fuel tank
            draw_point_bar(points, window, xCenter, screenYpixels);

            % plot planets for the given mini block
            planetList = Practise;
            draw_planets(planetList, window, PlanetsTexture, planetsPos);

            % Draw the rocket at the starting position 
            Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,start)');
            vbl = Screen('flip', window);

            % Wait for a key press
            [secs, keyCode, deltaSecs] = KbPressWait;

            if strcmp(KbName(keyCode), 'RightArrow')
                p = state_transition_matrix(3, start, :);
                next = find(cumsum(p)>=rand,1);
                
                points = points + actionCost(2);
                % move the rocket
                time = 0;
                locStart = imagePos(start, :);
                locEnd = imagePos(next, :);
                while time < md

                    % Position of the square on this frame
                    xpos = locStart(1) + time/md*(locEnd(1) - locStart(1));
                    ypos = locStart(2) + time/md*(locEnd(2) - locStart(2));

                    % Center the rectangle on the centre of the screen
                    cRect = CenterRectOnPointd(rocketRect, xpos, ypos);
                    
                    % draw fuel tank
                    draw_point_bar(points, window, xCenter, screenYpixels);

                    % draw planets
                    draw_planets(planetList, window, PlanetsTexture, planetsPos);

                    % Draw the rect to the screen
                    Screen('DrawTexture', window, RocketTexture, [], cRect);
                    % Flip to the screen

                    vbl  = Screen('Flip', window, vbl + 0.5*ifi);

                    % Increment the time
                    time = time + ifi;
               end
               points = points + actionCost(2);
               
               WaitSecs(1);
               Screen('Flip', window);
               WaitSecs(1);
               break;
            elseif strcmp(KbName(keyCode), 'LeftArrow')
               DrawFormattedText(window, 'Please Press RIGHT', 'center', 'center', white);
               Screen('Flip', window); 
               WaitSecs(1);                   
            end
        end
    end
end
 
% %%%%%%%%%%%% PRACTISE RIGHT ACTION IN HIGH NOISE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text = ['Some planetary systems will have space debris in them.'...
        '\n Under this condition taking action RIGHT is highly unreliable.' ... 
        '\n\n To give you a feeling for this we will simulate it here.'...
        '\n\n Note that the travel pattern remains the same,'...
        '\n but missing the target planet becomes more likely.']; 


% Draw all the text in one go
DrawFormattedText(window, text, 'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Press any key to continue', ...
                  'center', screenYpixels*0.9);

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;

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

points = 1000;
 
NoTrials = 12;
for t = 1:NoTrials
    while(true)
        if t < 7
            start = t;
        else
            start = t-6;
        end
        % Draw debris
        Screen('DrawTexture', window, DebrisTexture, [], debrisPos)
        
        % draw fuel tank
        draw_point_bar(points, window, xCenter, screenYpixels);

        % Draw planets for the given mini block
        planetList = Practise;
        draw_planets(planetList, window, PlanetsTexture, planetsPos);

        % Draw the rocket at the starting position 
        Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,start)');
        
        vbl = Screen('flip', window);

        % Wait for a key press
        [secs, keyCode, deltaSecs] = KbPressWait;

        if strcmp(KbName(keyCode), 'RightArrow')
            p = state_transition_matrix(4, start, :);
            next = find(cumsum(p)>=rand,1);

            % Move the rocket
            time = 0;
            locStart = imagePos(start, :);
            locEnd = imagePos(next, :);
            while time < md

                % Position of the square on this frame
                xpos = locStart(1) + time/md*(locEnd(1) - locStart(1));
                ypos = locStart(2) + time/md*(locEnd(2) - locStart(2));

                % Center the rectangle on the centre of the screen
                cRect = CenterRectOnPointd(rocketRect, xpos, ypos);

                % Draw debris
                Screen('DrawTexture', window, DebrisTexture, [], debrisPos)
                
                % draw fuel tank
                draw_point_bar(points, window, xCenter, screenYpixels);
                
                draw_planets(planetList, window, PlanetsTexture, planetsPos);

                % Draw the rect to the screen
                Screen('DrawTexture', window, RocketTexture, [], cRect);
                % Flip to the screen

                vbl  = Screen('Flip', window, vbl + 0.5*ifi);


                % Increment the time
                time = time + ifi;

            end
           points = points + actionCost(2);

           WaitSecs(1);
           Screen('Flip', window);
           WaitSecs(1);
           break;
        elseif strcmp(KbName(keyCode), 'LeftArrow')
              DrawFormattedText(window, 'Please Press RIGHT', 'center', 'center', white);
              Screen('Flip', window); 
              WaitSecs(1);                   
        end
    end
end

%%%%%%%%%%%% INSTRUCTIONS 5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% text
text = ['When traveling between planets you can win or loose fuel ' ... 
         '\n\n depending on which planet you travel to.'...
         '\n\n '...
         '\n\n To survive in space it is important for you to try'...
         '\n\n to collect as much fuel as possible.'...
         '\n\n '...
         '\n\n To enable you to do this I will show you next' ...
         '\n\n which planets are good and which planets are bed fuel sources.'...   
         '\n\n '...
         '\n\n '];

% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Press any key to continue', ...
                  'center', screenYpixels*0.9);
              
% Flip to the screen
Screen('Flip', window);

KbStrokeWait;        

%%%%%%%%%%%% INSTRUCTIONS 5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Position of Planets
Rew_Pos =[[xCenter-600, yCenter]
          [xCenter-300, yCenter]
          [xCenter, yCenter]
          [xCenter+300, yCenter]
          [xCenter+600, yCenter]];

      
% Create planet Position
rewPlanetsPos = NaN(4, 5);
for i=1:5
    rewPlanetsPos(:,i) = CenterRectOnPointd(planetRect, Rew_Pos(i,1), Rew_Pos(i,2));
end

planetList = Rew_Planets;
draw_planets(planetList, window, PlanetsTexture, rewPlanetsPos);   
    
% Show Rewards
          
Screen('DrawText', window, ...
    '-20', xCenter-625, yCenter+150); 

Screen('DrawText', window, ...
    '-10', xCenter-325, yCenter+150); 

Screen('DrawText', window, ...
    '0', xCenter, yCenter+150); 

Screen('DrawText', window, ...
    '+10', xCenter+275, yCenter+150); 

Screen('DrawText', window, ...
    '+20', xCenter+575, yCenter+150); 
    
DrawFormattedText(window, 'Please remember fuel rewards for each planet: ', ...
                  'center',  screenYpixels*0.25, white); 

% Press Key to continue  
DrawFormattedText(window, 'Press any key to continue', ...
                  'center', screenYpixels*0.8); 

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;  
                
%%%%%%%%%%%% INSTRUCTIONS 5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% text
text = ['When arriving to a new planetary system you might get' ...
          '\n 3 or 4 actions to collect as much fuel as possible.'...
         'n\n Before going to the next planetary system you have to use all actions.' ...
         '\n\n The number of green squares corresponds to the number of remaining actions' ...
         '\n before you are sent to a new planetary system.'];


% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Press any key to continue', ...
                  'center', screenYpixels*0.8); 

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;                

%%%%%%%%%%%% Action points %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Variables

% Specify number of MiniBlocks

planets = Practise;
NoTrials = 3;

       
% Initial point and planet specific rewards
points = 1000;
% points bar
bar = [0, 0, 100, 1000];

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
% % Get the size of the button image
% [b1, b2, ~] = size(button);
% 
% buttonRect = [0 0 b2 b1];
% 
% %generate buttons locations
% buttonsPos = NaN(4, 2);
% xPos = [-800, 800];
% for i=1:2
%     buttonsPos(:,i) = CenterRectOnPointd(buttonRect, xCenter + xPos(i), yCenter+300);
% end

% draw point bar
draw_point_bar(points, window, xCenter, screenYpixels);
    
% draw remaining action counter
draw_remaining_actions(window, 1, NoTrials, xCenter, yCenter);
    
    % draw buttons
%     draw_buttons(window, ButtonsTexture, buttonsPos);
    
% Draw planets
draw_planets(planets, window, PlanetsTexture, planetsPos);
    
start = 1;
% Draw the rocket at the starting position 
Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,start)');
    
% Press Key to continue  
DrawFormattedText(window, 'Press any key to continue', ...
                  'center', screenYpixels*0.8);

vbl = Screen('flip', window);
    
KbStrokeWait;       



%%%%%%%%%%% INSTRUCTIONS 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% text
text = ['You can now practise the task a few times' ... 
         '\n\n '...
         '\n\n '];


% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Press any key to continue', ...
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
    
    text = ['You soon arrive in a new planetary system...'];
    
    % Draw all the text in one go
    DrawFormattedText(window, text,...
                      'center', screenYpixels * 0.25, white);

    % Flip to the screen
    Screen('Flip', window);
    WaitSecs(1.5);
    
    cond = conditionsPractise.noise{n};
    NoTrials = conditionsPractise.notrials{n};
    
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
    
    for t = 1:NoTrials
        % Wait for a key press
        [secs, keyCode, deltaSecs] = KbPressWait;
        
        if strcmp(KbName(keyCode), 'LeftArrow')
            p = state_transition_matrix(1, start, :);
            next = find(cumsum(p)>=rand,1);
            points = min(points + actionCost(1), 1000);
        elseif strcmp(KbName(keyCode), 'RightArrow')
            if strcmp(cond, 'low')
                p = state_transition_matrix(3, start, :);
            else
                p = state_transition_matrix(4, start, :);
            end
            next = find(cumsum(p)>=rand,1);
            points = min(points + actionCost(2), 1000);
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
        points = min(points + reward, 1000);
        
        if reward > 0
            s = strcat('+', int2str(reward));
        else
            s = int2str(reward);
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
end



%%%%%%%%%%%% END INSTRUCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% text
text_2 = ['Congratulations!' ... 
         '\n\n '...
         '\n\n You are now ready to start your space adventure' ...
         '\n\n '...
         '\n\n Please turn to the experimenter.'];


% Draw all the text in one go
DrawFormattedText(window, text_2,...
    'center', screenYpixels * 0.25, white);

% Flip to the screen
Screen('Flip', window);

WaitSecs(2)

sca