%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%   TRAINING  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
Screen('TextFont', window, 'Times');
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

                     
% Position "Press Key to Continue"

 posContinue_1 = xCenter-400;
 posContinue_2 = yCenter+400;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%  INSTRUCTIONS START FROM HERE   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% Introductory text
text = ['Hello space cadet!' ...
        '\n\n'...
         '\n\n Wellcome to the training grounds '... 
         '\n\n '...
         '\n\n Today you will take part in a space adventure '...
         '\n\n'...         
         '\n\n In what follows you will learn how to survive in space.' ...
         '\n\n '];
     
% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
    Screen('DrawText', window, ...
    'Press any key to continue',  posContinue_1, posContinue_2); 

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;


%%%%%%%%%%%% INSTRUCTIONS 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%text 
text_2 = ['You will travel through different planetsystems' ... 
         '\n\n '...
         '\n\n One could for example look a bit like this: '...
          '\n\n '...
           '\n\n '];


% Draw all the text in one go
DrawFormattedText(window, text_2,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
    Screen('DrawText', window, ...
    'Press any key to continue',  posContinue_1, posContinue_2); 

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;

%%%%%%%%%%%% SHOW PLANETSYSTEM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Specify number of MiniBlocks

planets = Experiment_Practise;
       
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

planetRect = [0 0 p1 p2];

%generate planet locations
planetsPos = NaN(4, 6);
for i=1:6
    planetsPos(:,i) = CenterRectOnPointd(planetRect, imagePos(i,1), imagePos(i,2));
end
  
% plot planets for the given mini block
    planetList = planets(1,:);
    draw_planets(planetList, window, PlanetsTexture, planetsPos);
    
% Press Key to continue  
    Screen('DrawText', window, ...
    'Press any key to continue',  posContinue_1, posContinue_2); 

Screen('flip', window);
 


%
 KbStrokeWait;
% 
% 
% %%%%%%%%%%%% INSTRUCTIONS 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text_2 = ['Your spaceship shows you your current position' ... 
            '\n\n '];
        

% Draw all the text in one go
DrawFormattedText(window, text_2,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
    Screen('DrawText', window, ...
    'Press any key to continue',  posContinue_1, posContinue_2); 

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
[s1, s2, s3] = size(rocket);

rocketRect = [0 0 s1 s2];

%generate rocket locations
rocketPos = NaN(4, 6);
for i=1:6
    rocketPos(:,i) = CenterRectOnPointd(rocketRect, imagePos(i,1), imagePos(i,2));
end

% Make the rocket into a texture
RocketTexture = Screen('MakeTexture', window, rocket);

start = Practise_Starts(1);
Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,start)');

% Press Key to continue  
    Screen('DrawText', window, ...
    'Press any key to continue',  posContinue_1, posContinue_2); 

Screen('flip', window);

 KbStrokeWait;

 
%%%%%%%%%%%% INSTRUCTIONS 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text_2 = ['Your task is to collect as much fuel as possible.' ... 
           '\n\n To get fuel you have to jump from planet to planet.'...
           '\n\n '...
           '\n\n You can choose between jumping LEFT or RIGHT.'...
           '\n\n '...
           '\n\n If you jump LEFT you always travel clockwise to your neighbouring planet.'...
           '\n\n You can practise this a few times now.'...
           '\n\n '...
           '\n\n ']; 


% Draw all the text in one go
DrawFormattedText(window, text_2,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
    Screen('DrawText', window, ...
    'Press any key to continue',  posContinue_1, posContinue_2); 

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;
 

 % %%%%%%%%%%%% PRACTISE JUMPING LEFT  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NoMiniBlocks = 1;
NoTrials = 12;

planets = Practise; 

%generate planet locations
planetsPos = NaN(4, 6);
for i=1:6
    planetsPos(:,i) = CenterRectOnPointd(planetRect, imagePos(i,1), imagePos(i,2));
end

%generate rocket locations
rocketPos = NaN(4, 6);
for i=1:6
    rocketPos(:,i) = CenterRectOnPointd(rocketRect, imagePos(i,1), imagePos(i,2));
end

for n = 1:NoMiniBlocks
    
    Screen('flip', window);
    
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
  
            
             % move the rocket
             md = .5; %movement duration
             time = 0;
             locStart = imagePos(start, :);
             locEnd = imagePos(next, :);
                 while time < md
  
            % Position of the square on this frame
            xpos = locStart(1) + time/md*(locEnd(1) - locStart(1));
            ypos = locStart(2) + time/md*(locEnd(2) - locStart(2));
        
            % Center the rectangle on the centre of the screen
            cRect = CenterRectOnPointd(rocketRect, xpos, ypos);

            % Draw the rect to the screen
            
            draw_planets(planetList, window, PlanetsTexture, planetsPos);
            Screen('DrawTexture', window, RocketTexture, [], cRect);

            % Flip to the screen
            vbl  = Screen('Flip', window, vbl + 0.5*ifi);

            % Increment the time
            time = time + ifi;
            
                 end
        
                % set start to a new location
                start = next;
  
                draw_planets(planetList, window, PlanetsTexture, planetsPos);
        
        % draw_buttons(window, ButtonsTexture, buttonsPos);

            Screen('DrawTexture', window, RocketTexture, [], cRect);

             
        
        else strcmp(KbName(keyCode), 'RightArrow');
              Screen('DrawText', window, ...
                     'Please Press LEFT', xCenter-200, yCenter);

        end
        
        Screen('Flip', window);
    end
end




%%%%%%%%%%%% INSTRUCTIONS 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text_2 = ['If you jump RIGHT your space travel follows a different pattern.' ... 
           '\n\n '...
           '\n\n Next you can try out what happens on each position when you jump RIGHT.'...
           '\n\n '...
           '\n\n It is important that you try to remember this.'...
           '\n\n '...
           '\n\n ']; 


% Draw all the text in one go
DrawFormattedText(window, text_2,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
    Screen('DrawText', window, ...
    'Press any key to continue',  posContinue_1, posContinue_2); 

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;
 

 % %%%%%%%%%%%% PRACTISE JUMPING Right %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NoMiniBlocks = 12;
NoTrials = 1;

planets = Practise; 

%generate planet locations
planetsPos = NaN(4, 6);
for i=1:6
    planetsPos(:,i) = CenterRectOnPointd(planetRect, imagePos(i,1), imagePos(i,2));
end

%generate rocket locations
rocketPos = NaN(4, 6);
for i=1:6
    rocketPos(:,i) = CenterRectOnPointd(rocketRect, imagePos(i,1), imagePos(i,2));
end

for n = 1:NoMiniBlocks
    
             
    % plot planets for the given mini block
    planetList = planets(1 ,:);
    draw_planets(planetList, window, PlanetsTexture, planetsPos);
    
    start = Practise_Starts(n);
    % Draw the rocket at the starting position 
    Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,start)');
    vbl = Screen('flip', window);
    
    for t = 1:NoTrials
        % Wait for a key press
        [secs, keyCode, deltaSecs] = KbPressWait;
        
        if strcmp(KbName(keyCode), 'RightArrow')
            p = state_transition_matrix(2, start, :);
            next = find(cumsum(p)>=rand,1);
  
            
             % move the rocket
             md = .5; %movement duration
             time = 0;
             locStart = imagePos(start, :);
             locEnd = imagePos(next, :);
                 while time < md
  
            % Position of the square on this frame
            xpos = locStart(1) + time/md*(locEnd(1) - locStart(1));
            ypos = locStart(2) + time/md*(locEnd(2) - locStart(2));
        
            % Center the rectangle on the centre of the screen
            cRect = CenterRectOnPointd(rocketRect, xpos, ypos);

            draw_planets(planetList, window, PlanetsTexture, planetsPos);

            % Draw the rect to the screen
            Screen('DrawTexture', window, RocketTexture, [], cRect);
            % Flip to the screen
            
            vbl  = Screen('Flip', window, vbl + 0.5*ifi);

            
            % Increment the time
            time = time + ifi;
            
                 end
        
           % set start to a new location
                start = next;
  
                draw_planets(planetList, window, PlanetsTexture, planetsPos);
                Screen('DrawTexture', window, RocketTexture, [], cRect);

             
        
        elseif strcmp(KbName(keyCode), 'LeftArrow');
              Screen('DrawText', window, ...
                     'Please Press RIGHT', xCenter-200, yCenter);
                 
              Screen('Flip', window); 
              
              KbStrokeWait;   

          % plot planets for the given mini block
             planetList = planets(1 ,:);
             draw_planets(planetList, window, PlanetsTexture, planetsPos);

            start = Practise_Starts(n);
            % Draw the rocket at the starting position 
             Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,start)');
            vbl = Screen('flip', window);

                  p = state_transition_matrix(2, start, :);
                next = find(cumsum(p)>=rand,1);


                 % move the rocket
                 md = .5; %movement duration
                 time = 0;
                 locStart = imagePos(start, :);
                 locEnd = imagePos(next, :);
                 
               
                     while time < md

                % Position of the square on this frame
                xpos = locStart(1) + time/md*(locEnd(1) - locStart(1));
                ypos = locStart(2) + time/md*(locEnd(2) - locStart(2));

                % Center the rectangle on the centre of the screen
                cRect = CenterRectOnPointd(rocketRect, xpos, ypos);

                % Draw the rect to the screen
               
                draw_planets(planetList, window, PlanetsTexture, planetsPos);
                 Screen('DrawTexture', window, RocketTexture, [], cRect);
                % Flip to the screen

                vbl  = Screen('Flip', window, vbl + 0.5*ifi);

           

                 % Increment the time
                time = time + ifi;       
           % set start to a new location
                start = next;
  
                draw_planets(planetList, window, PlanetsTexture, planetsPos);
                Screen('DrawTexture', window, RocketTexture, [], cRect);
                
              end
                   
        end

            
            WaitSecs(1);
            Screen('Flip', window);        
            Screen('DrawText', window, ...
            '', xCenter, yCenter);
            Screen('Flip', window); 
            WaitSecs(0.5);

    end
end

 

 %%%%%%%%%%%% INSTRUCTIONS 5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% text
text_2 = ['When traveling through space you can win or loose fuel ' ... 
         '\n\n depending to which planet you travel to '...
         '\n\n '...
         '\n\n To survive in space it is important for you to try'...
         '\n\n  to collect as much fuel as possible'...
         '\n\n '...
         '\n\n To enable you to do this I will show you next' ...
         '\n\n which planets reward you and which ones you should better avoid '...   
         '\n\n '...
         '\n\n '];

% Draw all the text in one go
DrawFormattedText(window, text_2,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
    Screen('DrawText', window, ...
    'Press any key to continue',  posContinue_1, posContinue_2); 
% Flip to the screen
Screen('Flip', window);

KbStrokeWait;        

%%%%%%%%%%%% INSTRUCTIONS 5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

planets = Rew_Planets; % Planets 1-5

% Position of Planets
Rew_Pos =[[xCenter-600, yCenter]
          [xCenter-300, yCenter]
          [xCenter, yCenter]
          [xCenter+300, yCenter]
          [xCenter+600, yCenter]];

      
% Create planet Position
planetsPos = NaN(4, 5);
for i=1:5
    planetsPos(:,i) = CenterRectOnPointd(planetRect, Rew_Pos(i,1), Rew_Pos(i,2));
end

    planetList = planets;
    draw_planets(planetList, window, PlanetsTexture, planetsPos);   
    
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
    
Screen('DrawText', window, ...
    'Please try to remember: ', posContinue_1, yCenter-300); 

% Press Key to continue  
    Screen('DrawText', window, ...
    'Press any key to continue',  posContinue_1, posContinue_2); 

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;  
                
%%%%%%%%%%%% INSTRUCTIONS 5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% text
text_2 = ['You can see your current fuel level on the top of the screen ' ... 
         '\n\n '...
         '\In each planetsystem you have up to 3 jumps to collect as much fuel as possible '...
         'n\n ' ...
         '\n\n The bars in the middle show you how many jumps you have left' ...
         '\n\n '...
         '\n\n '];


% Draw all the text in one go
DrawFormattedText(window, text_2,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
    Screen('DrawText', window, ...
    'Press any key to continue',  posContinue_1, posContinue_2); 

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;                

%%%%%%%%%%%% Bar in the middle %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Variables

% Specify number of MiniBlocks

NoMiniBlocks = 1;
planets = Practise;
NoTrials = 3;

       
% Initial point and planet specific rewards
points = 500;
planetRewards = [-20, -10, 0, 10, 20];
actionCost = [-3 -6];

% points bar
bar = [0, 0, 100, 1000];

%generate planet locations
planetsPos = NaN(4, 6);
for i=1:6
    planetsPos(:,i) = CenterRectOnPointd(planetRect, imagePos(i,1), imagePos(i,2));
end


buttons_img_location = ['button1.png'; 'button2.png'];

 ButtonsTexture = NaN(2);
 for i=1:2
  [button, ~, balpha] = imread(buttons_img_location(i, :)); 
  % Add transparency to the background
  button(:,:,4) = balpha;
  % Make the planets into a texture
  ButtonsTexture(i) = Screen('MakeTexture', window, button);
end

% Get the size of the planet image
[b1, b2, b3] = size(planet);

buttonRect = [0 0 b1 b2];

%generate planet locations
buttonsPos = NaN(4, 2);
xPos = [-800, 800];
for i=1:2
    buttonsPos(:,i) = CenterRectOnPointd(buttonRect, xCenter + xPos(i), yCenter+300);
end


% Maximum priority level
topPriorityLevel = MaxPriority(window);
Priority(topPriorityLevel);

for n = 1:NoMiniBlocks
    
    % draw point bar
    draw_point_bar(points, window, xCenter, yCenter);
    
    % draw remaining action counter
    draw_remaining_actions(window, 1, NoTrials, xCenter, yCenter);
    
    % draw buttons
%     draw_buttons(window, ButtonsTexture, buttonsPos);
    
    % plot planets for the given mini block
    planetList = planets(n,:);
    draw_planets(planetList, window, PlanetsTexture, planetsPos);
    
    start = starts(n);
    % Draw the rocket at the starting position 
    Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,start)');
    
    
    vbl = Screen('flip', window);
    
% Press Key to continue  
    Screen('DrawText', window, ...
    'Press any key to continue',  posContinue_1, posContinue_2); 

    KbStrokeWait;       
end


%%%%%%%%%%% INSTRUCTIONS 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% text
text_2 = ['You can now practise the task a few times' ... 
         '\n\n '...
         '\n\n '];


% Draw all the text in one go
DrawFormattedText(window, text_2,...
    'center', screenYpixels * 0.25, white);


% Press Key to continue  
    Screen('DrawText', window, ...
    'Press any key to continue',  posContinue_1, posContinue_2); 


% Flip to the screen
Screen('Flip', window);



KbStrokeWait;      


%%%%%%%%%%%% PRACTISE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Variables

% Specify number of MiniBlocks

NoMiniBlocks = 4;
planets = Experiment_Practise;
NoTrials = 3;


       
% Initial point and planet specific rewards
points = 500;
planetRewards = [-20, -10, 0, 10, 20];
actionCost = [-3 -6];

% points bar
bar = [0, 0, 100, 1000];

%generate planet locations
planetsPos = NaN(4, 6);
for i=1:6
    planetsPos(:,i) = CenterRectOnPointd(planetRect, imagePos(i,1), imagePos(i,2));
end


buttons_img_location = ['button1.png'; 'button2.png'];

 ButtonsTexture = NaN(2);
 for i=1:2
  [button, ~, balpha] = imread(buttons_img_location(i, :)); 
  % Add transparency to the background
  button(:,:,4) = balpha;
  % Make the planets into a texture
  ButtonsTexture(i) = Screen('MakeTexture', window, button);
end

% Get the size of the planet image
[b1, b2, b3] = size(planet);

buttonRect = [0 0 b1 b2];

%generate planet locations
buttonsPos = NaN(4, 2);
xPos = [-800, 800];
for i=1:2
    buttonsPos(:,i) = CenterRectOnPointd(buttonRect, xCenter + xPos(i), yCenter+300);
end


% Maximum priority level
topPriorityLevel = MaxPriority(window);
Priority(topPriorityLevel);


for n = 1:NoMiniBlocks
    
    text_2 = ['You soon arrive in a new planetsystem'];
 
    % Draw all the text in one go
    DrawFormattedText(window, text_2,...
    'center', screenYpixels * 0.25, white);

    % Flip to the screen
    Screen('Flip', window);

    WaitSecs(1.5);
    
    % draw point bar
    draw_point_bar(points, window, xCenter, yCenter);
    
    % draw remaining action counter
    draw_remaining_actions(window, 1, NoTrials, xCenter, yCenter);
    
    % draw buttons
%     draw_buttons(window, ButtonsTexture, buttonsPos);
    
    % plot planets for the given mini block
    planetList = planets(n,:);
    draw_planets(planetList, window, PlanetsTexture, planetsPos);
    
    start = Practise_Starts(n);
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
            p = state_transition_matrix(2, start, :);
            next = find(cumsum(p)>=rand,1);
            points = points + actionCost(2);
        end
        
        % move the rocket
        md = .5; %movement duration
        time = 0;
        locStart = imagePos(start, :);
        locEnd = imagePos(next, :);
        while time < md
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