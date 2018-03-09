
%% Prepare Matlab for experiment
sca;
close all;
clear all;


% Here we call some default settings for setting up Psychtoolbox
PsychDefaultSetup(2);

%% Load everything needed for the experiment
load('matrices.mat')

%makes screen transparent for debugging
PsychDebugWindowConfiguration();

Screen('Preference', 'SkipSyncTests', 1);

%BlockMatrix = num; % includes all MiniBlock decision tasks 

% load Planet_ Position_Matrix

%[num,txt,raw] = xlsread('Planet_Position_Matrix'); 
%Planet_Position = num; 

% load pictures

background=imread('1.jpg');

imagedata_1 = imread('Planet_1.jpg'); 
imagedata_2 = imread('Planet_2.jpg'); 
imagedata_3 = imread('Planet_3.jpg'); 
imagedata_4 = imread('Planet_4.jpg'); 
imagedata_5 = imread('Planet_5.jpg'); 
imagedata_6 = imread('Planet_6.jpg'); 

line_red = imread('Line_red.jpg');
line_green = imread('Line_green.jpg');
line_yellow = imread('Line_yellow.jpg');

%% Variables

% Specify number of MiniBlocks

NoMiniBlocks = 3;
NoTrials = 3;

% Variable Position Planets
planet_pos = [[400 500 500 600];
              [650 300 750 400];
              [1150 300 1250 400];
              [1400 500 1500 600];
              [1150 700 1250 800];
              [650 700 750 800];];

% Variable Reward Planets
initi_points = 50;
planet_rewards = [-2, -1, 0, 1, 2];

% Variable Position Reward Bar
bar_pos = [500, 1000, 500, 1030];

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

% Setup the text type for the window
Screen('TextFont', window, 'Ariel');
Screen('TextSize', window, 50);

% Get the centre coordinate of the window
[xCenter, yCenter] = RectCenter(windowRect);

rocket_pos = [[xCenter-500, yCenter];
              [xCenter-250, yCenter-250];
              [xCenter+250, yCenter-250];
              [xCenter+500, yCenter];
              [xCenter+250, yCenter+250];
              [xCenter-250, yCenter+250]];

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

% Draw all the text in one go
Screen('TextSize', window, 70);
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.25, white);

% Flip to the screen
Screen('Flip', window);

KbStrokeWait;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%EXPERIMENT STARTS FROM HERE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rocket_img_location = 'rocket.png';
[rocket, ~, ralpha] = imread(rocket_img_location);

% Get the size of the image
[s1, s2, s3] = size(rocket);

% Add transparency to the background
rocket(:,:,4) = ralpha;
% Make the image into a texture
RocketTexture = Screen('MakeTexture', window, rocket);

% Our square will oscilate with a sine wave function to the left and right
% of the screen. These are the parameters for the sine wave
% See: http://en.wikipedia.org/wiki/Sine_wave
amplitude = screenXpixels * 0.25;
frequency = 0.2;
angFreq = 2 * pi * frequency;
startPhase = 0;
time = 0;

% Sync us and get a time stamp
vbl = Screen('Flip', window);
waitframes = 1;

% Maximum priority level
topPriorityLevel = MaxPriority(window);
Priority(topPriorityLevel);

baseRect = [0 0 s1 s2];

% for n = 1:No_MiniBlocks
%     init_loc = 1;
%     start_pos = rocket_pos(init_loc);
%     for t = 1:No_Trials
%         % animate the movement to a new location
%         vbl = move_rocket(rocket_pos(i-1,:),... 
%                     rocket_pos(i,:),...
%                     window,...
%                     RocketTexture,... 
%                     baseRect, ...
%                     ifi, ...
%                     vbl);
%     end
%     
% end

% [secs_1, keyCode_1, deltaSecs_1] = KbPressWait; 
% 
% Key_1= KbName(keyCode_1); 

% % set size text
% Screen('TextSize', window,40); 
% 
% %% Open empty Matrix to collect response key and response times
% 
% Response_Matrix = zeros(No_MiniBlocks,7);
% 
% % create randomised order for miniBlocks
% 
% rand_Block= randperm(No_MiniBlocks); 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %% welcome screen
% 
% Screen('DrawText', window, ...
%     'Welcome to the Experiment', 400,500);  
% 
% % Show window
% Screen('Flip',window); 
% 
% WaitSecs(2);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %% Start MiniBlock 'For Loop' 
% 
% for i = 1 : No_MiniBlocks
%     
% 
% %% Find out which Block_ID for Planet Positions
% 
% Block_ID = rand_Block(i); 
% 
% 
% %% Calculate new length rewardbar
% Line_right = Line_right+Reward 
% 
% 
% %% create first slide
% 
% Background= Screen('MakeTexture', window, background);
% Planet_1= Screen('MakeTexture', window, imagedata_1); 
% Planet_2= Screen('MakeTexture', window, imagedata_2); 
% Planet_3= Screen('MakeTexture', window, imagedata_3); 
% Planet_4= Screen('MakeTexture', window, imagedata_4); 
% Planet_5= Screen('MakeTexture', window, imagedata_5); 
% Planet_6= Screen('MakeTexture', window, imagedata_6); 
% 
% Line_red = Screen('MakeTexture', window, line_red); 
% Line_yel = Screen('MakeTexture', window, line_yellow); 
% Line_gre = Screen('MakeTexture', window, line_green);  
% 

% %% Draw Reward Bar
% 
% line_length = Line_right - Line_left
% 
% if line_length < 500
%     Screen('DrawTexture',window,Line_red, [], [Line_left Line_top Line_right Line_bottom]);
% elseif line_length < 750
%         Screen('DrawTexture',window,Line_yel, [], [Line_left Line_top Line_right Line_bottom]);
% else Screen('DrawTexture',window,Line_gre, [], [Line_left Line_top Line_right Line_bottom]);
% end 
% 
% %% Check Position for each Planet and Draw Planet
% 
% % Planet 1
% x=Planet_Position(Block_ID,2); % checks matrix 
% 
% if x == 1
%     Pos_Pla_1 = Position_1;
% elseif x== 2
%     Pos_Pla_1 = Position_2;
% elseif x == 3
%     Pos_Pla_1 = Position_3;
% elseif x == 4
%     Pos_Pla_1= Position_4;
% elseif x == 5
%     Pos_Pla_1= Position_5;
%     
% else Pos_Pla_1= Position_6;
% end
%    
% Screen('DrawTexture', window, Planet_1,[],[Pos_Pla_1]); % Planet 1    
% 
% 
% % Planet 2
% x=Planet_Position(Block_ID,3)
% 
% if x == 1
%     Pos_Pla_2= Position_1
% elseif x== 2
%     Pos_Pla_2 = Position_2
% elseif x == 3
%     Pos_Pla_2 = Position_3
% elseif x == 4
%     Pos_Pla_2 = Position_4
% elseif x == 5
%     Pos_Pla_2 = Position_5
% else Pos_Pla_2 = Position_6
% end
% 
% Screen('DrawTexture', window, Planet_2,[],[Pos_Pla_2]); % Planet 2 
% 
% 
% % Planet 3
% x=Planet_Position(Block_ID,4)
% 
% if x == 1
%     Pos_Pla_3 = Position_1
% elseif x== 2
%     Pos_Pla_3 = Position_2
% elseif x == 3
%     Pos_Pla_3 = Position_3
% elseif x == 4
%     Pos_Pla_3= Position_4
% elseif x == 5
%     Pos_Pla_3= Position_5
% else Pos_Pla_3= Position_6
% end
%     
% Screen('DrawTexture', window, Planet_3,[],[Pos_Pla_3]); % Planet 3 
% 
% 
% % Planet 4
% x=Planet_Position(Block_ID,5)
% 
% if x == 1
%    Pos_Pla_4 = Position_1;
% elseif x== 2
%     Pos_Pla_4 = Position_2
% elseif x == 3
%     Pos_Pla_4 = Position_3
% elseif x == 4
%     Pos_Pla_4 = Position_4
% elseif x == 5
%     Pos_Pla_4= Position_5
% else Pos_Pla_4 = Position_6
% end
%     
% Screen('DrawTexture', window, Planet_4,[],[Pos_Pla_4]); % Planet 4 
% 
% 
% % Planet 5
% x=Planet_Position(Block_ID,6)
% 
% if x == 1
%     Pos_Pla_5 = Position_1
% elseif x== 2
%     Pos_Pla_5 = Position_2
% elseif x == 3
%     Pos_Pla_5 = Position_3
% elseif x == 4
%     Pos_Pla_5= Position_4
% elseif x == 5
%     Pos_Pla_5= Position_5
% else Pos_Pla_5= Position_6
% end
% 
% Screen('DrawTexture', window, Planet_5,[],[Pos_Pla_5]); % Planet 5
% 
% 
% % Planet 6
% x=Planet_Position(Block_ID,7)
% 
% if x == 1
%     Pos_Pla_6 = Position_1
% elseif x== 2
%     Pos_Pla_6 = Position_2
% elseif x == 3
%    Pos_Pla_6 = Position_3
% elseif x == 4
%     Pos_Pla_6 = Position_4
% elseif x == 5
%     Pos_Pla_6 = Position_5
% else Pos_Pla_6 = Position_6
% end
%    
% Screen('DrawTexture', window, Planet_6,[],[Pos_Pla_6]); % Planet 6
% 
% 
% %% Draw Circle to show current state
% 
% Screen('FrameOval', window, [255 0 0], [Circle_Planet_1],10); 
% 
% %% Show window
% Screen('Flip',window); 
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% STAGE 2
% %% Collect keyboard respnse -at the moment 1 for left or anything else for
% % right
% 
% [secs_1, keyCode_1, deltaSecs_1] = KbPressWait; 
% 
% Key_1= KbName(keyCode_1); 
% 
% %% use response to determine next stage
%     
% % Draw Background
% 
% Screen('DrawTexture',window,Background);  % Background
% Screen('DrawTexture', window, Planet_1,[],[Pos_Pla_1]); % Planet 1  
% Screen('DrawTexture', window, Planet_2,[],[Pos_Pla_2]); % Planet 2  
% Screen('DrawTexture', window, Planet_3,[],[Pos_Pla_3]); % Planet 3
% Screen('DrawTexture', window, Planet_4,[],[Pos_Pla_4]); % Planet 4
% Screen('DrawTexture', window, Planet_5,[],[Pos_Pla_5]); % Planet 5  
% Screen('DrawTexture', window, Planet_6,[],[Pos_Pla_6]); % Planet 6  
% 
% Screen('DrawText', window, ...
%     '2', 900,500); 
% 
% 
% %% Draw Line
% 
% line_length = Line_right - Line_left
% 
% if line_length < 500
% 
%     Screen('DrawTexture',window,Line_red, [], [Line_left Line_top Line_right Line_bottom]);
%     
% elseif line_length < 750
%         Screen('DrawTexture',window,Line_yel, [], [Line_left Line_top Line_right Line_bottom]);
%         
% else 
%         Screen('DrawTexture',window,Line_gre, [], [Line_left Line_top Line_right Line_bottom]);
%     
% end 
% 
% %% Use response to check for next stage in MiniBlockMatrix 
% 
% if Key_1 == '1!' % Decision left 
%     x = BlockMatrix(2);
% else
%     x = BlockMatrix(3); % Decision right
%     
% end
% 
% if x ==1
%     y= Circle_Planet_1;
%     
%     elseif x==2
%         y = Circle_Planet_2;
%         
%         elseif x ==3 
%             y = Circle_Planet_3;
%             elseif x == 4
%                 y= Circle_Planet_4;
%                 elseif x==5
%                     y = Circle_Planet_5;
%                     
%                 else y = Circle_Planet_6;
% end
% 
% % Draw Circle around Stage 2
% Screen('FrameOval', window, [255 0 0], [y],10); 
% 
% 
% % Show next Stage 
% Screen('Flip',window); 
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%2
% %% STAGE 3
% 
% %% Collect keyboard respnse -at the moment 1 for left or anything else for
% % right
% 
% [secs_2, keyCode_2, deltaSecs_2] = KbPressWait; 
% 
% Key_2= KbName(keyCode_2); 
% 
% %% use response to determine next stage
%     
% % Draw Background and Planets
% 
% Screen('DrawTexture',window,Background);  % Background
% Screen('DrawTexture', window, Planet_1,[],[Pos_Pla_1]); % Planet 1  
% Screen('DrawTexture', window, Planet_2,[],[Pos_Pla_2]); % Planet 2  
% Screen('DrawTexture', window, Planet_3,[],[Pos_Pla_3]); % Planet 3
% Screen('DrawTexture', window, Planet_4,[],[Pos_Pla_4]); % Planet 4
% Screen('DrawTexture', window, Planet_5,[],[Pos_Pla_5]); % Planet 5  
% Screen('DrawTexture', window, Planet_6,[],[Pos_Pla_6]); % Planet 6  
% 
% Screen('DrawText', window, ...
%     '1', 900,500); 
% 
% 
% %% Draw Line
% 
% line_length = Line_right - Line_left
% 
% if line_length < 500
% 
%     Screen('DrawTexture',window,Line_red, [], [Line_left Line_top Line_right Line_bottom]);
%     
% elseif line_length < 750
%         Screen('DrawTexture',window,Line_yel, [], [Line_left Line_top Line_right Line_bottom]);
%         
% else 
%         Screen('DrawTexture',window,Line_gre, [], [Line_left Line_top Line_right Line_bottom]);
%     
% end 
% 
% %% Use response to check for next stage in MiniBlockMatrix 
% 
% if (Key_1 == '1!' ) & (Key_2 == '1!') % In both Stages decision left 
%     x = BlockMatrix(4);
%     
% elseif Key_1=='1!'  % In Stage 1 decision left, Stage 2 decision right
%      x = BlockMatrix(6);
% else
%     x = BlockMatrix(5); % In both stages decision right
% end
% 
% 
% % check Position Circle
% if x ==1
%     y= Circle_Planet_1
%     
%     elseif x==2
%         y = Circle_Planet_2
%         
%         elseif x ==3 
%             y = Circle_Planet_3
%             elseif x == 4
%                 y= Circle_Planet_4
%                 elseif x==5
%                     y = Circle_Planet_5
%                     
%                 else y = Circle_Planet_6
% end
% 
% % Draw Circle around Stage 3
% Screen('FrameOval', window, [255 0 0], [y],10); 
% 
% % Show next Stage 
% Screen('Flip',window); 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %% STAGE 4
% 
% %% Collect keyboard respnse -at the moment 1 for left or anything else for
% % right
% 
% [secs_3, keyCode_3, deltaSecs_3] = KbPressWait; 
% 
% Key_3= KbName(keyCode_3); 
% 
% %% use response to determine next stage
%     
% % Draw Background and Planets
% 
% Screen('DrawTexture',window,Background);  % Background
% Screen('DrawTexture', window, Planet_1,[],[Pos_Pla_1]); % Planet 1  
% Screen('DrawTexture', window, Planet_2,[],[Pos_Pla_2]); % Planet 2  
% Screen('DrawTexture', window, Planet_3,[],[Pos_Pla_3]); % Planet 3
% Screen('DrawTexture', window, Planet_4,[],[Pos_Pla_4]); % Planet 4
% Screen('DrawTexture', window, Planet_5,[],[Pos_Pla_5]); % Planet 5  
% Screen('DrawTexture', window, Planet_6,[],[Pos_Pla_6]); % Planet 6 
% 
% %% Draw Line
% 
% line_length = Line_right - Line_left
% if line_length < 500
%     Screen('DrawTexture',window,Line_red, [], [Line_left Line_top Line_right Line_bottom]);
% elseif line_length < 750
%         Screen('DrawTexture',window,Line_yel, [], [Line_left Line_top Line_right Line_bottom]);
% else    Screen('DrawTexture',window,Line_gre, [], [Line_left Line_top Line_right Line_bottom]);
% end 
% 
% 
% %% Use response to check for next stage in MiniBlockMatrix 
% 
% if (Key_1 == '1!' ) & (Key_2 == '1!') & (Key_3 =='1!') % Decision left 
%     x = BlockMatrix(8);
%    
% elseif (Key_1=='1!') & (Key_2 == '1!')
%     x = BlockMatrix(14)
%    
% elseif (Key_1=='1!') & (Key_3 == '1!')
%     
%      x = BlockMatrix(12);
%      
% elseif (Key_2=='1!') & (Key_3 == '1!')
%     
%      x = BlockMatrix(11);
%      
% elseif (Key_1=='1!')     
%      x = BlockMatrix(10);
%      
% elseif (Key_2=='1!')     
%      x = BlockMatrix(13);
%      
% elseif (Key_3=='1!')     
%      x = BlockMatrix(15);
% else
%     x = BlockMatrix(9); % Decision right
% end
% 
% 
% % check Position Circle
% if x ==1
%     y= Circle_Planet_1
%     Reward = Rew_Planet_1
%     
%     elseif x==2
%         y = Circle_Planet_2
%     Reward = Rew_Planet_2
%         elseif x ==3 
%             y = Circle_Planet_3
%             Reward = Rew_Planet_3
%             elseif x == 4
%                 y= Circle_Planet_4
%                 Reward = Rew_Planet_4
%                 elseif x==5
%                     y = Circle_Planet_5
%                     Reward = Rew_Planet_5
%                     
%                 else y = Circle_Planet_6
%                     Reward = Rew_Planet_6
% end
% 
% % Draw Circle around Stage 3
% Screen('FrameOval', window, [255 0 0], [y],10); 
% 
% % Show next Stage 
% Screen('Flip',window); 
% 
% WaitSecs(2);
% Screen('DrawText', window, ...
%     '+' ,800,500,[0 0 0]); 
% 
% Screen('Flip', window);
% 
% WaitSecs(1);
% 
% %save response in Matrix
% 
% u = Key_1(1);
% v = secs_1;
% 
% w = Key_2(1);
% x = secs_2;
% 
% y = Key_3(1);
% z = secs_3;
% 
% 
% 
% 
% Response_Matrix(i,:) = [Block_ID, u,v,w,x,y,z];
% 
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %% End screen
% 
% Screen('DrawText', window, ...
%     'End' ,800,500,[0 0 0]); 
% 
% Screen('Flip', window);
% 
% WaitSecs(2);

% clear the screen
sca;

%%%%