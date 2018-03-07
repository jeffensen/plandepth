
%% Prepare Matlab for experiment
clear all; 

% load MiniBlockMatrix

[num,txt,raw] = xlsread('Block_Matrix');
  
BlockMatrix = num; % includes all MiniBlock decision tasks 

% load Planet Position Matrix

[num,txt,raw] = xlsread('Planet_Position_Matrix'); 

Planet_Position = num; 

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

% Variables

% Specify number of MiniBlocks

No_MiniBlocks = 3

% Variable Circles 

Circle_Planet_1 = [350 450 550 650];
Circle_Planet_2 = [600 250 800 450];
Circle_Planet_3 = [1100 250 1300 450];
Circle_Planet_4 = [1350 450 1550 650];
Circle_Planet_5 = [1100 650 1300 850];
Circle_Planet_6 = [600 650 800 850];


% Variable Position Planets

Position_1 = [400 500 500 600];
Position_2 = [650 300 750 400];
Position_3 = [1150 300 1250 400];
Position_4 = [1400 500 1500 600];
Position_5 = [1150 700 1250 800];
Position_6 =[650 700 750 800];

% Variable Reward Planets

Reward = 300;
Rew_Planet_1 = 100;
Rew_Planet_2 = 50; 
Rew_Planet_3 = -50; 
Rew_Planet_4 = 250; 
Rew_Planet_5 = -100;
Rew_Planet_6 = -250; 


% Variable PositionLine
Line_left = 500; 
Line_top = 1000;
Line_right = 500;
Line_bottom = 1030;


% Open Screeen
window = Screen('OpenWindow',0); 
Screen('TextSize', window,40); % set size text

% Open empty Matrix to collect response and response times

Response_Matrix = zeros(No_MiniBlocks,7);

% create randomised order for miniBlocks

rand_Block= randperm(No_MiniBlocks);

%% Start MiniBlock 'For Loop' 

for i = 1 : No_MiniBlocks
    

%% Chose MiniBlock by random

Block_ID = rand_Block(i); 

Line_right = Line_right+Reward


%% create first slide


Background= Screen('MakeTexture', window, background);
Planet_1= Screen('MakeTexture', window, imagedata_1); 
Planet_2= Screen('MakeTexture', window, imagedata_2); 
Planet_3= Screen('MakeTexture', window, imagedata_3); 
Planet_4= Screen('MakeTexture', window, imagedata_4); 
Planet_5= Screen('MakeTexture', window, imagedata_5); 
Planet_6= Screen('MakeTexture', window, imagedata_6); 

Line_red = Screen('MakeTexture', window, line_red); 
Line_yel = Screen('MakeTexture', window, line_yellow); 
Line_gre = Screen('MakeTexture', window, line_green);  

%% Draw Background

Screen('DrawTexture',window,Background);  % Background
Screen('DrawText', window, ...
    '3', 900,500); 

%% Draw Line

line_length = Line_right - Line_left

if line_length < 500

    Screen('DrawTexture',window,Line_red, [], [Line_left Line_top Line_right Line_bottom]);
    
elseif line_length < 750
        Screen('DrawTexture',window,Line_yel, [], [Line_left Line_top Line_right Line_bottom]);
        
else 
        Screen('DrawTexture',window,Line_gre, [], [Line_left Line_top Line_right Line_bottom]);
    
end 

%% Check Position for each Planet and Draw Planet

% Planet 1

x=Planet_Position(Block_ID,2);

if x == 1
    Pos_Pla_1 = Position_1
elseif x== 2
    Pos_Pla_1 = Position_2;
elseif x == 3
    Pos_Pla_1 = Position_3
elseif x == 4
    Pos_Pla_1= Position_4
elseif x == 5
    Pos_Pla_1= Position_5
    
else Pos_Pla_1= Position_6
end
   
Screen('DrawTexture', window, Planet_1,[],[Pos_Pla_1]); % Planet 1    


% Planet 2

x=Planet_Position(Block_ID,3)

if x == 1
    Pos_Pla_2= Position_1
elseif x== 2
    Pos_Pla_2 = Position_2
elseif x == 3
    Pos_Pla_2 = Position_3
elseif x == 4
    Pos_Pla_2 = Position_4
elseif x == 5
    Pos_Pla_2 = Position_5
else Pos_Pla_2 = Position_6
end

Screen('DrawTexture', window, Planet_2,[],[Pos_Pla_2]); % Planet 2 


% Planet 3

x=Planet_Position(Block_ID,4)

if x == 1
    Pos_Pla_3 = Position_1
elseif x== 2
    Pos_Pla_3 = Position_2
elseif x == 3
    Pos_Pla_3 = Position_3
elseif x == 4
    Pos_Pla_3= Position_4
elseif x == 5
    Pos_Pla_3= Position_5
else Pos_Pla_3= Position_6
end
    
Screen('DrawTexture', window, Planet_3,[],[Pos_Pla_3]); % Planet 3 


% Planet 4

x=Planet_Position(Block_ID,5)

if x == 1
   Pos_Pla_4 = Position_1;
elseif x== 2
    Pos_Pla_4 = Position_2
elseif x == 3
    Pos_Pla_4 = Position_3
elseif x == 4
    Pos_Pla_4 = Position_4
elseif x == 5
    Pos_Pla_4= Position_5
else Pos_Pla_4 = Position_6
end
    
Screen('DrawTexture', window, Planet_4,[],[Pos_Pla_4]); % Planet 4 


% Planet 5

x=Planet_Position(Block_ID,6)

if x == 1
    Pos_Pla_5 = Position_1
elseif x== 2
    Pos_Pla_5 = Position_2
elseif x == 3
    Pos_Pla_5 = Position_3
elseif x == 4
    Pos_Pla_5= Position_4
elseif x == 5
    Pos_Pla_5= Position_5
else Pos_Pla_5= Position_6
end

Screen('DrawTexture', window, Planet_5,[],[Pos_Pla_5]); % Planet 5


% Planet 6

x=Planet_Position(Block_ID,7)

if x == 1
    Pos_Pla_6 = Position_1
elseif x== 2
    Pos_Pla_6 = Position_2
elseif x == 3
   Pos_Pla_6 = Position_3
elseif x == 4
    Pos_Pla_6 = Position_4
elseif x == 5
    Pos_Pla_6 = Position_5
else Pos_Pla_6 = Position_6
end

    
Screen('DrawTexture', window, Planet_6,[],[Pos_Pla_6]); % Planet 6


% Draw Circle to show current state

Screen('FrameOval', window, [1 0 1], [Circle_Planet_1],10); 

% Show window
Screen('Flip',window); 




%% STAGE 2
%% Collect keyboard respnse -at the moment 1 for left or anything else for
% right

[secs_1, keyCode_1, deltaSecs_1] = KbPressWait; 

Key_1= KbName(keyCode_1); 

%% use response to determine next stage
    
% Draw Background

Screen('DrawTexture',window,Background);  % Background
Screen('DrawTexture', window, Planet_1,[],[Pos_Pla_1]); % Planet 1  
Screen('DrawTexture', window, Planet_2,[],[Pos_Pla_2]); % Planet 2  
Screen('DrawTexture', window, Planet_3,[],[Pos_Pla_3]); % Planet 3
Screen('DrawTexture', window, Planet_4,[],[Pos_Pla_4]); % Planet 4
Screen('DrawTexture', window, Planet_5,[],[Pos_Pla_5]); % Planet 5  
Screen('DrawTexture', window, Planet_6,[],[Pos_Pla_6]); % Planet 6  

Screen('DrawText', window, ...
    '2', 900,500); 


%% Draw Line

line_length = Line_right - Line_left

if line_length < 500

    Screen('DrawTexture',window,Line_red, [], [Line_left Line_top Line_right Line_bottom]);
    
elseif line_length < 750
        Screen('DrawTexture',window,Line_yel, [], [Line_left Line_top Line_right Line_bottom]);
        
else 
        Screen('DrawTexture',window,Line_gre, [], [Line_left Line_top Line_right Line_bottom]);
    
end 

%% Use response to check for next stage in MiniBlockMatrix 

if Key_1 == '1!' % Decision left 
    x = BlockMatrix(2);
else
    x = BlockMatrix(3); % Decision right
    
end

if x ==1
    y= Circle_Planet_1
    
    elseif x==2
        y = Circle_Planet_2
        
        elseif x ==3 
            y = Circle_Planet_3
            elseif x == 4
                y= Circle_Planet_4
                elseif x==5
                    y = Circle_Planet_5
                    
                else y = Circle_Planet_6
end

% Draw Circle around Stage 2
Screen('FrameOval', window, [1 0 1], [y],10); 


% Show next Stage 
Screen('Flip',window); 




%% STAGE 3

%% Collect keyboard respnse -at the moment 1 for left or anything else for
% right

[secs_2, keyCode_2, deltaSecs_2] = KbPressWait; 

Key_2= KbName(keyCode_2); 

%% use response to determine next stage
    
% Draw Background and Planets

Screen('DrawTexture',window,Background);  % Background
Screen('DrawTexture', window, Planet_1,[],[Pos_Pla_1]); % Planet 1  
Screen('DrawTexture', window, Planet_2,[],[Pos_Pla_2]); % Planet 2  
Screen('DrawTexture', window, Planet_3,[],[Pos_Pla_3]); % Planet 3
Screen('DrawTexture', window, Planet_4,[],[Pos_Pla_4]); % Planet 4
Screen('DrawTexture', window, Planet_5,[],[Pos_Pla_5]); % Planet 5  
Screen('DrawTexture', window, Planet_6,[],[Pos_Pla_6]); % Planet 6  

Screen('DrawText', window, ...
    '1', 900,500); 


%% Draw Line

line_length = Line_right - Line_left

if line_length < 500

    Screen('DrawTexture',window,Line_red, [], [Line_left Line_top Line_right Line_bottom]);
    
elseif line_length < 750
        Screen('DrawTexture',window,Line_yel, [], [Line_left Line_top Line_right Line_bottom]);
        
else 
        Screen('DrawTexture',window,Line_gre, [], [Line_left Line_top Line_right Line_bottom]);
    
end 

%% Use response to check for next stage in MiniBlockMatrix 

if (Key_1 == '1!' ) & (Key_2 == '1!') % In both Stages decision left 
    x = BlockMatrix(4);
    
elseif Key_1=='1!'  % In Stage 1 decision left, Stage 2 decision right
     x = BlockMatrix(6);
else
    x = BlockMatrix(5); % In both stages decision right
end


% check Position Circle
if x ==1
    y= Circle_Planet_1
    
    elseif x==2
        y = Circle_Planet_2
        
        elseif x ==3 
            y = Circle_Planet_3
            elseif x == 4
                y= Circle_Planet_4
                elseif x==5
                    y = Circle_Planet_5
                    
                else y = Circle_Planet_6
end

% Draw Circle around Stage 3
Screen('FrameOval', window, [1 0 1], [y],10); 

% Show next Stage 
Screen('Flip',window); 

%% STAGE 4

%% Collect keyboard respnse -at the moment 1 for left or anything else for
% right

[secs_3, keyCode_3, deltaSecs_3] = KbPressWait; 

Key_3= KbName(keyCode_3); 

%% use response to determine next stage
    
% Draw Background and Planets

Screen('DrawTexture',window,Background);  % Background
Screen('DrawTexture', window, Planet_1,[],[Pos_Pla_1]); % Planet 1  
Screen('DrawTexture', window, Planet_2,[],[Pos_Pla_2]); % Planet 2  
Screen('DrawTexture', window, Planet_3,[],[Pos_Pla_3]); % Planet 3
Screen('DrawTexture', window, Planet_4,[],[Pos_Pla_4]); % Planet 4
Screen('DrawTexture', window, Planet_5,[],[Pos_Pla_5]); % Planet 5  
Screen('DrawTexture', window, Planet_6,[],[Pos_Pla_6]); % Planet 6 

%% Draw Line

line_length = Line_right - Line_left

if line_length < 500

    Screen('DrawTexture',window,Line_red, [], [Line_left Line_top Line_right Line_bottom]);
    
elseif line_length < 750
        Screen('DrawTexture',window,Line_yel, [], [Line_left Line_top Line_right Line_bottom]);
        
else 
        Screen('DrawTexture',window,Line_gre, [], [Line_left Line_top Line_right Line_bottom]);
    
end 


%% Use response to check for next stage in MiniBlockMatrix 

if (Key_1 == '1!' ) & (Key_2 == '1!') & (Key_3 =='1!') % Decision left 
    x = BlockMatrix(8);
   
elseif (Key_1=='1!') & (Key_2 == '1!')
    x = BlockMatrix(14)
   
elseif (Key_1=='1!') & (Key_3 == '1!')
    
     x = BlockMatrix(12);
     
elseif (Key_2=='1!') & (Key_3 == '1!')
    
     x = BlockMatrix(11);
     
elseif (Key_1=='1!')     
     x = BlockMatrix(10);
     
elseif (Key_2=='1!')     
     x = BlockMatrix(13);
     
elseif (Key_3=='1!')     
     x = BlockMatrix(15);
else
    x = BlockMatrix(9); % Decision right
end


% check Position Circle
if x ==1
    y= Circle_Planet_1
    Reward = Rew_Planet_1
    
    elseif x==2
        y = Circle_Planet_2
    Reward = Rew_Planet_2
        elseif x ==3 
            y = Circle_Planet_3
            Reward = Rew_Planet_3
            elseif x == 4
                y= Circle_Planet_4
                Reward = Rew_Planet_4
                elseif x==5
                    y = Circle_Planet_5
                    Reward = Rew_Planet_5
                    
                else y = Circle_Planet_6
                    Reward = Rew_Planet_6
end

% Draw Circle around Stage 3
Screen('FrameOval', window, [1 0 1], [y],10); 

% Show next Stage 
Screen('Flip',window); 

WaitSecs(2);
Screen('DrawText', window, ...
    'End Miniblock' ,500,500,[0 0 0]); 

Screen('Flip', window);

WaitSecs(3);

%save response in Matrix

u = Key_1(1);
v = secs_1;

w = Key_2(1);
x = secs_2;

y = Key_3(1);
z = secs_3;




Response_Matrix(i,:) = [Block_ID, u,v,w,x,y,z];

end

sca

%%%%