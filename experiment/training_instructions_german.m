%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%   TRAINING  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Prepare Matlab for experiment
sca;
close all;
clear all;

% Here we call some default settings for setting up Psychtoolbox
PsychDefaultSetup(2);

% PsychTweak('UseGPUIndex', 0);

%% Load everything needed for the experiment
load('experimental_variables_new.mat')

%makes screen transparent for debugging
%PsychDebugWindowConfiguration();

Screen('Preference', 'SkipSyncTests', 1);

screen_number = max(Screen('Screens'));

% Hide mouse cursor during the experiment
HideCursor;

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
Screen('TextSize', window, 26);

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


jumps = [5 4 5 6 2 2];        
md = .5; %movement duration
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%  INSTRUCTIONS START FROM HERE   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% Introductory text
text = ['Hallo Abenteurer!' ...
        '\n\n\n\n Willkommen im Trainingslager'... 
        '\n\n\n\n Heute haben Sie die Gelegenheit an einer Reise durch den Weltraum teilzunehmen.'...
        '\n\n\n\n Um möglichst viele Punkte zu sammeln, ist es wichtig, dass Sie Ihre Reisen immer im Voraus planen.'...        
        '\n\n\n\n Bevor es losgeht, zeige ich Ihnen jedoch, wie Sie im Weltraum überleben können.'];
     
% Draw all the text in one go
DrawFormattedText(window, text, 'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9);

% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end
%%%%%%%%%%%% INSTRUCTIONS 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%text 
text = ['Sie werden durch viele verschiedene Planetensysteme reisen.' ...
        '\n\n\n\n Jedes dieser Systeme besteht aus 6 Planeten.' ...
         '\n\n\n\n Drücken Sie eine Taste, um fortzufahren.'];


% Draw all the text in one go
DrawFormattedText(window, text, 'center', screenYpixels * 0.25, white);

% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

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
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9); 

Screen('flip', window);
 
[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end;
% 
% 
% %%%%%%%%%%%% INSTRUCTIONS 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text = ['Ein Raumschiff zeigt Ihnen zu jedem Zeitpunkt Ihre aktuelle Position an.'];
        

% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.1, white);

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
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9);  

Screen('flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end
 
%%%%%%%%%%%% INSTRUCTIONS 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text = ['Kurz gesagt:'...
        '\n\n Um im Weltraum zu überleben, müssen Sie möglichst viel Treibstoff sammeln, indem Sie von Planet zu Planet reisen.' ... 
        '\n\n\n\n Sie bereisen Planeten, indem Sie immer zwischen den Kommandos RECHTE PFEILTASTE und (S)prung wählen.'...
        '\n\n\n\n Planen Sie die Reisen sorgfältig im Voraus, um Treibstoff zu sparen und zu überleben.',...
        '\n\n\n\n RECHTE PFEILTASTE kostet Sie dabei immer 2 Treibstoffeinheiten, während (S)prung 5 Treibstoffeinheiten verbraucht.'...
        '\n\n\n\n Je nachdem auf welchem Planeten Sie landen, gewinnen oder verlieren Sie Treibstoff.',...
        '\n\n\n\n Im Folgenden werde ich Ihnen alles genauer erklären.']; 


% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9);

% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end
%%%%%%%%%%%%% INTRODUCE FUEL TANK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%% INSTRUCTIONS 5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% text
text = ['Damit Sie nicht den Überblick über die Menge des verfügbaren Treibstoffes verlieren,'...
         '\n\n zeigt der Balken am oberen Bildschirmrand Ihren aktuellen Treibstofflevel an.'...
        '\n\n\n Wenn Ihr Treibstofflevel gefährlich absinkt, färbt sich der Balken rot.'];
  


% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.8); 

% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end
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
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9);

vbl = Screen('flip', window);
    
[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end
 
% %%%%%%%%%%%% PRACTISE JUMPING LEFT  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
text = ['Jetzt zeige ich Ihnen, wie man Planeten bereist und erkläre dabei die Kommandos.',...
        '\n\n\n\n Zuerst schauen wir uns an, was passiert, wenn Sie die RECHTE PFEILTASTE wählen.' ...
        '\n\n\n\n Die RECHTE PFEILTASTE ermöglicht es Ihnen, im Uhrzeigersinn zum nächsten' ...
        '\n\n benachbarten Planeten zu reisen.' ...
        '\n\n\n\n Sie können dies nun ein paar Mal ausprobieren.']; 


% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9); 

% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

NoMiniBlocks = 1;
NoTrials = 12;  %change to 12
points = 1000;
start = 1;
for t = 1:NoTrials
    while(true)
        % draw fuel tank
        draw_point_bar(points, window, xCenter, screenYpixels);
   
        % plot planets for the given mini block
        draw_planets(Practise, window, PlanetsTexture, planetsPos);

        % Draw the rocket at the starting position 
        Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,start)');
        vbl = Screen('flip', window);

        % Wait for a key press
        [secs, keyCode, deltaSecs] = KbPressWait;
        if strcmp(KbName(keyCode), 'ESCAPE')
            sca;
            return;
        elseif strcmp(KbName(keyCode), 'RightArrow')
            p = state_transition_matrix(1, start, :);
            next = find(cumsum(p)>=rand,1);

            % move the rocket
            time = 0;
            locStart = imagePos(start, :);
            locEnd = imagePos(next, :);
            points = points + actionCost(1);
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
                
                DrawFormattedText(window, '-2', 'center', yCenter-100, white);
                
                % Flip to the screen
                vbl  = Screen('Flip', window, vbl + 0.5*ifi);

                % Increment the time
                time = time + ifi;

            end
            % set start to a new location
            start = next;

            % draw fuel tank
            draw_point_bar(points, window, xCenter, screenYpixels);

            % draw planets
            draw_planets(Practise, window, PlanetsTexture, planetsPos);

            % draw_buttons(window, ButtonsTexture, buttonsPos);
            Screen('DrawTexture', window, RocketTexture, [], cRect);

            Screen('Flip', window);
            WaitSecs(1.)
            break;
        else
              DrawFormattedText(window, 'Bitte RECHTS drücken',...
                                        'center', 'center', white);
              Screen('Flip', window);
              WaitSecs(1.);
        end
    end
end

%%%%%%% SHORT BREAK%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text = ['Haben Sie bemerkt, wie der Treibstofflevel abgesunken ist?',...
         '\n\n\n\n Als nächstes zeige ich Ihnen was beim Kommando (S)prung passiert.'];

 DrawFormattedText(window, text, 'center', screenYpixels * 0.5, white);
              
% DrawFormattedText(window, 'Als nächstes zeige ich Dir was beim Kommando (S)prung passiert.',...
%                   'center', 'center', white);
                                
Screen('flip', window);

WaitSecs(4.);

% %%%%%%%%%%%% PRACTISE JUMPING Right %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text = [' Wenn Sie (S)prung wählen, reist Ihr Raumschiff zu einem nicht benachbarten Planeten.' ... 
        '\n\n\n\n Als nächstes können Sie ausprobieren, was an jeder Planetenposition passiert,'...
        '\n\n wenn Sie (S)prung wählen.'...
        '\n\n\n\n Es ist wichtig, dass Sie sich das gezeigte Flugmuster gut einprägen.']; 


% Draw all the text in one go
DrawFormattedText(window, text,'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9);

% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

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
            if strcmp(KbName(keyCode), 'ESCAPE')
                sca;
                return;
            elseif strcmp(KbName(keyCode), 's')
                p = state_transition_matrix(2, start, :);
                next = find(cumsum(p)>=rand,1);

                % move the rocket
                time = 0;
                locStart = imagePos(start, :);
                locEnd = imagePos(next, :);
                points = points + actionCost(2);
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

                    DrawFormattedText(window, '-5', 'center', yCenter-100, white);
                   
                    vbl  = Screen('Flip', window, vbl + 0.5*ifi);


                    % Increment the time
                    time = time + ifi;
               end
                
               WaitSecs(1);
               Screen('Flip', window);
               WaitSecs(1);
               break;
            else 
               DrawFormattedText(window, 'Bitte S drücken', 'center', 'center', white);
               Screen('Flip', window); 
               WaitSecs(1);                   
            end
        end
    end
end

% %%%%%%%%%%%% SHOW REISEMUSTER FOR  JUMPING Right %%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text = [' Hier sehen Sie das Flugmuster noch einmal als Ganzes.' ...
        ' Bitte prägen Sie es sich gut ein:',...
        '\n\n\n\n ']; 

    
Reisemuster= imread ('Reisemuster_Vorlage_2.jpg');
ReiseTexture= Screen('MakeTexture', window, Reisemuster);
Screen('DrawTexture', window, ReiseTexture);


% Draw all the text in one go
DrawFormattedText(window, text,'center', screenYpixels * 0.10, white);


% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9);

Screen('flip', window);

WaitSecs(2.)

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

% %%%%%%%%%%%% PRACTISE TRAVEL PATTERN  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text = [' Um Ihnen zu helfen, das Muster noch mehr zu verinnerlichen, spielen wir jetzt ein kleines Spiel.' ... 
        '\n\n\n\n Ich zeige Ihnen verschiedene Startpositionen und Sie können mit den Tasten 1-6 sagen,'...
        '\n\n\n auf welchem Zielplaneten Sie landen, wenn Sie (S)prung wählen.'...
        '\n\n\n\n Anschließend bekommen Sie ein Feedback, ob Sie richtig geantwortet haben.']; 


% Draw all the text in one go
DrawFormattedText(window, text,'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9);

% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%   Training_S_Test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NoTrials=18; %change to 18

points = 1000;
TestStarts = [1,2,3,4,5,6,2,5,1,6,3,4,3,6,4,1,5,2];
RightAnswer = [5,4,5,6,2,2,4,2,5,2,5,6,5,2,6,5,2,4];

for n = 1:NoMiniBlocks
    for t = 1:NoTrials
            start=TestStarts(t);
            
            % draw fuel tank
            draw_point_bar(points, window, xCenter, screenYpixels);
            
            % plot planets for the given mini block
            planetList = Practise;
            draw_planets(planetList, window, PlanetsTexture, planetsPos);
            
            % Draw numbers
            DrawFormattedText(window, '1', xCenter-520, yCenter+30);
            DrawFormattedText(window, '2', xCenter-220, yCenter-220);
            DrawFormattedText(window, '3', xCenter+180, yCenter-220);
            DrawFormattedText(window, '4', xCenter+480, yCenter+30);
            DrawFormattedText(window, '5', xCenter+180, yCenter+280);
            DrawFormattedText(window, '6', xCenter-220, yCenter+280);

            % Draw the rocket at the starting position 
            Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,start)');
            vbl = Screen('flip', window);

            % Wait for a key press
            [secs, keyCode, deltaSecs] = KbPressWait;
            
            Resp = KbName(keyCode);
            if strcmp(Resp, 'ESCAPE')
                sca;
                return;
            end
            Resp = str2double(Resp(1));
            
            % draw fuel tank
            draw_point_bar(points, window, xCenter, screenYpixels);
            
            % plot planets for the given mini block
            planetList = Practise;
            draw_planets(planetList, window, PlanetsTexture, planetsPos);
            
            % Draw numbers
            DrawFormattedText(window, '1', xCenter-520, yCenter+30);
            DrawFormattedText(window, '2', xCenter-220, yCenter-220);
            DrawFormattedText(window, '3', xCenter+180, yCenter-220);
            DrawFormattedText(window, '4', xCenter+480, yCenter+30);
            DrawFormattedText(window, '5', xCenter+180, yCenter+280);
            DrawFormattedText(window, '6', xCenter-220, yCenter+280);

            % Draw the rocket at the starting position 
            Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,start)');

            if Resp == RightAnswer(t)
                DrawFormattedText(window, 'Richtig', 'center', 'center', white)
                vbl = Screen('flip', window);
                WaitSecs(0.5);
            else
                DrawFormattedText(window, 'Falsch, die richtige Antwort wäre:', 'center', 'center', white);  % ask
                vbl = Screen('flip', window);
                WaitSecs(2);
            end  
                p = state_transition_matrix(2, start, :);
                next = find(cumsum(p)>=rand,1);

                % move the rocket
                time = 0;
                locStart = imagePos(start, :);
                locEnd = imagePos(next, :);
                points = points + actionCost(2);

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
            
                    % draw number        
                    DrawFormattedText(window, '1', xCenter-520, yCenter+30);
                    DrawFormattedText(window, '2', xCenter-220, yCenter-220);
                    DrawFormattedText(window, '3', xCenter+180, yCenter-220);
                    DrawFormattedText(window, '4', xCenter+480, yCenter+30);
                    DrawFormattedText(window, '5', xCenter+180, yCenter+280);
                    DrawFormattedText(window, '6', xCenter-220, yCenter+280);

                    % Draw the rect to the screen
                    Screen('DrawTexture', window, RocketTexture, [], cRect);
                    
                    % Draw travel costs
                    DrawFormattedText(window, '-5', 'center', yCenter-100, white);
                    
                    % Flip to the screen
                    vbl  = Screen('Flip', window, vbl + 0.5*ifi);

                    % Increment the time
                    time = time + ifi;
                end
               
               WaitSecs(1);
               Screen('Flip', window);
               WaitSecs(1);
    end 
 end
    



%%%%%%% SHORT BREAK%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DrawFormattedText(window, 'Als nächstes zeige ich Ihnen eine Besonderheit des Kommandos (S)prung.',...
                  'center', 'center', white);
                                
Screen('flip', window);

WaitSecs(3.);


% %%%%%%%%%%%% PRACTISE JUMP ACTION IN LOW NOISE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text = ['Im richtigen Experiment kann es passieren, dass Ihre Reise mit (S)prung unzuverlässig ist.'...
    '\n\n\n\n In solchen Fällen verfehlt das Raumschiff den Zielplaneten'...
    '\n\n und landet stattdessen auf einem der beiden Nachbarplaneten des Zielplaneten.'];

      
% Draw all the text in one go
DrawFormattedText(window, text, 'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9);

% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

% %%%%%%%%%%%% PRACTISE RIGHT ACTION IN LOW NOISE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text = ['Das Bild soll Ihnen noch einmal verdeutlichen was das bedeutet.'...
        '\n\n Die durchgezogene rote Linie zeigt den Sprung zum erwarteten Zielplaneten, '...
          '\n\n die gestrichelten Linien zeigen Ihnen wo Ihr Raumschiff landen kann, wenn es den Zielplaneten verfehlt.'];

          
Reisemuster= imread ('Reisemuster_Vorlage_3.jpg');
ReiseTexture= Screen('MakeTexture', window, Reisemuster);
Screen('DrawTexture', window, ReiseTexture);
       
% Draw all the text in one go
DrawFormattedText(window, text, 'center', screenYpixels * 0.1, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.95);

% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%% SHOW REISEMUSTER FOR  JUMPING Right %%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text = ['In manchen Planetensystemen befinden sich Asteroiden.'...
        '\n\n In diesen passiert es besonders häufig, dass Sie den Zielplaneten bei (S)prung verfehlen.'...
        '\n\n In den Planetensystemen ohne Asteroiden passiert das im Vergleich viel seltener.'...
        '\n\n\n\n\n\n Um Ihnen ein Gefühl dafür zu geben, können Sie (S)prung jetzt in beiden Bedingungen ausprobieren.'...
        '\n\n Beachten Sie, dass das Reisemuster dabei gleich bleibt und'...
        '\n\n lediglich weniger zuverlässig ist.']; 




% Draw all the text in one go
DrawFormattedText(window, text,'center', screenYpixels * 0.10, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9);

Screen('flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
text = ['Wir fangen mit Planetensystemen ohne Asteroiden an.'...
        '\n\n Hier landen Sie, wenn Sie (S)prung wählen, fast immer auf dem erwarteten Zielplaneten.'...
        '\n\n Hinweis: Die Häufigkeit mit der Sie in dieser Bedingung Ihren Zielplaneten verfehlen,'...
        '\n\n ist über das gesamte Experiment gleich.'];


% Draw all the text in one go
DrawFormattedText(window, text,'center', screenYpixels * 0.10, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9);

Screen('flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rnd_val = [0.2320    0.7325    0.9631    0.6932    0.8595    0.8387  ...
    0.0786    0.0716    0.0324    0.9084    0.6857    0.0666];
NoTrials = 12; %change to 12
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
            if strcmp(KbName(keyCode), 'ESCAPE')
                sca;
                return;
            elseif strcmp(KbName(keyCode), 's')
                p = state_transition_matrix(3, start, :);
                next = find(cumsum(p)>=rnd_val(t),1);
                if next ~= jumps(start)
                    missed = 1;
                else
                    missed = 0;
                end
                
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
                    
                    DrawFormattedText(window, '-5', 'center', yCenter-100, white);

                    vbl  = Screen('Flip', window, vbl + 0.5*ifi);

                    % Increment the time
                    time = time + ifi;
               end
               
               if missed
                    % draw fuel tank
                    draw_point_bar(points, window, xCenter, screenYpixels);

                    % plot planets for the given mini block
                    planetList = Practise;
                    draw_planets(planetList, window, PlanetsTexture, planetsPos);

                    % Draw the rocket at the starting position 
                    Screen('DrawTexture', window, RocketTexture, [], cRect);
                    DrawFormattedText(window, 'Ziel verfehlt', 'center', 'center', [1 0 0]);
                    WaitSecs(1);
                    vbl = Screen('Flip', window); 
                    WaitSecs(2);
                    break;
               else
                    WaitSecs(1);
                    Screen('Flip', window);
                    WaitSecs(1);
                    break;
               end 
               
            else
               DrawFormattedText(window, 'Bitte S drücken', 'center', 'center', white);
               Screen('Flip', window); 
               WaitSecs(1);                   
            end
        end
    end
end

 
% %%%%%%%%%%%% PRACTISE RIGHT ACTION IN HIGH NOISE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text = ['Jetzt lernen Sie Planetensysteme mit Asteroiden kennen.'...
        '\n\n\n\n Ob Sie sich in einem Planetensystem mit Asteroiden befinden,'...
        '\n\n können Sie immer am Hintergrund erkennen.'...
        '\n\n\n\n In diesen Bedingungen ist das Kommando (S)prung hochgradig unzuverlässig.' ... 
        '\n\n\n\n Um Ihnen auch hierfür ein Gefühl zu geben, können Sie nun auch hier (S)prung'...
        '\n\n ein paar Mal ausprobieren.'...
        '\n\n\n\n Beachten Sie, dass auch hier das Reisemuster gleich bleibt,'...
        '\n\n aber es auch wahrscheinlicher wird, dass Sie den Zielplaneten verfehlen.'...
        '\n\n Auch in dieser Bedingung bleibt die Häufigkeit, mit der Sie ihren Zielplaneten verfehlen, gleich.']; 


% Draw all the text in one go
DrawFormattedText(window, text, 'center', screenYpixels * 0.15, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9);

% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

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

rng(54321);
points = 1000;
rnd_val = [0.9116    0.6238    0.7918    0.4298    0.5430    0.4135    0.0856 ...
    0.7776    0.4889    0.0505    0.5384    0.0415];
NoTrials = 12;  %% change to 12 
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
        if strcmp(KbName(keyCode), 'ESCAPE')
            sca;
            return;
        elseif strcmp(KbName(keyCode), 's')
            p = state_transition_matrix(4, start, :);
            next = find(cumsum(p)>=rnd_val(t),1);
            
                if next ~= jumps(start)
                    missed = 1;
                else
                    missed = 0;
                end   

            % Move the rocket
            time = 0;
            locStart = imagePos(start, :);
            locEnd = imagePos(next, :);
            points = points + actionCost(2);

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
                
                DrawFormattedText(window, '-5', 'center', yCenter-100, white);

                vbl  = Screen('Flip', window, vbl + 0.5*ifi);


                % Increment the time
                time = time + ifi;

           end

           if missed
                % Draw debris
                Screen('DrawTexture', window, DebrisTexture, [], debrisPos)
                
                % draw fuel tank
                draw_point_bar(points, window, xCenter, screenYpixels);

                % plot planets for the given mini block
                planetList = Practise;
                draw_planets(planetList, window, PlanetsTexture, planetsPos);

                % Draw the rocket at the starting position 
                Screen('DrawTexture', window, RocketTexture, [], cRect);
                DrawFormattedText(window, 'Ziel verfehlt', 'center', 'center', [1 0 0]);
                Screen('Flip', window); 
                WaitSecs(1);
                Screen('Flip', window); 
                WaitSecs(1);
                break;
           else
                WaitSecs(1);
                Screen('Flip', window);
                WaitSecs(1);
                break;
           end 
        else 
              DrawFormattedText(window, 'Bitte S drücken', 'center', 'center', white);
              Screen('Flip', window); 
              WaitSecs(1);                   
        end
    end
end

     

%%%%%%%%%%%% INSTRUCTIONS 5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% text
text = ['Zusätzlich zu dem Treibstoff, der Sie das Reisen kostet,' ...
        '\n\n kann Sie auch das Landen auf einem Planeten Treibstoff kosten,'...
        '\n\n aber Sie können auf manchen Planeten auch zusätzlichen Treibstoff finden.'...
        '\n\n\n\n Ob und wie viel Sie gewinnen oder verlieren hängt davon ab, auf welchem Zielplaneten Sie landen.'...
        '\n\n\n\n Um zu überleben, ist es wichtig, dass Sie versuchen so viel Treibstoff wie möglich zu sammeln.'...
        '\n\n\n\n Hierzu zeige ich Ihnen im nächsten Schritt,' ...
        '\n\n welche Planeten gute und welche schlechte Treibstoffquellen sind.'];

% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9);
              
% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

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
    
DrawFormattedText(window, 'Bitte merken Sie sich die Treibstoffbelohnung für jeden Planeten: ', ...
                  'center',  screenYpixels*0.25, white); 

% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.8); 

% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

%%%%%%%%%%%% INSTRUCTIONS 5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% text
text = ['In jedem neuen Planetensystem können Sie immer nur 2 oder 3 Planeten bereisen und dort Treibstoff sammeln.'...
         '\n\n Die Anzahl der grünen Quadrate zeigt Ihnen an, wie oft Sie reisen müssen,'...
         '\n\n bevor es weiter zum nächsten Planetensystem geht.',...
         '\n\n\n\n Um zum nächsten Planetensystem reisen zu können, müssen Sie also alle grünen Quadrate aufbrauchen.'];


% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücke eine Taste um fortzufahren.', ...
                  'center', screenYpixels*0.8); 

% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

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
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9);

vbl = Screen('flip', window);
    
[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

%%%%%%%%%%% SUMMARY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Summary text
text = ['Ich fasse noch einmal alles für Sie zusammen:' ...
        '\n\n\n Ihre Aufgabe ist es, möglichst viel Treibstoff zu sammeln, indem Sie von Planet zu Planet reisen.'... 
        '\n\n\n  Sie haben entweder 2 oder 3 Reisen pro Planetensystem.'...
        '\n\n Sie können immer entweder zum Nachbarplaneten im Uhrzeigersinn reisen, was 2 Treibstoffeinheiten kostet,'...
        '\n\n oder für 5 Treibstoffeinheiten springen. Je nachdem auf welchem Planeten Sie landen,'...
        '\n\n gewinnen oder verlieren Sie Treibstoff.'... 
        '\n\n\n Der blaue Balken am oberen Rand zeigt Ihnen Ihren aktuellen Treibstoffstand.'...     
        '\n\n\n Manchmal passiert es beim Springen, dass Sie, statt auf dem erwarteten Zielplaneten,'...
        '\n\n auf einem seiner Nachbarplaneten landen. Das passiert besonders häufig in Planetensystemen'...
        '\n\n mit Asteroiden, kann aber auch (selten) in den anderen Planetensystemen passieren.'...
        '\n\n Die unterschiedlichen Häufigkeiten, wie oft Sie in den beiden Bedingungen den Zielplaneten verfehlen,'...
        '\n\n bleiben während des gesamten Experimentes gleich.'];
     
% Draw all the text in one go
DrawFormattedText(window, text, 'center', screenYpixels * 0.1, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.95);

% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

%%%%%%%%%%% TIPP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Introductory text
text = ['Hier noch ein kleiner Tipp:' ...
        '\n\n\n\n Es hilft Ihnen bei der Aufgabe, wenn Sie mehrere Schritte im Voraus planen.'... 
        '\n\n\n\n Ich zeige Ihnen das an einem Beispiel.'];
     
% Draw all the text in one go
DrawFormattedText(window, text, 'center', screenYpixels * 0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9);

% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

%%%%%%%%%%% TIPP 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text = [' Wenn Sie nur einen Schritt im Voraus planen, würden Sie wahrscheinlich direkt zum blauen Planeten springen.' ...
        '\n\n\n Danach wären Sie jedoch gezwungen, auf einen der roten Planeten zu springen.'...
        '\n\n Insgesamt würden Sie so mindestens 17 Treibstoffeinheiten verlieren (-5 +10 -2 -20)'...
        '\n\n\n\n ']; 

    
Tip= imread ('Tipp_1.jpg');
ReiseTexture= Screen('MakeTexture', window, Tip);
Screen('DrawTexture', window, ReiseTexture);


% Draw all the text in one go
DrawFormattedText(window, text,'center', screenYpixels * 0.10, white);


% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9);

% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

%%%%%%%%%%% TIPP 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text = [' In diesem Planetensystem wäre es also besser gewesen, erst wenige Punkte zu verlieren,' ...
        '\n\n\n und erst danach zum blauen Planeten zu reisen.'...
        '\n\n Insgesamt hätten Sie so nur 4 Treibstoffeinheiten verloren (-2 -10 +10 -2)'...
        '\n\n\n\n ']; 

    
Tip= imread ('Tipp_2.jpg');
ReiseTexture= Screen('MakeTexture', window, Tip);
Screen('DrawTexture', window, ReiseTexture);


% Draw all the text in one go
DrawFormattedText(window, text,'center', screenYpixels * 0.10, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.9);


% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

%%%%%%%%%%% INSTRUCTIONS 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% text
text = ['Sie können die Aufgabe nun ein paar Mal üben.'...
        '\n\n Versuchen Sie dabei so zu planen, dass Sie insgesamt den bestmöglichen Reiseweg wählen.'];


% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels*0.25, white);

% Press Key to continue  
DrawFormattedText(window, 'Drücken Sie eine Taste, um fortzufahren.', ...
                  'center', screenYpixels*0.8);

% Flip to the screen
Screen('Flip', window);

[secs, keyCode, deltaSecs] = KbPressWait;
if strcmp(KbName(keyCode), 'ESCAPE')
    sca;
    return;
end

%%%%%%%%%%%% PRACTISE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Variables

% Specify number of MiniBlocks

NoMiniBlocks = 4;
planets = Planet_Feedback;
starts = conditionsFeedback.starts;

correctResponses = [[0 1 0]; [0 1 0]; [1 0 1]; [0 0 1]];

for n = 1:NoMiniBlocks
    while true
        text = ['In Kürze erreichen Sie ein neues Planetensystem...'];
    
        % Draw all the text in one go
        DrawFormattedText(window, text,...
                      'center', screenYpixels * 0.25, white);

        % Flip to the screen
        Screen('Flip', window);
        WaitSecs(1.5);
    
        NoTrials = conditionsFeedback.notrials(n);
    
        % draw point bar
        draw_point_bar(points, window, xCenter, yCenter);
    
        % draw remaining action counter
        draw_remaining_actions(window, 1, NoTrials, xCenter, yCenter);
      
        % plot planets for the given mini block
        planetList = planets(n,:);
        draw_planets(planetList, window, PlanetsTexture, planetsPos);
    
        start = starts(n);
        % Draw the rocket at the starting position 
        Screen('DrawTexture', window, RocketTexture, [], rocketPos(:,start)');
        vbl = Screen('flip', window);
        
        TestKey = zeros(1,3);
        for t = 1:NoTrials
            % Wait for a key press
            while true
                [secs, keyCode, deltaSecs] = KbPressWait;
                Key = KbName(keyCode);
                if strcmp(Key, 'RightArrow') || strcmp(Key, 's') || strcmp(Key, 'ESCAPE')
                    break;
                end
            end
        
            if strcmp(Key, 'RightArrow')
                p = state_transition_matrix(1, start, :);
                next = find(cumsum(p)>=rand,1);
                ac = actionCost(1);
                points = points + ac;
                TestKey(t) = 1;
            elseif strcmp(Key, 's')
                p = state_transition_matrix(2, start, :);
                next = find(cumsum(p)>=rand,1);
                ac = actionCost(2);
                points = points + ac;
                TestKey(t) = 0;
            elseif strcmp(Key, 'ESCAPE')
                sca;
                return;
            end

            % move the rocket
            md = .5; %movement duration
            time = 0;
            locStart = imagePos(start, :);
            locEnd = imagePos(next, :);
            while time < md
                draw_point_bar(points, window, xCenter, yCenter);
                draw_remaining_actions(window, t, NoTrials, xCenter, yCenter);
                % draw_buttons(window, ButtonsTexture, buttonsPos);
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
            
            xpos = locEnd(1);
            ypos = locEnd(2);

            % Center the rectangle on the centre of the screen
            cRect = CenterRectOnPointd(rocketRect, xpos, ypos);
                
            % set start to a new location
            start = next;
            reward = planetRewards(planetList(next));
            points = points + reward;

            if reward > 0
                s = strcat('+', int2str(reward));
            else
                s = int2str(reward);
            end

            DrawFormattedText(window, s, xpos - 25, ypos - 120, white);
            draw_point_bar(points, window, xCenter, yCenter);
            draw_remaining_actions(window, t+1, NoTrials, xCenter, yCenter);
            draw_planets(planetList, window, PlanetsTexture, planetsPos);
            Screen('DrawTexture', window, RocketTexture, [], cRect);

            vbl = Screen('Flip', window);
        end
        WaitSecs(2);

        %% TEST FOR OPTIMAL TRAVEL PATH AND CREATE FEEDBACK %%
        if all(TestKey == correctResponses(n,:))
            text = ['Sehr gut, Sie haben den optimalen Reiseweg gewählt.'];
            correct = 1;
        else
            text = ['Guter Versuch, allerdings gibt es noch einen besseren Weg. Versuchen Sie es noch einmal...'];
            correct = 0;
        end
        
        % Draw all the text in one go
        DrawFormattedText(window, text,...
                          'center', screenYpixels*0.25, white);

        Screen('Flip', window);

        WaitSecs(2);
        if correct
            break;
        end
    end
end
  


%%%%%%%%%%%% END INSTRUCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% text
text = ['Glückwunsch! Sie haben jetzt alles gelernt, was Sie für Ihr Weltraumabenteuer wissen müssen.' ... 
         '\n\n Bitte melden Sie sich beim Versuchsleiter.'];


% Draw all the text in one go
DrawFormattedText(window, text,...
    'center', screenYpixels * 0.25, white);

% Flip to the screen
Screen('Flip', window);

WaitSecs(5);

sca