function [vbl] = move_rocket(loc_start,... 
                             loc_end,...
                             window,...
                             RocketTexture, ...
                             PlanetsTexture, ...
                             planetPos,...
                             planetList,...
                             rect, ifi, vbl)
md = .5; %movement duration
time = 0;
    while time < md
        
        draw_planets(planetList, window, PlanetsTexture, planetPos);

        % Position of the square on this frame
        xpos = loc_start(1) + time/md*(loc_end(1) - loc_start(1));
        ypos = loc_start(2) + time/md*(loc_end(2) - loc_start(2));
        
        % Center the rectangle on the centre of the screen
        cRect = CenterRectOnPointd(rect, xpos, ypos);

        % Draw the rect to the screen
        Screen('DrawTexture', window, RocketTexture, [], cRect);

        % Flip to the screen
        vbl  = Screen('Flip', window, vbl + 0.5*ifi);

        % Increment the time
        time = time + ifi;
    end

end

