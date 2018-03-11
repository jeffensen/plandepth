function draw_point_bar(points, window, xCenter, yCenter)
    centBar = CenterRectOnPointd([0 0 1000 50], xCenter, yCenter-400);
    pointBar = centBar + 5;
    pointBar(3) = pointBar(1) + points;
    pointBar(4) = pointBar(4) - 10;
    red = [1 0 0];
    blue = [0 0 1];
    
    % Draw the bar to the screen
    Screen('FillRect', window, [.15 .15 .15], centBar);
    Screen('FillRect', window, red + points*(blue-red)/points, pointBar);
end

