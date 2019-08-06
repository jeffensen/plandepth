function draw_point_bar(points, window, xCenter, screenYpixels)
    max_points = 2000;
    centBar = CenterRectOnPointd([0 0 1000 50], xCenter, screenYpixels*0.1);
    pointBar = centBar + 5;
    pointBar(3) = pointBar(1) + points/2;
    pointBar(4) = pointBar(4) - 10;
    red = [1 0 0];
    blue = [0 0 1];
    
    % Draw the bar to the screen
    Screen('FillRect', window, [.15 .15 .15], centBar);
    Screen('FillRect', window, red + points*(blue-red)/max_points, pointBar);
end

