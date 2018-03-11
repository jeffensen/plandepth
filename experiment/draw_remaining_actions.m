function draw_remaining_actions(window, trial, NoTrials, xCenter, yCenter)
    % Make a base Rect
    side = 40;
    baseRect = [0 0 side side];

    % Screen X positions of our three rectangles
    x = NoTrials - 1;
    squareXpos = linspace(-x, x, NoTrials)*side + xCenter;

    % Set the colors to Red, Green and Blue
    allColors = NaN(3, NoTrials);

    % Make our rectangle coordinates
    allRects = nan(4, NoTrials);
    for i = 1:NoTrials
        allColors(:, i) = [0 1 0];
        allRects(:, i) = CenterRectOnPointd(baseRect,...
            squareXpos(i), yCenter);
    end

    if trial < NoTrials + 1
        % Draw the rect to the screen
        Screen('FillRect', window, allColors(:, trial:end), allRects(:, trial:end));
    end
end