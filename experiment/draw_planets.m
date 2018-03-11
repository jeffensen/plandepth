function draw_planets( planetList, window, planetTexture, planetPos)
    for i = 1:length(planetList)
        % Draw the planet to the screen
        ptype = planetList(i);
        Screen('DrawTexture', window, planetTexture(ptype), [], planetPos(:,i)');
    end
end

