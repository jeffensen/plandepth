function draw_buttons(window, ButtonsTexture, buttonsPos)
    for i = 1:2
        Screen('DrawTexture', window, ButtonsTexture(i), [], buttonsPos(:,i)');
    end
end

