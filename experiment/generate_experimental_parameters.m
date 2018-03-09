
p_init = ones(1,6)/6;
n_blocks = 100;
for i=1:n_blocks
 init_pos(i) = find(cumsum(p)>=rand,1);
end