from torch import zeros, eye

def make_transition_matrix(transition_probability, ns, na):
    p = transition_probability
    transition_matrix = zeros(na, ns, ns)
    transition_matrix[0, :-1, 1:] = eye(ns-1)
    transition_matrix[0,-1,0] = 1
    transition_matrix[1, -2:, 0:3] = (1-p)/2; transition_matrix[1, -2:, 1] = p
    transition_matrix[1, 2, 3:6] = (1-p)/2; transition_matrix[1, 2, 4] = p
    transition_matrix[1, 0, 3:6] = (1-p)/2; transition_matrix[1, 0, 4] = p
    transition_matrix[1, 3, 0] = (1-p)/2; transition_matrix[1, 3, -2] = (1-p)/2; 
    transition_matrix[1, 3, -1] = p
    transition_matrix[1, 1, 2:5] = (1-p)/2; transition_matrix[1, 1, 3] = p
    
    return transition_matrix
