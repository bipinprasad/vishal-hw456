'''
Created on Feb 23, 2014

Why do the parameters stop chaging after the first loop. Another test code.

@author: Vishal Prasad
'''
import sys
import os
import random
import traceback
from decimal import Decimal

verbose_flag = True
training_words = None  # list of all words in the training initialized by loadTraingFile()

 
'''
HMM PARAMETERS
 Transition probability: is a list of transition probabilities from each state. Elements in the list correspond to 
                         transition to the "other" state given by the index number. Each list must add up to one.
'''
# num_states      = 2
# state_numbers   = [x for x in range(num_states)]
# pi              = [0.2 for i in state_numbers]
# transitionProbs = [[0.5 for i in state_numbers] for j in state_numbers]
# emissionProbs   = [dict() for i in state_numbers]

def loadTrainingFile(fname):
    global training_words

    if not os.path.exists(fname):
        sys.exit('Training file %s does not exist' % (fname))
    if verbose_flag:
        print '--------- Loading Training from file %s -------' % (fname)
        
    training_words = list()
    
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            line = line.strip().lower()
            if not line: 
                continue            # ignore empty lines
            if line[-1] != '#':
                line = line + '#'   # append terminating character if it does not exist
            training_words.append(line)


def setInitialParams(state_count):
    '''
    Set up all parameters for the Hidden Markov Model. The probabilities are all
    randomly assigned and scaled to 1.0 where appropriate.
    '''
    useRandom       = True
    
    num_states      = state_count
    state_numbers   = [x for x in range(num_states)]
    pi              = [0.2 for i in state_numbers]
    transitionProbs = [[0.1 for i in state_numbers] for j in state_numbers]
    emissionProbs   = [dict() for i in state_numbers]

    training_chars = list(sorted(set([y for x in training_words for y in x])))

    if useRandom:
        # assign random numbers to initial state probabilities and then divide by the sum to scale to a total of 1
        # Sorting, keeps the lower prob in from and 
        # this cuts down on symmetrical distribution where state0,1 are simply mirror images
        # Note that the transition and transmission probabilities have to be unordered random numbers.
        pi = [x for x in sorted([random.random() for i in state_numbers])]   
        tmp = sum(pi)
        for i in state_numbers:
            pi[i] /= tmp
       
        # Transition Probabilities: again assign random number and scale it so that it will sum to one for each state transition
        for i in state_numbers:
            for j in state_numbers:
                transitionProbs[i][j] = random.random()
            tmp = sum(transitionProbs[i])
            for j in state_numbers:
                transitionProbs[i][j] /= tmp
    
        # Emmission Probabilities: again assign random number and scale it so that it will sum to one for each state transition
        for i in state_numbers:
            emissionProb     = dict()
            emissionProbs[i] = emissionProb
            for c in training_chars:
                emissionProb[c] = random.random()
            tmp = sum(emissionProb.values())
            for c in training_chars:
                emissionProb[c] /= tmp
    else:
        pi = [0.1*(i+1) for i in state_numbers]   
        tmp = sum(pi)
        for i in state_numbers:
            pi[i] /= tmp
       
        # Transition Probabilities: again assign random number and scale it so that it will sum to one for each state transition
        for i in state_numbers:
            for j in state_numbers:
                transitionProbs[i][j] = (i+1.0) * (j+1.0)
            tmp = sum(transitionProbs[i])
            for j in state_numbers:
                transitionProbs[i][j] /= tmp
    
        # Emmission Probabilities: again assign random number and scale it so that it will sum to one for each state transition
        for i in state_numbers:
            emissionProb     = dict()
            emissionProbs[i] = emissionProb
            for i_char,c in enumerate(training_chars):
                emissionProb[c] = (i+1.0) * (i_char + 1.0)
            tmp = sum(emissionProb.values())
            for c in training_chars:
                emissionProb[c] /= tmp

    return pi,transitionProbs,emissionProbs

def calculate_forward_probabilities(word, pi, transitionProbs, emissionProbs, verbose_flag):
    '''
    Forward trellis: the forward variable ai(t) is stored at (si,t) in the trellis and 
    expresses the total probability of ending up in state si at time t 
    (given that the observations o..o(t-1) were seen) - that is o(t) has not been seen yet.
    
    The first time step is initialized with pi(s) values.
    '''
    state_numbers = [i for i in range(len(pi))]

    word_len = len(word)    
    t_len    = word_len+1
    
    if verbose_flag:
        print "*** word: ", word, "***"
        print "Forward"
        for i in state_numbers:
            print "Pi of state\t", i, "\t", pi[i]
    
    #initialization
    alpha_lattice = [[0.0 for i in state_numbers] for t in range(t_len)]
    for i in state_numbers:
        alpha_lattice[0][i] = pi[i]

#     for t in range(t_len):
#         alpha_lattice.append([])
#     
#     for i in state_numbers:
#         alpha_lattice[0].append(pi[i])
#         if verbose_flag:
#             print "Pi of state\t", i, "\t", pi[i]

    if verbose_flag:
        t = 0
        print "\ttime ", t 
        for j in state_numbers:
            print "\t\tin state = ", j, ":", alpha_lattice[t][j] 

    #induction
    for t in range(word_len):
        tmp_sum_of_alphas = 0
        c = word[t]
        if verbose_flag:
            print "\ttime ", t + 1, ': \'', c, '\''
        for j in state_numbers:
            if verbose_flag:
                print "\t\tto state: ", j
            prob_sum = 0
            formulas = list()
            for i in state_numbers:
                alpha_i = alpha_lattice[t][i]
                a_ij    = transitionProbs[i][j]
                b       = emissionProbs[i][c]
                tmp     = alpha_i * a_ij * b
                formula = '%s*%s*%s=%s' % (alpha_i,a_ij,b,tmp) 
                formulas.append(formula)
                if verbose_flag == 1:
                    print "\t\t  from state ", i, " previous Alpha times arc\'s a and b: ", tmp, ' using ', formula
                prob_sum += tmp
            tmp_sum_of_alphas += prob_sum
            alpha_lattice[t+1][j] = prob_sum

    if verbose_flag:
        print
        for t in range(t_len):
            print "\t\tAlpha at time = ", t,
            for i in state_numbers:
                print "\tState %d:\t%s" % (i, alpha_lattice[t][i]),
            print
                
    if verbose_flag:
        print "\t\tSum of alpha's at time = 0: ", sum(alpha_lattice[0])
        print ""
        for t in range(1,t_len):
            print "\t\tSum of alpha's at time = ", t, ' char ', word[t-1], ": ", sum(alpha_lattice[t])
    #Total
    return alpha_lattice

def calculate_backward_probabilities(word, pi, transitionProbs, emissionProbs, verbose_flag):
    '''
    '''
    state_numbers = [i for i in range(len(pi))]

    word_len = len(word)    
    t_len    = word_len+1
    time_steps = len(word)    
    beta_lattice = [[0.0 for i in state_numbers] for t in range(t_len)]
   
    if verbose_flag:
        print "*** word: ", word, "***"
        print "Backward"
    #initialization
    if verbose_flag:
        print "*** word: ", word, "***"
    
    #initialization
    for t in state_numbers:
        beta_lattice[-1][t] = 1.0  # initialize last time elements to 1.0
    
    #induction
    for t in range(time_steps-1,-1,-1):
        c = word[t]
        tmp_sum_of_betas = 0
        for i in state_numbers:
            prob_sum = 0
            formulas = list()
            for j in state_numbers:
                beta_j_next = beta_lattice[t+1][j]
                a_ij = transitionProbs[i][j]
                b_i = emissionProbs[i][c]
                tmp = beta_j_next * a_ij * b_i
                formula = '%s*%s*%s=%s' % (beta_j_next,a_ij,b_i,tmp)
                formulas.append(formula)
                prob_sum += tmp
                if verbose_flag:
                    print "\t\t  from state ", i, ", char ", c, ", next Beta times arc\'s a and b: ", tmp, ' using ', formula
            tmp_sum_of_betas += prob_sum
            if verbose_flag:
                print "\t\tBeta at time = ", t, ", state = ", i, ":", prob_sum, ' using ', ','.join(formulas)
            beta_lattice[t][i] = prob_sum
            
    if verbose_flag:
        print
        for t in range(t_len):
            # c = word[t-1] if t>0 else ''
            # print "\ttime ", t, ': \'', c, '\''
            print "\t\tBeta at time = ", t, 
            for i in state_numbers:
                prob_sum = beta_lattice[t][i]
                print "\tstate  %d: %s" % (i, prob_sum),
            print 

    return beta_lattice
    

def performOneCalculationLoop(
                trainingWords,
                pi,
                transitionProbs,
                emissionProbs,
                verbose_flag):
    '''
    For each occurrence of each letter in the corpus, calculate the expected count of its production 
    from State S1 and its production from state S2. This is the most delicate part of the calculation. 
    These soft counts should sum to 1.0 over all of the state transitions for each letter generated.
    '''
    verbose_alpha_beta = verbose_flag
    verbose_alpha_beta = False

    state_numbers = [i for i in range(len(pi))]
    trainingChars = set([c for word in trainingWords for c in word ])
    startChars    = [word[0] for word in trainingWords]
    
    forward_probabilities_by_word   = dict()
    backward_probabilities_by_word  = dict()
    word_probabilities              = dict()
    p_ijt                           = [[[None] for i in state_numbers] for j in state_numbers] # arc probability
    
    new_pi  = [0.0 for i in state_numbers]
    new_transitionProbs = [[0.0 for i in state_numbers] for j in state_numbers]
    new_emissionProbs   = [dict() for i in state_numbers]
    for i in state_numbers:
        for c in trainingChars:
            new_emissionProbs[i][c] = 0.0
    
    total_word_prob = 0.0
    for word in trainingWords:
        t_len = len(word)+1
        
        a = calculate_forward_probabilities(word, pi, transitionProbs, emissionProbs, verbose_flag=verbose_alpha_beta)
        b = calculate_backward_probabilities(word, pi, transitionProbs, emissionProbs, verbose_flag=verbose_alpha_beta)
        
        if verbose_alpha_beta:
            print '\tSum of word "%s" probabilities at each time should be same (sum alpha*beta at each time point):' % word
            for t in range(t_len):
                tmp = sum([a[t][i] * b[t][i] for i in state_numbers])
                print '\t\tTime %d: %s' % (t, tmp)
        total_word_prob += sum([a[len(word)][i] for i in state_numbers])
        
        p = sum([a[t_len-1][i] for i in state_numbers])

        forward_probabilities_by_word[word] = a
        backward_probabilities_by_word[word]= b
        word_probabilities[word]            = p

        for i in state_numbers:
            for j in state_numbers:
                p_ijt[i][j] = [0.0 for t in range(len(word))] 
        for i in state_numbers:
            for j in state_numbers:
                for t in range(len(word)):
                    c = word[t]
                    p_ijt[i][j][t] += a[t][i] * transitionProbs[i][j] * emissionProbs[i][c] * b[t+1][j]
        
        for i in state_numbers:
            # print 'p_ijt at time 0: %s, char %s, alpha[0][i]=%s, beta[1]=%s, transition=%s, emission=%s' % ([p_ijt[i][j][0] for j in state_numbers], word[0], a[0][i], b[1], transitionProbs[i], emissionProbs[i][word[0]])
            tmp_prob = sum([p_ijt[i][j][0] for j in state_numbers]) # lot of underflow in beta at t=1 and thus p_ijt at t=0
            #tmp_prob = sum([transitionProbs[i][j] * emissionProbs[i][c] for j in state_numbers for c in startChars])
            new_pi[i] += tmp_prob
            
        for i in state_numbers:
            gamma_sum = sum([p_ijt[i][j][t] for j in state_numbers for t in range(len(word))])
            gamma_sum = 1.0     # this removes the denominator, effectively summing over ALL words
            for j in state_numbers:
                new_transitionProbs[i][j] += (sum([p_ijt[i][j][t] for t in range(len(word))]) / gamma_sum)
                
        for i in state_numbers:
            denom = sum([p_ijt[i][j][t] for t in range(len(word)) for j in state_numbers])
            denom = 1.0         # this removes the denominator, effectively summing over ALL words
            for t in range(len(word)):
                c = word[t]
                new_emissionProbs[i][c] += (sum([p_ijt[i][j][t] for t in range(len(word)) for j in state_numbers]) / denom)
        

    # normalize probabilities
    subtotal_pi         = sum([new_pi[i] for i in state_numbers])
    for i in state_numbers:
        new_pi[i] /= subtotal_pi
        subtotal_transition = sum([new_transitionProbs[i][j] for j in state_numbers])
        subtotal_emission   = sum([new_emissionProbs[i][c] for c in trainingChars])
        for j in state_numbers:
            new_transitionProbs[i][j] /= subtotal_transition
        for c in trainingChars:
            new_emissionProbs[i][c] /= subtotal_emission

    return new_pi, new_transitionProbs, new_emissionProbs, total_word_prob        

def performCalculationLoops(
                title,
                loopCnt,
                total_prob,                 
                trainingWords,
                pi,
                transitionProbs,
                emissionProbs,
                verbose_flag):
    
    loopCnt += 1
    
    state_numbers = [i for i in range(len(pi))]
    
    print '----------------------------------------------'
    print '%s: - Loop #%d' % (title,loopCnt)
    print '----------------------------------------------'
    new_pi, new_transitionProbs, new_emissionProbs, new_total_prob = performOneCalculationLoop(
                                                                        trainingWords,
                                                                        pi,
                                                                        transitionProbs,
                                                                        emissionProbs,
                                                                        verbose_flag)
    print '%s: %d loop, new probability (%s), old probability (%s)' % (title, loopCnt, new_total_prob, total_prob)
    print '%s: %d loop, new pi (%s), old pi (%s), sum(new)=%s, sum (old)=%s' % (title, loopCnt, new_pi, pi, sum(new_pi), sum(pi))
    print '%s: %d loop, new transition=%s, old transition=%s, sum(new)=%s, sum(old)=%s' % (title, loopCnt, new_transitionProbs, transitionProbs, 
                                                                                           [sum(new_transitionProbs[from_state]) for from_state in state_numbers],
                                                                                           [sum(transitionProbs[from_state]) for from_state in state_numbers])
    print '%s: %d loop, new emission (%s), old emission (%s), sum(new)=%s, sum(old)=%s' % (title, loopCnt, new_emissionProbs, emissionProbs,
                                                                   [sum(new_emissionProbs[from_state].values()) for from_state in state_numbers],
                                                                   [sum(emissionProbs[from_state].values()) for from_state in state_numbers])
    
    if abs(new_total_prob - total_prob) < 0.00000001: 
        print '%s: Completed after %d loops, since new probability (%s), did not change enough from old probability (%s)' % (title, loopCnt, new_total_prob, total_prob)
        return new_pi, new_transitionProbs, new_emissionProbs, new_total_prob
    if loopCnt >= 1000:
        print '%s: Completed after %d loops, since loopCnt exceeded allowed limit, new probability (%s)' % (title, loopCnt, new_total_prob)
        return new_pi, new_transitionProbs, new_emissionProbs, new_total_prob
        
    return performCalculationLoops(
                title,
                loopCnt,
                new_total_prob,                 
                trainingWords,
                new_pi,
                new_transitionProbs,
                new_emissionProbs,
                verbose_flag)

def main():
    global verbose_flag
    
    if len(sys.argv) < 2:
        sys.exit('''
                ==============================================================
                Error: the training file name must be supplied to this program
                ==============================================================''')
        
    if verbose_flag: print 'Loading training file %s' % sys.argv[1]
    loadTrainingFile(sys.argv[1])
    
    pi,transitionProbs,emissionProbs = setInitialParams(2) # set the initialrandomized parameters

    new_pi, new_transitionProbs, new_emissionProbs, new_total_prob = performCalculationLoops(
                'Training',
                0,
                0.0,                 
                training_words,
                pi,
                transitionProbs,
                emissionProbs,
                verbose_flag)


if __name__ == '__main__':
    main()