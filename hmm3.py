'''
Created on Feb 2, 2014

@author: Vishal Prasad
'''
import sys
import operator
import os
from collections import defaultdict
import random
import traceback


verbose_flag = False
training_words = None  # list of all words in the training initialized by loadTraingFile()
training_chars = None  # list of characters
doVisualization = True

if doVisualization:
    import graphit


forward_probabilities_by_word = dict()
backward_probabilities_by_word = dict()
total_probability_by_word = dict()

soft_count_probabilities_for_letter = dict()

 
'''
HMM PARAMETERS
 Transition probability: is a list of transition probabilities from each state. Elements in the list correspond to 
                         transition to the "other" state given by the index number. Each list must add up to one.
'''
num_states      = 2
state_numbers   = [x for x in range(num_states)]
pi              = [0.2 for i in state_numbers]
transitionProbs = [[0.5 for i in state_numbers] for j in state_numbers]
emissionProbs   = [dict() for i in state_numbers]

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
    global training_chars
    global num_states
    global state_numbers
    global pi
    global transitionProbs
    global emissionProbs
    global forward_probabilities_by_word
    global backward_probabilities_by_word
    global total_probability_by_word

    useRandom       = True  # set to true to seed with random values
    
    num_states      = state_count
    state_numbers   = [x for x in range(num_states)]
    pi              = [0.2 for i in state_numbers]
    transitionProbs = [[0.1 for i in state_numbers] for j in state_numbers]
    emissionProbs   = [dict() for i in state_numbers]

    training_chars = list(sorted(set([y for x in training_words for y in x])))
    
    if useRandom:
            # assign random numbers to initial state probabilities and then divide by the sum to scale to a total of 1    
        for i in state_numbers:
            pi[i] = random.random()
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

    forward_probabilities_by_word.clear()
    backward_probabilities_by_word.clear()
    total_probability_by_word.clear()


def printParameters(sortEmissionByProbability=False):    
    print 'Number of states: %d' % num_states
    print 'State numbers: %s' % state_numbers
    print 'State Probability, pi:' % pi
    for i in state_numbers:
        print '\tState %d  %s' % (i, pi[i])
    
    print 'Transition Probabilities:'
    for state_num in state_numbers:
        print '\tFrom state %d: %s' % (state_num, transitionProbs[state_num])
        
    print 'Emission Probabilities:'
    for state_num in state_numbers:
        emissionProb = emissionProbs[state_num]
        print '\tFrom state %d: total=%s' % (state_num, sum(emissionProb.values()))
        if sortEmissionByProbability:
            # print values emission probabilities sorted with the highest probability values
            # showing up first.
            for c,v in sorted(emissionProb.items(), key=lambda char_prob: char_prob[1], reverse=True):
                print '\t\t\t %s: %s' % (c, v)
        else:
            for c in training_chars:
                print '\t\t\t %s: %s' % (c, emissionProb[c])

    
def validateSetup():
    global emissionProbs
    '''
    Do the transition probabilities add up to 1 and the lists the same size?
    '''
    if not transitionProbs:
        sys.exit('TransitionProb is required')
    
    if num_states != len(transitionProbs):
        sys.exit('Transition probabilities should be defined for %d states, but found only %d entries' % (num_states, len(transitionProbs)))
    for stateNum,stateTransitionProb in enumerate(transitionProbs):
        if num_states != len(stateTransitionProb):
            sys.exit('Transition prob for state %d has %d entries, expecting 1 entry for each of the %d states' % (stateNum, len(stateTransitionProb), num_states))
        if round(sum(stateTransitionProb),2) != round(1.0,2):
            sys.exit('State transition probabilities for state %d adds up to %s, expecting 1.0' % (stateNum, sum(stateTransitionProb)))
        
    if num_states != len(emissionProbs):
        sys.exit('Emission probalities should be defined for %d states, but found only %d entries' % (num_states, len(emissionProbs)))

# def _loadEmissionProb(fname, use_random_emission=True):
#     '''
#     Load a file of training words. Assume that each word ends with a #.
#     
#     Return a tuple of (training_words, suggested_emission_probabilities)
#     '''
#     global training_words
#     training_words = list()
#     
#     freq = defaultdict(int)
#     with open(fname) as f:
#         content = f.readlines()
#         for line in content:
#             line = line.strip().lower()
#             if not line: 
#                 continue            # ignore empty lines
#             if line[-1] != '#':
#                 line = line + '#'   # append terminating character if it does not exist
#             training_words.append(line)
#             for c in line:
#                 freq[c] += 1
#     totalCharCnt = sum(freq.values())
#     retVal = dict()
#     for c,cnt in freq.items():
#         retVal[c] = (cnt * 1.0) / (totalCharCnt * 1.0)
#     return retVal

def printTransitionsAndEmissions():
    
    for i in state_numbers:
        transitionProb = transitionProbs[i]
        emissionProb   = emissionProbs[i]
        print 'Creating State %d' % i
        for n in state_numbers:
            print "\tTo State\t", n, "\t", transitionProb[n]

        print "Emissions"
        for c in training_chars:
            print "\tLetter\t", c, "\t", emissionProb[c]
              
    if verbose_flag:
        print "----------------------"
        print "Pi:"
        for n in state_numbers:
            print "State\t", n, "\t", pi[n]

def calculate_forward_probabilities(word):
    '''
    Forward trellis: the forward variable ai(t) is stored at (si,t) in the trellis and 
    expresses the total probability of ending up in state si at time t 
    (given that the observations o..o(t-1) were seen) - that is o(t) has not been seen yet.
    
    The first time step is initialized with pi(s) values.
    '''
    global num_states
    global states_list
    global verbose_flag

    word_len = len(word)    
    t_len    = word_len+1
    alpha_lattice = [] 
    
    if verbose_flag:
        print "*** word: ", word, "***"
        print "Forward"
    #initialization
    for t in range(t_len):
        alpha_lattice.append([])
    
    for i in range(num_states):
        alpha_lattice[0].append(pi[i])
        if verbose_flag:
            print "Pi of state\t", i, "\t", pi[i]

    if verbose_flag:
        t = 0
        print "\ttime ", t 
        for j in range(num_states):
            print "\t\tin state = ", j, ":", alpha_lattice[t][j] 

    #induction
    for t in range(word_len):
        tmp_sum_of_alphas = 0
        c = word[t]
        if verbose_flag:
            print "\ttime ", t + 1, ': \'', c, '\''
        for j in range(num_states):
            if verbose_flag:
                print "\t\tto state: ", j
            prob_sum = 0
            formulas = list()
            for i in state_numbers:
                alpha_i = alpha_lattice[t][i]
                a_ij = transitionProbs[i][j]
                b = emissionProbs[i][c]
                tmp = alpha_i * a_ij * b
                formula = '%s*%s*%s=%s' % (alpha_i,a_ij,b,tmp) 
                formulas.append(formula)
                if verbose_flag == 1:
                    print "\t\t  from state ", i, " previous Alpha times arc\'s a and b: ", tmp, ' using ', formula
                prob_sum += tmp
            tmp_sum_of_alphas += prob_sum
            alpha_lattice[t+1].append(prob_sum)

    if verbose_flag:
        print
        for t in range(t_len):
            print "\t\tAlpha at time = ", t,
            for j in range(num_states):
                print "\tState %d:\t%s" % (j, alpha_lattice[t][j]),
            print
                
    if verbose_flag:
        print "\t\tSum of alpha's at time = 0: ", sum(alpha_lattice[0])
        print ""
        for t in range(1,t_len):
            print "\t\tSum of alpha's at time = ", t, ' char ', word[t-1], ": ", sum(alpha_lattice[t])
    #Total
    return alpha_lattice

def calculate_backward_probabilities(word):
    '''
    '''
    global num_states
    global states_list
    global verbose_flag

    word_len = len(word)    
    t_len    = word_len+1
    time_steps = len(word)    
    beta_lattice = [] 
    
    if verbose_flag:
        print "*** word: ", word, "***"
        print "Backward"
    #initialization
    if verbose_flag:
        print "*** word: ", word, "***"
    
    #initialization
    for t in range(t_len):
        beta_lattice.append([])
    for t in range(num_states):
        beta_lattice[-1].append(1.0)  # initialize last time elements to 1.0
    
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
                    print "\t\t  from state ", i, " next Beta times arc\'s a and b: ", tmp, ' using ', formula
            tmp_sum_of_betas += prob_sum
            if verbose_flag:
                print "\t\tBeta at time = ", t, ", state = ", i, ":", prob_sum, ' using ', ','.join(formulas)
            beta_lattice[t].append(prob_sum)
            
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
    
#Generates the state initialization distribution. All states equiprobably.
# def generate_pi():
#     global pi
# 
#     # num_states = len(transitionProbs)
#     # for i in range(num_states):
#     #     pi.append(1.0/num_states)
# 
#     pi = [0.8057, 0.1943]

# def printForwardProbabilities(word):
#     num_states = len(transitionProbs)
#     
#     print '*** word: %s ***' % word
#     print
#     
#     for i,val in enumerate(pi):
#         print 'Pi of state %d: %s' % (i,val)
#     
#         
#     for i,c in enumerate(word):
#         print '\ttime %d: %s' % (i+1, c)
#         
#         for to_state in range(num_states):
#             print '\t\tto state: %d' % to_state
#             alpha_total = 0
#             for from_state in range(num_states):
#                 alpha = transitionProbs[from_state][to_state] * emissionProbs[to_state][c]
#                 alpha_total += alpha
#                 print '\t\t\tfrom state\t%d\tAlpha: %s' % (from_state, alpha)
#         print '\t\tTotal at this time: %s' % alpha_total
                    
#Takes in a corpus file, returns the total probabilities of each word in it
def calculate_total_probabilities():
    for word in training_words:
        calculate_forward_and_backward_probabilities(word)
        
    total_probability = sum(total_probability_by_word.values())
    
    if verbose_flag:
        print '--------------------'
        print 'Total Probabilities:'
        print '--------------------'
    
        for word in training_words:
#             word_probability = sum(total_probability_by_word[word])
#             grand_total_probability += word_probability
            print '\tword %s probability: %s' % (word, total_probability_by_word[word])

        print '\n\tTotal Probability for all words: %s' % (total_probability)

    
    return total_probability

#Calculates Alpha and Beta Probabilities, and Prints Total Probability
def calculate_forward_and_backward_probabilities(word):
    global forward_probabilities_by_word
    global backward_probabilities_by_word
    global total_probability_by_word
    
    time_range = [x for x in range(len(word)+1)]

    a = calculate_forward_probabilities(word)
    b = calculate_backward_probabilities(word)
    
    forward_probabilities_by_word[word] = a
    backward_probabilities_by_word[word] = b

    total_probability_by_word[word] = sum(a[0][i]*b[0][i] for i in state_numbers)
    
#     total_probs_by_state_for_word = [0.0 for i in state_numbers]
#     for state in state_numbers:
#         p = 0.0
#         for t in time_range:
#             p += a[t][state] * b[t][state]
#         total_probs_by_state_for_word[state] = p
#     total_probability_by_word[word] = total_probs_by_state_for_word
    
# #     word_len = len(word)
# #     word_probability = sum([a[word_len][i_state] for i_state in range(num_states)])
# #     return word_probability
#                            
#     #print ""
#     #print "Alpha"
#     if verbose_flag:
#         for time in range(len(a)):
#             print "Time ", time
#             for state in state_numbers:
#                 print"\tState ", state, ": ", a[time][state] 
#     tmp_alpha_total = 0
#     for alpha in a[len(word)]:
#         tmp_alpha_total += alpha
#     #print "Alpha total: ", tmp_alpha_total
# 
#     #print""
#     #print "Beta"
#     if verbose_flag:
#         for time in range(len(b)):
#             print "Time ", time
#             for state in range(len(b[time])):
#                 print"\tState ", state, ": ", b[time][state] 
#     tmp_beta_total = 0
#     for beta in b[0]:
#         tmp_beta_total += beta
#     #print "Beta total: ", tmp_beta_total
#     if verbose_flag:
#         print "Total probability of \'", word, "\': ", tmp_alpha_total
#     total_probability += tmp_alpha_total
# 
#     return tmp_alpha_total

def expected_count_of_state_production_of_letters():
    '''
    For each occurrence of each letter in the corpus, calculate the expected count of its production 
    from State S1 and its production from state S2. This is the most delicate part of the calculation. 
    These soft counts should sum to 1.0 over all of the state transitions for each letter generated.
    '''
    global soft_count_probabilities_for_letter
    soft_count_probabilities_for_letter = dict() # save info globally for later use
    

    print '-----------'
    print 'Soft Counts'
    print '-----------'

    ###########################################################################
    # Sanity Check code
    # At each time point sum_over_i(alpha(i,t)*beta(i,t)) is the same
    # And it the probability of observation of the word
    ###########################################################################
    #
    # Code tested and commented out below:
    # for word,forward_probabilities in forward_probabilities_by_word.items():
    #     backward_probabilities = backward_probabilities_by_word[word]
    #     wordlen = len(word)
    #     time_len = wordlen + 1
    #     word_probs = [sum([forward_probabilities[t][i] * backward_probabilities[t][i] for i in state_numbers]) for t in range(time_len)]  # all the elements should be the same
    #     print 'Word %s' % (word)
    #     for t,p in enumerate(word_probs):
    #         # all these are identical. So we could just take the last alpha, where we know beta is 1
    #         print '\t(%d) calculated prob at each time: %s' % (t,p) 
    #                  

    tmp_alpha_total = 0.0
    tmp_beta_total  = 0.0
    # Note, that in the alpha matrix at time t, presents chars upto t-1 having been seen.
    # So element t+1 represent upto t having been seen    
    for word,forward_probabilities in forward_probabilities_by_word.items():
        backward_probabilities = backward_probabilities_by_word[word]
        wordlen = len(word)
        tmp_alpha_total = sum(forward_probabilities[wordlen])
        tmp_beta_total  = sum([backward_probabilities[0][i]*pi[i] for i in state_numbers])
        for i_char,c in enumerate(word):
            if c not in soft_count_probabilities_for_letter:
                prob_matrix = [[0.0 for from_state in state_numbers] for to_state in state_numbers]
                soft_count_probabilities_for_letter[c] = prob_matrix
            else:
                prob_matrix = soft_count_probabilities_for_letter[c]
            t = i_char
            for from_state in state_numbers:
                for to_state in state_numbers:
                    prob = forward_probabilities[t][from_state] * transitionProbs[from_state][to_state] * emissionProbs[from_state][c] * backward_probabilities[t+1][to_state] / tmp_alpha_total 
                    prob_matrix[from_state][to_state] += (prob)
          
    # This matrix is NOT rationalize to 1.0 probability for each letter
    # Because we will use this same matrix to rationalize over the whole alphabet
    
    # Now print - rationalizing to probability of 1.0 for each alphabet - dont save     
    print '\t\tProbability from Alphas: %s' % tmp_alpha_total
    print '\t\tProbability from Betas : %s' % tmp_beta_total
    print '\t\t'
    
    # # Another sanity check
    # # All the probabilities in the probability matrix for char # should add up to the same number
    # # we get from Alphas and Beta.
    # prob_matrix_sum = sum([prob_matrix[i][j] for prob_matrix in soft_count_probabilities_for_letter.values() for i in state_numbers for j in state_numbers])
    # prob_matrix_sum_of_hash = sum([soft_count_probabilities_for_letter['#'][i][j] for i in state_numbers for j in state_numbers])
    # print 'Sum of all the character probability matrixes=%s' % prob_matrix_sum
    # print 'Sum of all "#" character probability matrixes=%s' % prob_matrix_sum_of_hash
    #
    # That test was a success. Still on the right track.
    #
    
    for c,prob_matrix in soft_count_probabilities_for_letter.items():
        print '\t\tLetter: %s' % c
        char_soft_total = sum([prob_matrix[x][y] for x in state_numbers for y in state_numbers])
            
        for from_state in state_numbers:
            print '\t\t\tFrom state: %s' % from_state
            for to_state in state_numbers:
                prob = prob_matrix[from_state][to_state] / char_soft_total if char_soft_total > 0.0 else 0.0
                print '\t\t\t\tTo state: %s\t%s;' % (to_state, prob)
                    
      
            
#         for i_char,c in enumerate(word):
#             print '\t\t\tLetter: %s' % c
#             total_prob_for_letter = sum([transitionProbs[from_state][to_state] * emissionProbs[to_state][c] 
#                                         for from_state in range(num_states) 
#                                             for to_state in range(num_states)])
#             if c not in soft_count_probabilities_for_letter:
#                 prob_matrix = [[None]*num_states]*num_states
#                 soft_count_probabilities_for_letter[c] = prob_matrix
#             else:
#                 prob_matrix = soft_count_probabilities_for_letter[c]
#                 
#             for from_state in range(num_states):
#                 print '\t\t\t\tFrom state: %s' % from_state
#                 for to_state in range(num_states):
#                     prob = transitionProbs[from_state][to_state] * emissionProbs[to_state][c] / total_prob_for_letter
#                     soft_count_probabilities_for_letter[c][from_state][to_state] = prob
#                     print '\t\t\t\t\tTo state: %s\t%s;' % (to_state, prob)
            
def maximization_part_1():
    '''
    calculate, for each state, what the total soft counts are for each letter generated by that state over 
    the whole corpus. And then normalize that into a distribution over the alphabet, which gives you the 
    probability distribution for the alphabet for that state.
    
    Return calculated_emissionProbs. This will serve as new values for subsequent loop in maximization.
    '''
    
    print '-----------'
    print 'Emission'
    print '-----------'

    calculated_emissionProbs = [dict() for i in state_numbers]
    
    for from_state in state_numbers:
        print '\tFrom State %s' % from_state
        state_soft_total = sum([soft_count_probabilities_for_letter[c][from_state][y] for c in training_chars for y in state_numbers])
        for c in training_chars:
            prob_matrix = soft_count_probabilities_for_letter[c]
            
            print '\t\tLetter: %s' % c
            
            for to_state in range(num_states):
                prob = prob_matrix[from_state][to_state] / state_soft_total
                print '\t\t\t\tTo state: %s\t%s;' % (to_state, prob)
            calculated_emissionProbs[from_state][c] = sum([prob_matrix[from_state][to_state] for to_state in state_numbers]) / state_soft_total
        
        # Now letter summary
        print
        print '\tFrom State %s' % from_state
        for c in training_chars:
            print '\t\tLetter: %s    probability: %s' % (c, calculated_emissionProbs[from_state][c])

    return calculated_emissionProbs
    
#     all_letters = set()
#     for word in forward_probabilities_by_word:
#         all_letters.update(word)
#         
#     if False: # all this matched what the emission probabilities show, so we are good and can comment out this code for now
#         letter_prob_by_to_state   = [defaultdict(float)]*num_states  # these both should come out to be the same!
#         letter_prob_by_from_state = [defaultdict(float)]*num_states
#         
#         for to_state in range(num_states):
#             prob_for_all_letters = sum([soft_count_probabilities_for_letter[c][from_state][to_state]
#                                         for c in all_letters 
#                                             for from_state in range(num_states)
#                                     ])
#             for c in all_letters:
#                 letter_prob_by_to_state[to_state][c] = [0.0]*num_states
#                 prob = sum([soft_count_probabilities_for_letter[c][from_state][to_state] 
#                                            for from_state in range(num_states)
#                         ])
#                 letter_prob_by_to_state[to_state][c] = prob / prob_for_all_letters
#      
#         for from_state in range(num_states):
#             prob_for_all_letters = sum([soft_count_probabilities_for_letter[c][from_state][to_state]
#                                         for c in all_letters 
#                                             for to_state in range(num_states)
#                                     ])
#             for c in all_letters:
#                 letter_prob_by_from_state[from_state][c] = [0.0]*num_states
#                 prob = sum([soft_count_probabilities_for_letter[c][from_state][to_state] 
#                                            for to_state in range(num_states)
#                         ])
#                 letter_prob_by_from_state[from_state][c] = prob / prob_for_all_letters
#                 
#         '''
#         These numbers are the same as those printed from emission_probs so nothing went amiss.
#         Comment out for now and use the emission probabilities only.
#         '''
#      
#         print '-----------'
#         print 'Emission'
#         print '-----------'
#         
#         for from_state in range(num_states):
#             print '\tFrom State %s' % from_state
#             for c in sorted(all_letters):
#                 prob = letter_prob_by_from_state[from_state][c]
#                 print '\t\tletter: %s\tprobability: %s; # to_state_prob=%s' % (from_state, prob, letter_prob_by_to_state[from_state][c])
# 
#         # End of ignored code - the code above was written to do a sanity check and see if we get back
#         # the emission_probs that we started out with - and the answer is yes. Ignore this and just use the emission probs then
#         
#      
#     print '-----------'
#     print 'Emission'
#     print '-----------'
# 
#     for from_state in range(num_states):
#         print '\tFrom state: %s' % from_state
#         for c in sorted(all_letters):
#             print '\t\tLetter: %s' % c
#             for to_state in range(num_states):
#                 prob = transitionProbs[from_state][to_state] * emissionProbs[to_state][c]
#                 print '\t\t\tTo state: %s\t%s;' % (to_state, prob)
#         for c in sorted(all_letters):
#             prob = emissionProbs[from_state][c]
#             print '\t\tletter: %s\tprobability: %s;' % (c, prob)
        
                      
def maximization_part_2():
    '''
    recalculate the transition probabilities, and the Pi probabilities and print them
    
    return recalulated values for transitionProbs and pi (for use in next iteration)
    '''   
    print '-----------'
    print 'Transition'
    print '-----------'

    calculated_transitionProbs = [[0.0 for i in state_numbers] for j in state_numbers]
    
    soft_count_probabilities_for_time = dict()
    
    for word,forward_probabilities in forward_probabilities_by_word.items():
        backward_probabilities = backward_probabilities_by_word[word]
        wordlen = len(word)
        tmp_alpha_total = sum(forward_probabilities[wordlen])
        #tmp_beta_total  = sum([backward_probabilities[0][i]*pi[i] for i in state_numbers])
        for t,c in enumerate(word):
            if t not in soft_count_probabilities_for_time:
                prob_matrix = [[0.0 for from_state in state_numbers] for to_state in state_numbers]
                soft_count_probabilities_for_time[t] = prob_matrix
            else:
                prob_matrix = soft_count_probabilities_for_time[t]
            for from_state in state_numbers:
                for to_state in state_numbers:
                    prob = forward_probabilities[t][from_state] * transitionProbs[from_state][to_state] * emissionProbs[from_state][c] * backward_probabilities[t+1][to_state] / tmp_alpha_total 
                    prob_matrix[from_state][to_state] += prob
            

    for from_state in state_numbers:
        print '\tFrom State %s' % from_state
        from_state_soft_total = sum([x[from_state][to_state] for x in soft_count_probabilities_for_time.values() for to_state in state_numbers])
        for to_state in state_numbers:
            fromto_soft_total = sum([x[from_state][to_state] for x in soft_count_probabilities_for_time.values()])
            calculated_transitionProbs[from_state][to_state] = fromto_soft_total / from_state_soft_total
            print '\t\tTo State %d      prob: %s   (%s over %s)' % (to_state, calculated_transitionProbs[from_state][to_state], fromto_soft_total, from_state_soft_total)
    
    for i in state_numbers:
        for j in state_numbers:    
            print 'State %s to %s: Old transitionProbs=%s, recalced=%s, equal=%s' % (i, j, transitionProbs[i][j], calculated_transitionProbs[i][j], transitionProbs[i][j] == calculated_transitionProbs[i][j])

    print '-----------'
    print 'pi'
    print '-----------'

    ''' 
    This is the ratio of of time that is spent transitioning from state i at the first time slot t=0
    Unfortunately we didn't save this information in any structure
    So - recalculate  gamma
    '''
    t0_soft_total = sum([ sum(soft_count_probabilities_for_time[0][from_state]) for from_state in state_numbers])
    calculated_pi = [0.0 for state in state_numbers]                  
    for from_state in state_numbers:
        calculated_pi[from_state] = sum(soft_count_probabilities_for_time[0][from_state]) / t0_soft_total
        print '\tState %s : %s' % (from_state, calculated_pi[from_state])
        
    # for state in state_numbers:
    #     transitionProb = transitionProbs[state]
    #     emissionProb   = emissionProbs[state]
    #     for word,forward_probabilities in forward_probabilities_by_word.items():
    #         backward_probabilities = backward_probabilities_by_word[word]
    #         word_prob = sum([forward_probabilities[0][m] * backward_probabilities[0][m] for m in state_numbers])
    #         next_char = word[0] # also first char
    #         numerator = sum([forward_probabilities[0][state] * 
    #                          transitionProb[to_state] * 
    #                          emissionProb[ word[0] ] *
    #                          backward_probabilities[1][to_state]
    #                         for to_state in state_numbers])
    #         state_transition_prob_for_word = numerator / word_prob
    #         if verbose_flag:
    #             print 'state %d, word "%s" %s' % (state, word, state_transition_prob_for_word)
    #         state_counts[state] += state_transition_prob_for_word

    return calculated_pi,calculated_transitionProbs
            
    # for from_state in state_numbers:
    #     print '\tFrom State %s' % from_state
    #     from_state_soft_total = sum([soft_count_probabilities_for_letter[c][from_state][to_state] for c in training_chars for to_state in state_numbers])
    #     for to_state in state_numbers:
    #         fromto_soft_total = sum([soft_count_probabilities_for_letter[c][from_state][to_state] for c in training_chars])
    #         calculated_transitionProbs[from_state][to_state] = fromto_soft_total / from_state_soft_total
    #         print '\t\tTo State %d      prob: %s   (%s over %s)' % (to_state, calculated_transitionProbs[from_state][to_state], fromto_soft_total, from_state_soft_total)
    #     
    # total_transition_to = sum([soft_count_probabilities_for_letter[c][from_state][to_state] for c in training_chars for from_state in state_numbers for to_state in state_numbers])
    # for to_state in state_numbers:
    #     prob = sum([soft_count_probabilities_for_letter[c][from_state][to_state] for c in training_chars for from_state in state_numbers])
    #     calculated_pi[to_state] = prob / total_transition_to
    #     print '\tState %s : %s' % (to_state, calculated_pi[to_state])

    # sanity check - sum the other way and see what the numbers are
    # So the numbers are definitely different.
    # What is the right way to calculate ?
    # as per the reading: 9.16, expected frequency (gamma) in state i in the first time period.
    #
    #     for from_state in state_numbers:
    #         prob = sum([soft_count_probabilities_for_letter[c][from_state][to_state] for c in training_chars for to_state in state_numbers])
    #         calculated_pi[from_state] = prob / total_transition_to
    #         print '\tState %s : %s' % (from_state, calculated_pi[from_state])
    #         

    # for from_state in state_numbers:
    #     for word,forward_probabilities in forward_probabilities_by_word.items():
    #         backward_probabilities = backward_probabilities_by_word[word]
    #         calculated_pi[from_state] += forward_probabilities[0][from_state] * backward_probabilities[0][from_state]
    # 
    # pi_total = sum(calculated_pi)
    # for state_num in state_numbers:
    #     calculated_pi[state_num] /= pi_total
    #     print '\tState %d  prob: %s' % (state_num, calculated_pi[state_num])
    
    return calculated_pi,calculated_transitionProbs
         
def expectation_maximization(title='Expectation Maximization',max_loop_cnt=1000, min_probability_increment=0.00000000001):
    '''
    Create a loop with the Expectation and Maximization functions that you have already written. 
    Need to set a stopping condition for the Expectation portion. 
    Give your program the ability to stop either 
        (i) based on the number of iterations (max_loop_cnt),
    or (ii) because the sum of the probabilities of all of the 
           words is not increasing significantly (min_probability_increment).
           
    We will vary two transition probability variables. Since this is a two state system, the other
    two transition variables are dependent (=1-x):
        p01 = transition probability of going from state0 to state1
        p10 = transition probability of going from state1 to state0
        
    And come to a maximized value.
    
    Lesson: Changing the transition probabilty makes no difference to the word probabilty.
            Ostensibly because both states have the same emission probailities so it does not
            make any difference which state the emission happens in. Therefore there is no
            change in the probability distribution of the emission.
            
    Randomly assign emission probabilities to two states and and calculate the effect
    
    Lesson 2: Randomization of emission probabilities also can a make a significant difference in
            the total probability. But the still the best state is that with no transitions!
            Optmization is hitting a dead end.
            
    '''
    global pi
    global transitionProbs
    global emissionProbs
    global verbose_flag

    if verbose_flag:
        print '%s: Performing expectation maximization by changing HMM model parameters' % title
    
    history = list()
    total_prob = calculate_total_probabilities()
    expected_count_of_state_production_of_letters()
    
    for loopCnt in range(max_loop_cnt):
        history.append((loopCnt,total_prob,pi,transitionProbs,emissionProbs))
        print '----------------------------------------------'
        print '%s: - Loop #%d' % (title,loopCnt)
        print '----------------------------------------------'
        calculated_emissionProbs = maximization_part_1()
        calculated_pi,calculated_transitionProbs = maximization_part_2()
        saved_verbose_flag = verbose_flag
        verbose_flag = False    # first loop prints are planned
        
        pi              = calculated_pi
        transitionProbs = calculated_transitionProbs
        emissionProbs   = calculated_emissionProbs
        total_prob_new  = calculate_total_probabilities()
        expected_count_of_state_production_of_letters()
        
        if (total_prob_new - total_prob) < min_probability_increment:
            print '%s: Completed after %d loops, since new probability (%s), did not change enough from old probability (%s)' % (title, loopCnt, total_prob_new, total_prob)
            break
        
        if verbose_flag:
            print '%s: %d loop, new probability (%s), old probability (%s)' % (title, loopCnt, total_prob_new, total_prob)
        total_prob = total_prob_new

    verbose_flag = saved_verbose_flag
    for loopCnt,total_prob1,pi1,transitionProbs1,emissionProbs1 in history:
        print 'Loop %d, total probability %s, transition_probs %s,emissionProbs1=%s' % (loopCnt, total_prob1, transitionProbs1, emissionProbs1)
        
    
    print '----------------------------------------------'
    print '%s: - Loop #%d' % (title,loopCnt)
    print ' PARAMETERS AT THE END OF THE OPTMIZATION PROCESS '
    print '----------------------------------------------'
    printParameters(sortEmissionByProbability=True)
            
    return history         

def showViterbiPath(verbose=True):
    
    print '-------------------------'
    print '      Viterbi path'
    print '-------------------------'
    print

    delta_by_word   = dict()
    chi_by_word     = dict()
    xhat_by_word    = dict()
    
    for word in training_words:
        # initialize the delta structure with proper size
        t_len = len(word)+1
        delta = [[0.0 for i in state_numbers] for t in range(t_len)]
        delta_by_word[word] = delta

        chi = [[0.0 for i in state_numbers] for t in range(t_len)]
        chi_by_word[word] = chi
        
        xhat = [0 for t in range(t_len)]
        xhat_by_word[word] = xhat
        
        # calculate delta as per alorithm in Manning-Schutze Cap 9, page 308
        # time is first index ( as x-axis, state is second index
        
        print 'Word: %s' % word
        print
            
        # initialization
        print '\tTime 0'
        for state in state_numbers:
            delta[0][state] = pi[state]
            if verbose: print '\t\tDelta[0] of state %s\t%s' % (state, delta[0][state])
            
        # induction
        for t in range(len(word)):
            if verbose: print '\tTime %s' % (t+1)
            for j in state_numbers:
                if verbose: print '\t\tat state %s' % j  
                c = word[t]
                vals = [delta[t][i] * transitionProbs[i][j] * emissionProbs[i][c] for i in state_numbers]
                for i,val in enumerate(vals):
                    if verbose: print '\t\t\tfrom state %s: %s' % (i,val)
                max_val = max(vals)
                delta[t+1][j] = max_val
                best_from_state = max([0.0 if (vals[idx]<max_val) else idx for idx in range(len(vals))]) # state that has the max value
                chi  [t+1][j] = best_from_state
                if verbose: print '\t\t\tbest state to come from is %s: %s' % (best_from_state,max_val)

        # path readout
        t = t_len-1
        vals = [delta[t][i] for i in state_numbers]
        max_val = max(vals)
        xhat[t] = max([0 if (vals[idx]<max_val) else idx for idx in state_numbers])
        if verbose: print '\t\ttime %s, best state to come from is %s: %s' % (t,best_from_state,max_val)
        if verbose: print; print '\t\tPath readout'
        for t in range(t_len-2,-1,-1):
            x = chi[t+1][xhat[t+1]]
            xhat[t] = x
            print '\tXhat at time %s: %s' % (t, x)  
    
    for word in training_words:
        t_len = len(word)+1
        delta = delta_by_word[word]
        chi   = chi_by_word[word]
        xhat  = xhat_by_word[word]
        
        print 
        print 'Viterbi Path:'
        print '\ttime:\t', '\t'.join([str(t) for t in range(t_len)])
        print '\tstate:\t', '\t'.join([str(xhat[t]) for t in range(t_len)])
        pass
    
     
def visualization(history, title='Expectation-Maximization'):
    '''
    First download GraphViz from here: http://www.graphviz.org/Download_windows.php
    Then install PyGraphViz from here: 
    PyGraphVis is a non-starter.
    Download PyX from here: http://pyx.sourceforge.net/
    '''
    if not doVisualization:
        print '''
            ---------------------------------------------------------------------
            doVisualization=%s flag is off and therefore graphics is turned off.
            Set this to true at top of file %s to enable graphics
            ---------------------------------------------------------------------
            ''' % (doVisualization, __file__)
        return
    
    try: 
        data = [(loopCnt,totalProb) for loopCnt,totalProb,_,_,_ in history]
        graphit.plot2d(data, title, xaxistitle="ProgramLoops", yaxistitle="Total Probability", mycolor=(0,0,0), minpoint=None, maxpoint=None)
    except Exception as e:
        traceback.print_exc()
        print e
        
    try: 
        data = [(transitionProbs1[0][1], transitionProbs1[1][0], total_prob1) for loopCnt,total_prob1,pi1,transitionProbs1,emissionProbs1 in history]
        graphit.plotIteration3D(data, '%s State Transition vs Total Probability' % title)
    except Exception as e:
        traceback.print_exc()
        print e
        
    
    #visualization is done above as part of expectation mazimization
    
def bonusProjectPhonemicTranscription():
    '''
    From http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/ you can download the CMU
    dictionary, which gives a phonemic transcription of a large English vocabulary. Convert your wordlist
    into a phonemic representation, and analyze the data with your HMM. Explain the significance of the
    differences you find between the structure that the HMM has learned from orthographic form and from
    phonemic form
    '''
    global training_words
    
    CMU_PHONEME_FILE = 'cmudict.0.7a.txt'
    if not os.path.exists(CMU_PHONEME_FILE):
        print('''
        ======================================================================
        ERROR: CMU_PHONEME_FILE %s is missing. 
        ERROR: Please copy to this directory (%s)  and rerun
        ======================================================================
        ''' % (CMU_PHONEME_FILE, os.curdir))
        return
    
    with open(CMU_PHONEME_FILE)as f:
        cmu_phoneme_content = f.readlines()
        
    cmu_phonemes = dict()
    for line in cmu_phoneme_content:
        if not line: continue
        if line[0] == ';': continue;
        words = line.strip().lower().split()
        cmu_phonemes[words[0]] = ''.join(words[1:])
        
    
    phonemes = set()
    for word in training_words:
        if word[-1] == '#':
            word = word[:-1]
        if word in cmu_phonemes:
            for x in cmu_phonemes[word].split():
                x = x.strip().lower()
                if not x:
                    continue
                if x[-1] != '#':
                    x += '#'
                phonemes.add(x)
        else:
            phonemes.add(word)

    print('--------')
    print('Phonemes')
    print('--------')
    for x in sorted(phonemes):
        print '\t%s' %x
        
    print('----------------------------------------')
    print('Using Phonemes instead of Training Words')
    print('----------------------------------------')
    
    training_words = list(phonemes)
    setInitialParams(2) # set the initialrandomized parameters
    printParameters(sortEmissionByProbability=False)
    calculate_total_probabilities() 
       
# def randomizedEmissionProbs():
#     letters = [x for x in emissionProbs[0].keys()]
#     letter_cnt = len(letters)
#     
#     retVal = [None]*num_states
#     for state_num in range(num_states):
#         random_freq = list()
#         for i in range(letter_cnt):
#             random_freq.append(random.randint(1,100))
#         totalCharCnt = sum(random_freq)
#         emissionProb = dict()
#         for i,c in enumerate(letters):
#             emissionProb[c] = (random_freq[i] * 1.0) / (totalCharCnt * 1.0)
#         retVal[state_num] = emissionProb
#     return retVal
            
def main():
    global verbose_flag
    
    if len(sys.argv) < 2:
        sys.exit('''
                ==============================================================
                Error: the training file name must be supplied to this program
                ==============================================================''')
        
    if verbose_flag: print 'Loading training file %s' % sys.argv[1]
    loadTrainingFile(sys.argv[1])
    
    setInitialParams(2) # set the initialrandomized parameters
    
    validateSetup()  # just check the geometry of variables - this method should probabably go away since methods above are generating all variables properly

    def performCalculations():
        if verbose_flag:
            printParameters()
        
        printTransitionsAndEmissions()
        calculate_total_probabilities() 
        expected_count_of_state_production_of_letters()
    
        calculated_emissionProbs = maximization_part_1()
        calculated_pi,calculated_transitionProbs = maximization_part_2()
        
        return calculated_pi, calculated_transitionProbs, calculated_emissionProbs
    
    calculated_pi, calculated_transitionProbs, calculated_emissionProbs = performCalculations()
    showViterbiPath(verbose=True)
    
    title = 'Expectation Maximization'
    history = expectation_maximization(title=title)
    doVisualization = True
    if not doVisualization:
        print '-----------------------------------------------------------------'
        print 'File: %s' % __file__
        print 'Visualization has been temporary disable to do debugging.'
        print 'set doVisualization=True to reenable'
        print '-----------------------------------------------------------------'
    if doVisualization:
        visualization(history[1:], title)   # drop first one in history becuase it was randomly generated and skews the graph
    
    doPhonemicMaximization = False
    if not doPhonemicMaximization:
        print '-----------------------------------------------------------------'
        print 'File: %s' % __file__
        print 'Phonemic maximization has been temporary disable to do debugging.'
        print 'set doPhonemicMaximization=True to reenable'
        print '-----------------------------------------------------------------'
        
    if doPhonemicMaximization:
        title = 'Phonemic Maximization'
        bonusProjectPhonemicTranscription()
        calculated_pi, calculated_transitionProbs, calculated_emissionProbs = performCalculations()
        history = expectation_maximization(title=title)
        visualization(history[1:], title) # drop first one in history becuase it was randomly generated and skews the graph
         
    #alpha_lattice = calculate_forward_probabilities('babi#')
    #beta_lattice  = calculate_backward_probabilities('babi#')
    #printForwardProbabilities('babi#')
    
if __name__ == "__main__":
    main()
    

