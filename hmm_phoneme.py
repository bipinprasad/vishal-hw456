'''
Created on Feb 2, 2014

@author: Vishal Prasad
'''
import sys
import os
import random
import traceback
import math
from collections import defaultdict

verbose_flag = True
doVisualization = True
doRegularMaximization = True
doPhonemicMaximization = True
redirectOutput = True
cleanTrainingWord = False   # clean puncutation from training words - turned off to ensure good match between phonemic and regular normalization.

# training = None
# hmmModel = None

if doVisualization:
    import graphit

############################################################################################################
############################################################################################################

class HmmModel(object):
    def __init__(self, pi, transitionProbs, emissionProbs):
        # create a copy to make sure there are no unexpected changes
        self.forward_probabilities_by_word  = dict()
        self.backward_probabilities_by_word = dict()
        self.total_probability_by_word      = dict()
        self.p_tij_by_word                  = dict()
    
        if not pi:
            return
        
        self.state_cnt = len(pi)
        self.state_numbers   = [x for x in range(self.state_cnt)]
        self.pi = [pi[i] for i in range(self.state_cnt)]
        
        self.transitionProbs = [None for from_state in range(self.state_cnt)]
        for i in range(self.state_cnt):
            self.transitionProbs[i] = [transitionProbs[i][j] for j in range(self.state_cnt)]
            
        self.emissionProbs = [dict() for from_state in range(self.state_cnt)]
        for i in range(self.state_cnt):
            self.emissionProbs[i].update(emissionProbs[i])

        self.forward_probabilities_by_word = dict()
        self.backward_probabilities_by_word = dict()
        self.total_probability_by_word = dict()
        self.soft_count_probabilities_for_letter = dict()   # letter may mean phoneme
        
        self.allEmissionChars = set()
        for i in range(self.state_cnt):
            self.allEmissionChars.update(self.emissionProbs[i].keys())

    def setInitialParams(self, state_count, allEmissionChars):
        '''
        Set up all parameters for the Hidden Markov Model. The probabilities are all
        randomly assigned and scaled to 1.0 where appropriate.
        '''
        self.allEmissionChars = allEmissionChars
        
        useRandom       = True  # set to true to seed with random values
        
        self.state_cnt  = state_count
        self.state_numbers   = [x for x in range(self.state_cnt)]
        pi              = [0.2 for i in self.state_numbers]
        transitionProbs = [[0.1 for i in self.state_numbers] for j in self.state_numbers]
        emissionProbs   = [dict() for i in self.state_numbers]
    
        if useRandom:
                # assign random numbers to initial state probabilities and then divide by the sum to scale to a total of 1    
            for i in self.state_numbers:
                pi[i] = random.random()
            tmp = sum(pi)
            for i in self.state_numbers:
                pi[i] /= tmp
           
            # Transition Probabilities: again assign random number and scale it so that it will sum to one for each state transition
            for i in self.state_numbers:
                for j in self.state_numbers:
                    transitionProbs[i][j] = random.random()
                tmp = sum(transitionProbs[i])
                for j in self.state_numbers:
                    transitionProbs[i][j] /= tmp
        
            # Emmission Probabilities: again assign random number and scale it so that it will sum to one for each state transition
            for i in self.state_numbers:
                emissionProb     = dict()
                emissionProbs[i] = emissionProb
                for c in self.allEmissionChars:
                    emissionProb[c] = random.random()
            HmmModel.normalizeEmissionProbs(emissionProbs)
    
        else:
            pi = [0.1*(i+1) for i in self.state_numbers]   
            tmp = sum(pi)
            for i in self.state_numbers:
                pi[i] /= tmp
           
            # Transition Probabilities: again assign random number and scale it so that it will sum to one for each state transition
            for i in self.state_numbers:
                for j in self.state_numbers:
                    transitionProbs[i][j] = (i+1.0) * (j+1.0)
                tmp = sum(transitionProbs[i])
                for j in self.state_numbers:
                    transitionProbs[i][j] /= tmp
        
            # Emmission Probabilities: again assign random number and scale it so that it will sum to one for each state transition
            for i in self.state_numbers:
                emissionProb     = dict()
                emissionProbs[i] = emissionProb
                for i_char,c in enumerate(self.allEmissionChars):
                    emissionProb[c] = (i+1.0) * (i_char + 1.0)
                    
            HmmModel.normalizeEmissionProbs(emissionProbs)
            
        self.pi = pi
        self.transitionProbs = transitionProbs
        self.emissionProbs   = emissionProbs
        
        self.forward_probabilities_by_word.clear()
        self.backward_probabilities_by_word.clear()
        self.total_probability_by_word.clear()
        self.p_tij_by_word.clear()
    
    @classmethod
    def normalizeEmissionProbs(cls, emissionProbs):
        for emissionProb in emissionProbs:
            tmp = sum(emissionProb.values())
            for c in emissionProb:
                emissionProb[c] /= tmp
    
    def validateSetup(self):
        '''
        Do the transition probabilities add up to 1 and the lists the same size?
        '''
        if not self.transitionProbs:
            sys.exit('TransitionProb is required')
        
        if self.state_cnt != len(self.transitionProbs):
            sys.exit('Transition probabilities should be defined for %d states, but found only %d entries' % (self.state_cnt, len(self.transitionProbs)))
            
        for stateNum,stateTransitionProb in enumerate(self.transitionProbs):
            if self.state_cnt != len(stateTransitionProb):
                sys.exit('Transition prob for state %d has %d entries, expecting 1 entry for each of the %d states' % (stateNum, len(stateTransitionProb), self.state_cnt))
            if round(sum(stateTransitionProb),2) != round(1.0,2):
                sys.exit('State transition probabilities for state %d adds up to %s, expecting 1.0' % (stateNum, sum(stateTransitionProb)))
            
        if self.state_cnt != len(self.emissionProbs):
            sys.exit('Emission probalities should be defined for %d states, but found only %d entries' % (self.state_cnt, len(self.emissionProbs)))
    
    def charsWithZeroEmissionInAllStates(self):
        '''
        Return a list of characters that have zero emission in all states
        '''
        chars = list()
        for c in self.emissionProbs[0].keys():
            prob_across_states = sum([emissionProb[c] for emissionProb in self.emissionProbs])
            if prob_across_states <= 0.0:
                chars.append(c)
        return chars
        
    def printParameters(self, title, sortEmissionByProbability=False): 
        print '----------------------'
        print title
        print '----------------------'   
        print
        print 'Total Word Probabilities: %s for %d words' % (self.total_probability, len(self.training.training_words))
        print 'Number of states: %d' % self.state_cnt
        print 'State numbers: %s' % self.state_numbers
        print 'State Probability, pi (%s):' % sum(self.pi)
        for i in self.state_numbers:
            print '\tState %d  %s' % (i, self.pi[i])
        
        print 'Transition Probabilities:'
        for state_num in self.state_numbers:
            print '\tFrom state %d: %s, total=%s' % (state_num, self.transitionProbs[state_num], sum(self.transitionProbs[state_num]))
            
        print 'Emission Probabilities:'
        for state_num in self.state_numbers:
            emissionProb = self.emissionProbs[state_num]
            print '\tFrom state %d: total=%s for %d chars' % (state_num, sum(emissionProb.values()), len(emissionProb.keys()))
            if sortEmissionByProbability:
                # print values emission probabilities sorted with the highest probability values
                # showing up first.
                for c,v in sorted(emissionProb.items(), key=lambda char_prob: char_prob[1], reverse=True):
                    print '\t\t\t %s: %s' % (c, v)
            else:
                for c in self.allEmissionChars:
                    print '\t\t\t %s: %s' % (c, emissionProb[c])
                    
        print 'Emission Log Probability - log(State0/State1)'
        emissionProb = dict()
        for k in self.emissionProbs[0].keys():
            if self.emissionProbs[0][k] == 0.0:
                logProbability = 0.0
            elif self.emissionProbs[1][k] == 0:
                logProbability = '1000 Divide by zero: %s / %s' % (self.emissionProbs[0][k] , self.emissionProbs[1][k])
            else:
                logProbability = math.log(self.emissionProbs[0][k] / self.emissionProbs[1][k])
            emissionProb[k] = logProbability
        for c,v in sorted(emissionProb.items(), key=lambda char_prob: char_prob[1], reverse=True):
            print '\t\t\t %s: %s' % (c, v)
        
            
    
    def printTransitionsAndEmissions(self):
        
        for i in self.state_numbers:
            transitionProb = self.transitionProbs[i]
            emissionProb   = self.emissionProbs[i]
            print 'Creating State %d' % i
            for n in self.state_numbers:
                print "\tTo State\t", n, "\t", transitionProb[n]
    
            print "Emissions"
            for c in self.allEmissionChars:
                print "\tLetter\t", c, "\t", emissionProb[c]
                  
        if verbose_flag:
            print "----------------------"
            print "Pi:"
            for n in self.state_numbers:
                print "State\t", n, "\t", self.pi[n]
    

    def calculate_forward_probabilities(self, training_word):
        '''
        Forward trellis: the forward variable ai(t) is stored at (si,t) in the trellis and 
        expresses the total probability of ending up in state si at time t 
        (given that the observations o..o(t-1) were seen) - that is o(t) has not been seen yet.
        
        The first time step is initialized with pi(s) values.
        '''
        pi = self.pi
        transitionProbs = self.transitionProbs
        emissionProbs  = self.emissionProbs
        
        word_parts = training_word.parts()
        word_len = len(word_parts)    
        t_len    = word_len+1
        alpha_lattice = [] 
        
        if verbose_flag:
            print "*** word: %s : parts %s ***" % (training_word.word, ' '.join(word_parts))
            print "Forward"
        #initialization
        for t in range(t_len):
            alpha_lattice.append([])
        
        for i in range(self.state_cnt):
            alpha_lattice[0].append(pi[i])
            if verbose_flag:
                print "Pi of state\t", i, "\t", pi[i]
    
        if verbose_flag:
            t = 0
            print "\ttime ", t 
            for j in range(self.state_cnt):
                print "\t\tin state = ", j, ":", alpha_lattice[t][j] 
    
        #induction
        for t in range(word_len):
            tmp_sum_of_alphas = 0
            c = word_parts[t]
            if verbose_flag:
                print "\ttime ", t + 1, ': \'', c, '\''
            for j in range(self.state_cnt):
                if verbose_flag:
                    print "\t\tto state: ", j
                prob_sum = 0
                formulas = list()
                for i in self.state_numbers:
                    alpha_i = alpha_lattice[t][i]
                    a_ij = transitionProbs[i][j]
                    b = emissionProbs[i][c]
                    tmp = alpha_i * a_ij * b
                    formula = '%s*%s*%s=%s' % (alpha_i,a_ij,b,tmp) 
                    formulas.append(formula)
                    if verbose_flag:
                        print "\t\t  from state ", i, " previous Alpha times arc\'s a and b: ", tmp, ' using ', formula
                    prob_sum += tmp
                tmp_sum_of_alphas += prob_sum
                alpha_lattice[t+1].append(prob_sum)
    
        if verbose_flag:
            print
            for t in range(t_len):
                print "\t\tAlpha at time = ", t,
                for j in range(self.state_cnt):
                    print "\tState %d:\t%s" % (j, alpha_lattice[t][j]),
                print
                    
        if verbose_flag:
            print "\t\tSum of alpha's at time = 0: ", sum(alpha_lattice[0])
            print ""
            for t in range(1,t_len):
                print "\t\tSum of alpha's at time = ", t, ' char ', word_parts[t-1], ": ", sum(alpha_lattice[t])
        #Total
        return alpha_lattice
    
    def calculate_backward_probabilities(self, training_word):
        '''
        '''
        pi = self.pi
        transitionProbs = self.transitionProbs
        emissionProbs  = self.emissionProbs
        
        word_parts = training_word.parts()
        word_len = len(word_parts)    
        t_len    = word_len+1
        time_steps = len(word_parts)    
        beta_lattice = [] 
        
        if verbose_flag:
            print "*** word: %s : parts %s ***" % (training_word.word, ' '.join(word_parts))
            print "Backward"
        
        #initialization
        for t in range(t_len):
            beta_lattice.append([])
        for t in range(self.state_cnt):
            beta_lattice[-1].append(1.0)  # initialize last time elements to 1.0
        
        #induction
        for t in range(time_steps-1,-1,-1):
            c = word_parts[t]
            tmp_sum_of_betas = 0
            for i in self.state_numbers:
                prob_sum = 0
                formulas = list()
                for j in self.state_numbers:
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
                for i in self.state_numbers:
                    prob_sum = beta_lattice[t][i]
                    print "\tstate  %d: %s" % (i, prob_sum),
                print 
    
        return beta_lattice
        
    #Takes in a corpus file, returns the total probabilities of each word in it
    def calculate_total_probabilities(self, training):
        self.training = training
        for training_word in sorted(training.training_words, key=lambda x: x.word):
            a = self.calculate_forward_probabilities(training_word)
            b = self.calculate_backward_probabilities(training_word)
            
            self.forward_probabilities_by_word[training_word] = a
            self.backward_probabilities_by_word[training_word] = b
        
            self.total_probability_by_word[training_word] = sum(a[0][i]*b[0][i] for i in self.state_numbers)
        
        self._calculate_p_tij()
            
        self.total_probability = sum(self.total_probability_by_word.values())
        
        if verbose_flag:
            print '--------------------'
            print 'Total Probabilities:'
            print '--------------------'
        
            for training_word in sorted(training.training_words, key=lambda x: x.word):
                print '\tword %s probability: %s' % (training_word.word, self.total_probability_by_word[training_word])
    
            print '\n\tTotal Probability for all words: %s' % (self.total_probability)
    
        return self.total_probability

    def expected_count_of_state_production_of_letters(self):
        '''
        For each occurrence of each letter in the corpus, calculate the expected count of its production 
        from State S1 and its production from state S2. This is the most delicate part of the calculation. 
        These soft counts should sum to 1.0 over all of the state transitions for each letter generated.
        '''
        if not self.forward_probabilities_by_word:
            raise ValueError("self.forward_probabilities_by_word() not intialized. Call method calculate_total_probabilities()")

        self.soft_count_probabilities_for_letter = dict() 
        soft_count_probabilities_for_letter = self.soft_count_probabilities_for_letter
        

        if verbose_flag:    
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
        #     word_probs = [sum([forward_probabilities[t][i] * backward_probabilities[t][i] for i in self.state_numbers]) for t in range(time_len)]  # all the elements should be the same
        #     print 'Word %s' % (word)
        #     for t,p in enumerate(word_probs):
        #         # all these are identical. So we could just take the last alpha, where we know beta is 1
        #         print '\t(%d) calculated prob at each time: %s' % (t,p) 
        #                  
    
        tmp_alpha_total = 0.0
        tmp_beta_total  = 0.0
        # Note, that in the alpha matrix at time t, presents chars upto t-1 having been seen.
        # So element t+1 represent upto t having been seen    
        for training_word,forward_probabilities in self.forward_probabilities_by_word.items():
            backward_probabilities = self.backward_probabilities_by_word[training_word]
            word_parts = training_word.parts()
            wordlen = len(word_parts)
            tmp_alpha_total = sum(forward_probabilities[wordlen])
            tmp_beta_total  = sum([backward_probabilities[0][i] * self.pi[i] for i in self.state_numbers])
            for i_char,c in enumerate(word_parts):
                if c not in soft_count_probabilities_for_letter:
                    prob_matrix = [[0.0 for from_state in self.state_numbers] for to_state in self.state_numbers]
                    soft_count_probabilities_for_letter[c] = prob_matrix
                else:
                    prob_matrix = soft_count_probabilities_for_letter[c]
                t = i_char
                for from_state in self.state_numbers:
                    for to_state in self.state_numbers:
                        #prob = forward_probabilities[t][from_state] * self.transitionProbs[from_state][to_state] * self.emissionProbs[from_state][c] * backward_probabilities[t+1][to_state] / tmp_alpha_total 
                        prob = forward_probabilities[t][from_state] * self.transitionProbs[from_state][to_state] * self.emissionProbs[from_state][c] * backward_probabilities[t+1][to_state] 
                        prob_matrix[from_state][to_state] += prob
              
        # This matrix is NOT rationalize to 1.0 probability for each letter
        # Because we will use this same matrix to rationalize over the whole alphabet
        
        # Now print - rationalizing to probability of 1.0 for each alphabet - dont save 
        if verbose_flag:    
            print '\t\tProbability from Alphas: %s' % tmp_alpha_total
            print '\t\tProbability from Betas : %s' % tmp_beta_total
            print '\t\t'
        
        # # Another sanity check
        # # All the probabilities in the probability matrix for char # should add up to the same number
        # # we get from Alphas and Beta.
        # prob_matrix_sum = sum([prob_matrix[i][j] for prob_matrix in soft_count_probabilities_for_letter.values() for i in self.state_numbers for j in self.state_numbers])
        # prob_matrix_sum_of_hash = sum([soft_count_probabilities_for_letter['#'][i][j] for i in self.state_numbers for j in self.state_numbers])
        # print 'Sum of all the character probability matrixes=%s' % prob_matrix_sum
        # print 'Sum of all "#" character probability matrixes=%s' % prob_matrix_sum_of_hash
        #
        # That test was a success. Still on the right track.
        #
        
        for c,prob_matrix in soft_count_probabilities_for_letter.items():
            if verbose_flag: print '\t\tLetter: %s' % c
            char_soft_total = sum([prob_matrix[x][y] for x in self.state_numbers for y in self.state_numbers])
                
            for from_state in self.state_numbers:
                if verbose_flag: print '\t\t\tFrom state: %s' % from_state
                for to_state in self.state_numbers:
                    prob = prob_matrix[from_state][to_state] / char_soft_total if char_soft_total > 0.0 else 0.0
                    if verbose_flag: print '\t\t\t\tTo state: %s\t%s;' % (to_state, prob)
                        
                
    def _calculate_p_tij(self):
        '''
        calculate the p_tij matrix for each word and save it for use in recalculating the new state.
        
        self.p_tij_by_word[training_word] = p_tij
        
        '''
        
        
        for training_word,forward_probabilities in self.forward_probabilities_by_word.items():
            backward_probabilities = self.backward_probabilities_by_word[training_word]
            word_parts = training_word.parts()
            wordlen = len(word_parts)
            P_tij = [[[0.0 for from_state in self.state_numbers] for to_state in self.state_numbers] for t in range(wordlen)]
            self.p_tij_by_word[training_word] = P_tij
            
            tmp_alpha_total = sum(forward_probabilities[wordlen])
            tmp_beta_total  = sum([backward_probabilities[0][i] * self.pi[i] for i in self.state_numbers])
            
            for t,c in enumerate(word_parts):
                for from_state in self.state_numbers:
                    for to_state in self.state_numbers:
                        try:
                            prob = forward_probabilities[t][from_state] * self.transitionProbs[from_state][to_state] * self.emissionProbs[from_state][c] * backward_probabilities[t+1][to_state] # / tmp_alpha_total
                        except ZeroDivisionError:
                            prob = 0.0 
                        P_tij[t][from_state][to_state] = prob

    def maximization_part_1_new(self):
        '''
        calculate, for each state, what the total soft counts are for each letter generated by that state over 
        the whole corpus. And then normalize that into a distribution over the alphabet, which gives you the 
        probability distribution for the alphabet for that state.
        
        Return calculated_emissionProbs. This will serve as new values for subsequent loop in maximization.
        '''
        
        if verbose_flag:
            print '-----------'
            print 'Emission'
            print '-----------'
    
        calculated_emissionProbs = [defaultdict(float) for i in self.state_numbers]

        for training_word,forward_probabilities in self.forward_probabilities_by_word.items():
            backward_probabilities = self.backward_probabilities_by_word[training_word]
            word_parts = training_word.parts()
            wordlen = len(word_parts)
            p_tij = self.p_tij_by_word[training_word]
            
            
            for from_state in self.state_numbers:
                emissionProb = calculated_emissionProbs[from_state]
                '''expected number of transitions from i to j when k is observed'''
                '''expected number of transitions from i to j'''
                expected_transitions_from_state = sum( [ sum(p_tij[t][from_state]) for t in range(wordlen) ])

                for t,c in enumerate(word_parts):
                    expected_transition_from_state_for_c = sum(p_tij[t][from_state])
                    emissionProb[c] += expected_transition_from_state_for_c / expected_transitions_from_state

        HmmModel.normalizeEmissionProbs(calculated_emissionProbs)
        for from_state in self.state_numbers:
            emissionProb = calculated_emissionProbs[from_state]            
            # Now letter summary
            print
            if verbose_flag: print '\tFrom State %s' % from_state
            for c,v in emissionProb.items():
                if verbose_flag: print '\t\tLetter: "%s"    probability: %s' % (c, v)
    
        return calculated_emissionProbs
                          
                          
    def maximization_part_2_new(self):
        '''
        recalculate the transition probabilities, and the Pi probabilities and print them
        
        return recalulated values for transitionProbs and pi (for use in next iteration)
        '''   
        if verbose_flag:
            print '-----------'
            print 'Transition'
            print '-----------'
    
        soft_count_probabilities_for_time = dict()
        calculated_transitionProbs = [[0.0 for i in self.state_numbers] for j in self.state_numbers]
        
        for training_word,forward_probabilities in self.forward_probabilities_by_word.items():
            backward_probabilities = self.backward_probabilities_by_word[training_word]
            p_tij = self.p_tij_by_word[training_word]
            word_parts = training_word.parts()
            wordlen = len(word_parts)
            #tmp_alpha_total = sum(forward_probabilities[wordlen])
            #tmp_beta_total  = sum([backward_probabilities[0][i]*pi[i] for i in self.state_numbers])
            for t,c in enumerate(word_parts):
                if t not in soft_count_probabilities_for_time:
                    prob_matrix = [[0.0 for from_state in self.state_numbers] for to_state in self.state_numbers]
                    soft_count_probabilities_for_time[t] = prob_matrix
                else:
                    prob_matrix = soft_count_probabilities_for_time[t]
                for from_state in self.state_numbers:
                    for to_state in self.state_numbers:
                        prob = p_tij[t][from_state][to_state] 
                        prob_matrix[from_state][to_state] += prob
                
    
        for from_state in self.state_numbers:
            if verbose_flag: print '\tFrom State %s' % from_state
            from_state_soft_total = sum([x[from_state][to_state] for x in soft_count_probabilities_for_time.values() for to_state in self.state_numbers])
            for to_state in self.state_numbers:
                fromto_soft_total = sum([x[from_state][to_state] for x in soft_count_probabilities_for_time.values()])
                calculated_transitionProbs[from_state][to_state] = fromto_soft_total / from_state_soft_total
                if verbose_flag: print '\t\tTo State %d      prob: %s   (%s over %s)' % (to_state, calculated_transitionProbs[from_state][to_state], fromto_soft_total, from_state_soft_total)
        
        for i in self.state_numbers:
            for j in self.state_numbers:    
                if verbose_flag: print 'State %s to %s: Old transitionProbs=%s, recalced=%s, equal=%s' % (i, j, self.transitionProbs[i][j], calculated_transitionProbs[i][j], self.transitionProbs[i][j] == calculated_transitionProbs[i][j])
    
        if verbose_flag:
            print '-----------'
            print 'pi'
            print '-----------'
    
        ''' 
        This is the ratio of of time that is spent transitioning from state i at the first time slot t=0
        Unfortunately we didn't save this information in any structure
        So - recalculate  gamma
        '''
        t0_soft_total = sum([ sum(soft_count_probabilities_for_time[0][from_state]) for from_state in self.state_numbers])
        calculated_pi = [0.0 for state in self.state_numbers]                  
        for from_state in self.state_numbers:
            calculated_pi[from_state] = sum(soft_count_probabilities_for_time[0][from_state]) / t0_soft_total
            if verbose_flag: print '\tState %s : %s' % (from_state, calculated_pi[from_state])
            
        return calculated_pi, calculated_transitionProbs
         
    def maximization_part_1(self):
        '''
        calculate, for each state, what the total soft counts are for each letter generated by that state over 
        the whole corpus. And then normalize that into a distribution over the alphabet, which gives you the 
        probability distribution for the alphabet for that state.
        
        Return calculated_emissionProbs. This will serve as new values for subsequent loop in maximization.
        '''
        
        if verbose_flag:
            print '-----------'
            print 'Emission'
            print '-----------'
    
        soft_count_probabilities_for_letter = self.soft_count_probabilities_for_letter
        calculated_emissionProbs = [dict() for i in self.state_numbers]
        all_word_parts = [x for x in sorted(soft_count_probabilities_for_letter.keys())]
        
        for from_state in self.state_numbers:
            if verbose_flag: print '\tFrom State %s' % from_state
            state_soft_total = sum([soft_count_probabilities_for_letter[c][from_state][y] for c in all_word_parts for y in self.state_numbers])
            for c in all_word_parts:
                prob_matrix = soft_count_probabilities_for_letter[c]
                
                if verbose_flag: print '\t\tLetter: "%s"' % c
                
                for to_state in range(self.state_cnt):
                    prob = prob_matrix[from_state][to_state] / state_soft_total
                    if verbose_flag: print '\t\t\t\tTo state: %s\t%s;' % (to_state, prob)
                calculated_emissionProbs[from_state][c] = sum([prob_matrix[from_state][to_state] for to_state in self.state_numbers]) / state_soft_total
            
            # Now letter summary
            print
            if verbose_flag: print '\tFrom State %s' % from_state
            for c in all_word_parts:
                if verbose_flag: print '\t\tLetter: "%s"    probability: %s' % (c, calculated_emissionProbs[from_state][c])
    
        return calculated_emissionProbs
                          
    def maximization_part_2(self):
        '''
        recalculate the transition probabilities, and the Pi probabilities and print them
        
        return recalulated values for transitionProbs and pi (for use in next iteration)
        '''   
        if verbose_flag:
            print '-----------'
            print 'Transition'
            print '-----------'
    
        calculated_transitionProbs = [[0.0 for i in self.state_numbers] for j in self.state_numbers]
        
        soft_count_probabilities_for_time = dict()
        
        for training_word,forward_probabilities in self.forward_probabilities_by_word.items():
            backward_probabilities = self.backward_probabilities_by_word[training_word]
            word_parts = training_word.parts()
            wordlen = len(word_parts)
            #tmp_alpha_total = sum(forward_probabilities[wordlen])
            #tmp_beta_total  = sum([backward_probabilities[0][i]*pi[i] for i in self.state_numbers])
            for t,c in enumerate(word_parts):
                if t not in soft_count_probabilities_for_time:
                    prob_matrix = [[0.0 for from_state in self.state_numbers] for to_state in self.state_numbers]
                    soft_count_probabilities_for_time[t] = prob_matrix
                else:
                    prob_matrix = soft_count_probabilities_for_time[t]
                for from_state in self.state_numbers:
                    for to_state in self.state_numbers:
                        prob = forward_probabilities[t][from_state] * self.transitionProbs[from_state][to_state] * self.emissionProbs[from_state][c] * backward_probabilities[t+1][to_state] # / tmp_alpha_total 
                        prob_matrix[from_state][to_state] += prob
                
    
        for from_state in self.state_numbers:
            if verbose_flag: print '\tFrom State %s' % from_state
            from_state_soft_total = sum([x[from_state][to_state] for x in soft_count_probabilities_for_time.values() for to_state in self.state_numbers])
            for to_state in self.state_numbers:
                fromto_soft_total = sum([x[from_state][to_state] for x in soft_count_probabilities_for_time.values()])
                calculated_transitionProbs[from_state][to_state] = fromto_soft_total / from_state_soft_total
                if verbose_flag: print '\t\tTo State %d      prob: %s   (%s over %s)' % (to_state, calculated_transitionProbs[from_state][to_state], fromto_soft_total, from_state_soft_total)
        
        for i in self.state_numbers:
            for j in self.state_numbers:    
                if verbose_flag: print 'State %s to %s: Old transitionProbs=%s, recalced=%s, equal=%s' % (i, j, self.transitionProbs[i][j], calculated_transitionProbs[i][j], self.transitionProbs[i][j] == calculated_transitionProbs[i][j])
    
        if verbose_flag:
            print '-----------'
            print 'pi'
            print '-----------'
    
        ''' 
        This is the ratio of of time that is spent transitioning from state i at the first time slot t=0
        Unfortunately we didn't save this information in any structure
        So - recalculate  gamma
        '''
        t0_soft_total = sum([ sum(soft_count_probabilities_for_time[0][from_state]) for from_state in self.state_numbers])
        calculated_pi = [0.0 for state in self.state_numbers]                  
        for from_state in self.state_numbers:
            calculated_pi[from_state] = sum(soft_count_probabilities_for_time[0][from_state]) / t0_soft_total
            if verbose_flag: print '\tState %s : %s' % (from_state, calculated_pi[from_state])
            
        return calculated_pi, calculated_transitionProbs
         
    def getNextHmmModel(self, training):
        self.calculate_total_probabilities(training)
        self.expected_count_of_state_production_of_letters()
        
        calculated_emissionProbs = self.maximization_part_1()
        calculated_pi,calculated_transitionProbs = self.maximization_part_2()
        new_hmmModel = HmmModel(calculated_pi, calculated_transitionProbs, calculated_emissionProbs)
        new_hmmModel.calculate_total_probabilities(training)
        
        return new_hmmModel

    def showViterbiPath(self):
        
        if verbose_flag:
            print '-------------------------'
            print '      Viterbi path'
            print '-------------------------'
            print

            
        delta_by_word   = dict()
        chi_by_word     = dict()
        xhat_by_word    = dict()
        
        training_words = sorted([x for x in self.forward_probabilities_by_word.keys()], key=lambda x: x.word)
        
        for training_word in training_words:
            # initialize the delta structure with proper size
            word_parts = training_word.parts()
            t_len = len(word_parts)+1
            delta = [[0.0 for i in self.state_numbers] for t in range(t_len)]
            delta_by_word[training_word] = delta
    
            chi = [[0.0 for i in self.state_numbers] for t in range(t_len)]
            chi_by_word[training_word] = chi
            
            xhat = [0 for t in range(t_len)]
            xhat_by_word[training_word] = xhat
            
            # calculate delta as per alorithm in Manning-Schutze Cap 9, page 308
            # time is first index ( as x-axis, state is second index
            
            if verbose_flag:
                print 'Word: %s: %s' % (training_word.word, ' '.join(training_word.parts()))
                print
                
            # initialization
            if verbose_flag: print '\tTime 0'
            for state in self.state_numbers:
                delta[0][state] = self.pi[state]
                if verbose_flag: print '\t\tDelta[0] of state %s\t%s' % (state, delta[0][state])
                
            # induction
            for t in range(len(word_parts)):
                if verbose_flag: print '\tTime %s' % (t+1)
                for j in self.state_numbers:
                    if verbose_flag: print '\t\tat state %s' % j  
                    c = word_parts[t]
                    vals = [delta[t][i] * self.transitionProbs[i][j] * self.emissionProbs[i][c] for i in self.state_numbers]
                    for i,val in enumerate(vals):
                        if verbose_flag: print '\t\t\tfrom state %s: %s' % (i,val)
                    max_val = max(vals)
                    delta[t+1][j] = max_val
                    best_from_state = max([0.0 if (vals[idx]<max_val) else idx for idx in range(len(vals))]) # state that has the max value
                    chi  [t+1][j] = best_from_state
                    if verbose_flag: print '\t\t\tbest state to come from is %s: %s' % (best_from_state,max_val)
    
            # path readout
            t = t_len-1
            vals = [delta[t][i] for i in self.state_numbers]
            max_val = max(vals)
            xhat[t] = max([0 if (vals[idx]<max_val) else idx for idx in self.state_numbers])
            if verbose_flag: print '\t\ttime %s, best state to come from is %s: %s' % (t,best_from_state,max_val)
            if verbose_flag: print; print '\t\tPath readout'
            for t in range(t_len-2,-1,-1):
                x = chi[t+1][xhat[t+1]]
                xhat[t] = x
                if verbose_flag: print '\tXhat at time %s: %s' % (t, x)  
        
        for training_word in training_words:
            word_parts = training_word.parts()
            t_len = len(word_parts)+1
            delta = delta_by_word[training_word]
            chi   = chi_by_word[training_word]
            xhat  = xhat_by_word[training_word]
            
            print 
            print 'Viterbi Path for %s: %s' % (training_word.word, ' '.join(training_word.parts()))
            print '\ttime:\t', '\t'.join([str(t) for t in range(t_len)])
            print '\tstate:\t', '\t'.join([str(xhat[t]) for t in range(t_len)])
        
    def expectation_maximization(self, training, title='Expectation Maximization', min_loop_cnt=1, max_loop_cnt=1000, min_probability_increment=0.00000001):
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
        global verbose_flag
        
        if verbose_flag:
            print '%s: Performing expectation maximization by changing HMM model parameters' % title
    
        history = list()   
        hmmModel = self 
        hmmModel.calculate_total_probabilities(training)
        
        saved_verbose_flag = verbose_flag
    
        for loopCnt in range(max_loop_cnt):
            print '----------------------------------------------'
            print '%s: - Loop #%d - total probability %s' % (title,loopCnt, hmmModel.total_probability)
            print '----------------------------------------------'
            
            new_hmmModel = hmmModel.getNextHmmModel(training)
    
            total_prob     = hmmModel.total_probability 
            total_prob_new = new_hmmModel.total_probability 
            
            history.append(hmmModel)
    
            if len(history) > min_loop_cnt:
                if (total_prob_new - total_prob) < min_probability_increment:
                    print '%s: Completed after %d loops, since new probability (%s), did not change enough from old probability (%s)' % (title, loopCnt, total_prob_new, total_prob)
                    break
                chars = new_hmmModel.charsWithZeroEmissionInAllStates()
                if chars:
                    print '%s: Completed after %d loops, since new model with probability (%s), has following chars with zero emission in all states:\n\t"%s"' % (title, loopCnt, total_prob_new, '","'.join(chars))
                    break
            
            
            if verbose_flag:
                print '%s: %d loop, new probability (%s), old probability (%s)' % (title, loopCnt, total_prob_new, total_prob)
            
            hmmModel = new_hmmModel
    
        verbose_flag = saved_verbose_flag
        
        for loopCnt,hmm in enumerate(history):
            print 'Loop %d, total probability %s, transition_probs %s,emissionProbs1=%s' % (loopCnt, hmm.total_probability, hmm.transitionProbs, hmm.emissionProbs)
            
        
        max_prob = max([hmm.total_probability for hmm in history])
        hmmModel = None
        for hmm in history:
            if hmm.total_probability >= max_prob:
                hmmMax = hmm
                break
            
        print '----------------------------------------------'
        print '%s: - Loop #%d' % (title,loopCnt)
        print ' PARAMETERS AT THE END OF THE OPTMIZATION PROCESS '
        print '----------------------------------------------'
        
        hmmMax.printParameters(title, sortEmissionByProbability=True)
        if verbose_flag: hmmMax.showViterbiPath()
                
        return history, hmmMax         
    
         

############################################################################################################
############################################################################################################


class TrainingWordAlpha(object):
    def __init__(self, word):
        self.word = word
       
    def parts(self):
        return [c for c in self.word] 
        
############################################################################################################
############################################################################################################

class TrainingWordPhonemic(object):
    def __init__(self, word, word_as_phonemes):
        self.word = word
        self.phonemes = [x for x in word_as_phonemes]
        
    def parts(self):
        return [x for x in self.phonemes]

############################################################################################################
############################################################################################################

class TrainingNormal(object):
    
    def __init__(self, fname):
        self.fname = fname
        self.loadTrainingFile()

    def loadTrainingFile(self):
        if not os.path.exists(self.fname):
            sys.exit('Training file %s does not exist' % (self.fname))
        if verbose_flag:
            print '--------- Loading Training from file %s -------' % (self.fname)
            
        self.training_words = list()
        
        with open(self.fname) as f:
            content = f.readlines()
            for line in content:
                line = line.strip().lower()
                if not line: 
                    continue            # ignore empty lines
                if cleanTrainingWord:
                    cleaned_word = ''.join([x for x in line if x in 'abcdefghijklmnopqrstuvwxyz'] + ['#'])
                else:
                    cleaned_word = line if line[-1] == '#' else line + '#'
                self.training_words.append(TrainingWordAlpha(cleaned_word))
                
        self.training_chars = list(sorted(set([y for x in self.training_words for y in x.parts()])))
                
    def getAllChars(self):
        return self.training_chars
        

############################################################################################################
############################################################################################################

class TrainingPhonemic(TrainingNormal):
    
    def __init__(self, fname, phonemeFile):
        self.fname = fname
        self.phonemeFile = phonemeFile
        self.loadPhonemeFile()
        self.loadTrainingFile()

    def loadPhonemeFile(self):
        
        self.cmu_phonemes = dict()
        if not self.phonemeFile:
            return
        
        if not os.path.exists(self.phonemeFile):
            print('''
            ======================================================================
            ERROR: CMU_PHONEME_FILE %s is missing. 
            ERROR: Please copy to this directory (%s)  and rerun
            ======================================================================
            ''' % (self.phonemeFile, os.curdir))
            return
        
        with open(self.phonemeFile)as f:
            cmu_phoneme_content = f.readlines()
            
        self.cmu_phonemes = dict()
        for line in cmu_phoneme_content:
            if not line: continue
            if line[0].lower() not in 'abcdefghijklmnopqrstuvwxyz': continue;
            words = line.strip().lower().split()
            if cleanTrainingWord:
                cleaned_word = ''.join([x for x in words[0] if x in 'abcdefghijklmnopqrstuvwxyz'] + ['#'])
            else:
                cleaned_word = words[0] + '#'
            self.cmu_phonemes[cleaned_word] = [x for x in words[1:]] + ['#']
            
    def loadTrainingFile(self):
        if not os.path.exists(self.fname):
            sys.exit('Training file %s does not exist' % (self.fname))
        if verbose_flag:
            print '--------- Loading Training from file %s -------' % (self.fname)
            
        self.training_words = list()
        self.all_phonemes   = set()
        
        with open(self.fname) as f:
            content = f.readlines()
            for line in content:
                line = line.strip().lower()
                if not line: 
                    continue            # ignore empty lines
                if cleanTrainingWord:
                    cleaned_word = ''.join([x for x in line if x in 'abcdefghijklmnopqrstuvwxyz'] + ['#'])
                else:
                    cleaned_word = line if line[-1] == '#' else line + '#'
                if cleaned_word not in self.cmu_phonemes:
                    print 'Training word "%s" is missing from phoneme file "%s"' % (line, self.phonemeFile)
                    continue
                word_as_phonemes = self.cmu_phonemes[cleaned_word]
                self.training_words.append(TrainingWordPhonemic(cleaned_word, word_as_phonemes))
                for phoneme in word_as_phonemes:
                    self.all_phonemes.add(phoneme)
                
        print('--------')
        print('Phonemes')
        print('--------')
        for x in sorted(self.getAllChars()):
            print '\t%s' %x

    def getAllChars(self):
        return self.all_phonemes            
    



def visualization(history, title='Expectation-Maximization', result_dir=None):
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
        data = [(loopCnt,hmmModel.total_probability) for loopCnt,hmmModel in enumerate(history)]
        graphit.plot2d(data, title, result_dir=result_dir, xaxistitle="ProgramLoops", yaxistitle="Total Probability", mycolor=(0,0,0), minpoint=None, maxpoint=None)
    except Exception as e:
        traceback.print_exc()
        print e
        
    try: 
        data = [(hmmModel.transitionProbs[0][1], hmmModel.transitionProbs[1][0], hmmModel.total_probability) for loopCnt,hmmModel in enumerate(history)]
        graphit.plotIteration3D(data, '%s State Transition vs Total Probability' % title, result_dir=result_dir)
    except Exception as e:
        traceback.print_exc()
        print e
        
    
    #visualization is done above as part of expectation mazimization
    
def bonusProjectPhonemicTranscription(trainingFile, phonemeFile):
    '''
    From http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/ you can download the CMU
    dictionary, which gives a phonemic transcription of a large English vocabulary. Convert your wordlist
    into a phonemic representation, and analyze the data with your HMM. Explain the significance of the
    differences you find between the structure that the HMM has learned from orthographic form and from
    phonemic form
    '''
    training = TrainingPhonemic(trainingFile, phonemeFile)
    
    print('----------------------------------------')
    print('     Phonemic                           ')
    print('Using Phonemes instead of Training Words')
    print('----------------------------------------')
    
    hmmModel = HmmModel(None, None, None)
    hmmModel.setInitialParams(2, training.getAllChars())
    history, hmmMax = hmmModel.expectation_maximization(training)        
    return history, hmmMax
       
def perform_one_loop(result_dir=None):
    global verbose_flag
    
    if len(sys.argv) < 2:
        sys.exit('''
                ==============================================================
                Error: the training file name must be supplied to this program
                ==============================================================''')
    
    history = None

    if not doRegularMaximization:
        print '-----------------------------------------------------------------'
        print 'File: %s' % __file__
        print 'Maximization has been temporary disabled to do debugging.'
        print 'set doMaximization=True to reenable'
        print '-----------------------------------------------------------------'
        
    if doRegularMaximization:    
        if verbose_flag: print 'Loading training file %s' % sys.argv[1]
        training = TrainingNormal(sys.argv[1])
    
        hmmModel = HmmModel(None,None,None)
        hmmModel.setInitialParams(2, training.getAllChars())
        hmmModel.validateSetup()  # just check the geometry of variables - this method should probabably go away since methods above are generating all variables properly
    
        history, hmmMax = hmmModel.expectation_maximization(training)
    
    if not doVisualization:
        print '-----------------------------------------------------------------'
        print 'File: %s' % __file__
        print 'Visualization has been temporary disable to do debugging.'
        print 'set doVisualization=True to reenable'
        print '-----------------------------------------------------------------'
    if doVisualization and history:
        title = 'Expectation Maximization'
        visualization(history[1:], title, result_dir=result_dir)   # drop first one in history because it was randomly generated and skews the graph
    
    if not doPhonemicMaximization:
        print '-----------------------------------------------------------------'
        print 'File: %s' % __file__
        print 'Phonemic maximization has been temporary disabled to do debugging.'
        print 'set doPhonemicMaximization=True to reenable'
        print '-----------------------------------------------------------------'
        
    if doPhonemicMaximization:
        CMU_PHONEME_FILE = 'cmudict.0.7a.txt'
        history, hmmMax = bonusProjectPhonemicTranscription(sys.argv[1], CMU_PHONEME_FILE)
        if doVisualization:
            title = 'Phonemic Maximization'
            visualization(history[1:], title, result_dir=result_dir) # drop first one in history because it was randomly generated and skews the graph

def main():
    RESULT_ROOT_DIR = 'results_phoneme'
    NUM_LOOPS       = 1
    
    if not os.path.exists(RESULT_ROOT_DIR):
        os.mkdir(RESULT_ROOT_DIR)
    
    if redirectOutput:
        print 'All output will be redirected into %s' % os.path.join(RESULT_ROOT_DIR, 'result-*', 'out.log')
        
    for i in range(NUM_LOOPS):
        result_dir = os.path.join(RESULT_ROOT_DIR, 'result-%d' % i)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        stdout_file = os.path.join(result_dir, 'out.log')
        if redirectOutput:
            sys.stdout = open(stdout_file, "w")

        perform_one_loop(result_dir=result_dir)
             
if __name__ == "__main__":
    main()
    

