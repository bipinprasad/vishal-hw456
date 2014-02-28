import sys
import operator

verbose_flag = 1
alphabet = {}
num_total_characters = 0
total_probability = 0 

num_states = 2
states_list = []

class State:
    global alphabet
    global num_total_characters
    global verbose_flag 

    def __init__(self, state_number, num_total_states):
        #Assign state number
        self.state_number = state_number
        print "Creating State ", self.state_number

        #Assign transition probabilities
        self.transitions = []
        for n in range(num_total_states):
            self.transitions.append(0.5)

        #Assign emission probabilities
        self.emissions = []
        sorted_alphabet = sorted(alphabet.iteritems(), key=operator.itemgetter(1))
        sorted_alphabet.reverse()
        for key in sorted_alphabet:
            self.emissions.append((key[0], float(key[1]) / num_total_characters))
        #Assign pi
        self.pi = 1.0 / num_total_states

        #Assign alpha
        alpha = 1.0 

        #Assign beta
        beta = 1.0
        
        #Print
        if verbose_flag == 1:
            self.printout()
        
        
    def printout(self):
        print "Transitions"
        for n in range(len(self.transitions)):
            print "\tTo State\t", n, "\t",self.transitions[n]
        
        print "Emissions"
        for n in range(len(self.emissions)):
            print "\tLetter\t", self.emissions[n][0], "\t", self.emissions[n][1]
        
        print ""

def main():
    global verbose_flag
    global states_list 

    f = open(sys.argv[1], 'r')
    read_and_set_alphabet(f)

    create_states()

    #if verbose_flag == 1:
    #    print_pi()

    f.close()
    f = open(sys.argv[1], 'r')
    #calculate_forward_and_backward_probabilities("dadi#")
    calculate_total_probabilities(f)


#Prints the state initialization distribution.
def print_pi():
    global states_list
    
    print "------------------"
    for index in range(len(states_list)):
        print "State\t", index, "\t", states_list[index].pi

#Initializes the states
def create_states():
    global num_states
    global states_list

    for i in range(num_states):
        states_list.append(State(i,num_states))

#Reads in a corpus file, sets up the alphabet
def read_and_set_alphabet(f):
    global alphabet
    global num_total_characters
    global verbose_flag

    if verbose_flag == 1:
        print "----------------------\n-   Initialization   -\n----------------------"
    l = f.readlines()
    for line in (range(len(l))):
        word = l[line].lstrip().rstrip().rstrip('\n').lower() + "#"
        for c in word:
            num_total_characters += 1
            if alphabet.has_key(c) == False:
                alphabet[c] = 1
            else:
                alphabet[c] = alphabet[c] + 1

#Takes in a corpus file, returns the total probabilities of each word in it
def calculate_total_probabilities(f):
    global total_probability

    l = f.readlines()
    for line in (range(len(l))):
        word = l[line].lstrip().rstrip().rstrip('\n').lower() + "#"
        calculate_forward_and_backward_probabilities(word)
    print total_probability

#Calculates Alpha and Beta Probabilities, and Prints Total Probability
def calculate_forward_and_backward_probabilities(word):
    global total_probability

    a = calculate_forward_probabilities(word)
    b = calculate_backward_probabilities(word)
    
    #print ""
    #print "Alpha"
    if verbose_flag == 1:
        for time in range(len(a)):
            print "Time ", time
            for state in range(len(a[time])):
                print"\tState ", state, ": ", a[time][state] 
    tmp_alpha_total = 0
    for alpha in a[len(word) - 1]:
        tmp_alpha_total += alpha
    #print "Alpha total: ", tmp_alpha_total

    #print""
    #print "Beta"
    if verbose_flag == 1:
        for time in range(len(b)):
            print "Time ", time
            for state in range(len(b[time])):
                print"\tState ", state, ": ", b[time][state] 
    tmp_beta_total = 0
    for beta in b[0]:
        tmp_beta_total += beta
    #print "Beta total: ", tmp_beta_total

    #print "Total probability of \'", word, "\': ", tmp_alpha_total
    total_probability += tmp_alpha_total

def calculate_forward_probabilities(word):
    global num_states
    global states_list
    global verbose_flag

    time_steps = len(word)    
    alpha_lattice = [] 
    
    if verbose_flag == 1:
        print "*** word: ", word, "***"
        print "Forward"
    #initialization
    for i in range(time_steps):
        alpha_lattice.append([])
    for i in range(num_states):
        alpha_lattice[0].append(states_list[i].pi)
        if verbose_flag == 1:
            print "Pi of state\t", i, "\t", states_list[i].pi

    #induction
    for i in range(time_steps - 1):
        tmp_sum_of_alphas = 0
        if verbose_flag == 1:
            print "\ttime ", i + 1, ': \'', word[i+1], '\''
        for j in range(num_states):
            if verbose_flag == 1:
                print "\t\tto state: ", j
            prob_sum = 0
            for k in range (len(alpha_lattice[i])):
                ai = alpha_lattice[i][k]
                aij = states_list[k].transitions[j]
                b = states_list[k].emissions[i][1]
                tmp = ai * aij * b
                if verbose_flag == 1:
                    print "\t\t  from state ", k, " previous Alpha times arc\'s a and b: ", tmp
                prob_sum += tmp
            if verbose_flag == 1:
                print "\t\tAlpha at time = ", i + 1, ", state = ", j, ":", prob_sum 
            tmp_sum_of_alphas += prob_sum
            alpha_lattice[i+1].append(prob_sum)
        if verbose_flag == 1:
            print ""
            print "\t\tSum of alpha's at time = ", i + 1, ": ", tmp_sum_of_alphas
    #Total
    
    return alpha_lattice
"""    
    #printing
    for i in range(len(alpha_lattice)):
        print "Time ", i
        for j in range(len(alpha_lattice[i])):
            print alpha_lattice[i][j]
"""
           
def calculate_backward_probabilities(word):
    global num_states
    global states_list
    global verbose_flag

    time_steps = len(word)    
    beta_lattice = [] 
    
    if verbose_flag == 1:
        print "*** word: ", word, "***"
    
    #initialization
    for i in range(time_steps):
        beta_lattice.append([])
    for i in range(num_states):
        beta_lattice[time_steps - 1].append(1)
    

    #induction
    for i in range(time_steps - 1):
        tmp_sum_of_betas = 0
        for j in range(num_states):
            prob_sum = 0
            for k in range (len(beta_lattice[time_steps - 1 - i])):
                bj = beta_lattice[time_steps - 1 - i][k]
                aij = states_list[k].transitions[j]
                b = states_list[j].emissions[time_steps - 1 - i][1]
                tmp = bj * aij * b
                prob_sum += tmp
            tmp_sum_of_betas += prob_sum
            beta_lattice[time_steps - 1 - i - 1].append(prob_sum)

    return beta_lattice
"""
    #printing
    print ""
    print "Backwards"
    for i in range(len(beta_lattice)):
        print "Time ", i
        for j in range(len(beta_lattice[i])):
            print beta_lattice[i][j]
"""

def printout():
    global alphabet

    for letter in alphabet:
        print letter,'\t',alphabet[letter]

if __name__ == "__main__":
    main()
