import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from tqdm.notebook import tqdm 


class Qlearning:
    def __init__(self, alpha=0.1, num_steps=10000, temperature=2):
        self.alpha = alpha # learning rate
        self.num_steps = num_steps
        self.temperature = temperature # epsilon value for epsilon-greedy policy
        
        # List to store Q values and rewards
        self.state0_rewards = [0]*num_steps
        self.state1_rewards = [0]*num_steps
        
        # Initialize Q-matrix with zeros
        self.Q = np.zeros((1+num_steps,2, 2))
        
        # Define state transition probabilities
        self.P = np.array([[[0.25, 0.75], [0.75, 0.25]],
                           [[0.75, 0.25], [0.25, 0.75]]])
        
    def run(self):
        # Initialize state randomly
        state = np.random.randint(0, 2)

        # Q-learning algorithm
        for i in tqdm(range(self.num_steps), desc = "steps"):
            logits = self.Q[i][state]/self.temperature
            # Compute softmax probabilities
            probs = np.exp(logits) / np.sum(np.exp(logits))

            ### epsilon part 
            # if np.random.random() < epsilon:
            #     # Choose a random action
            #     action = np.random.randint(0, 2)
            # else:
            #     # Choose the action with the highest Q-value
            #     action = np.argmax(Q[state])


            ### Softmax part
            ##action chosen randomly on the basis of the probabilities
            action = np.random.choice([0, 1], p=probs)
            next_state = np.random.choice([0, 1], p=self.P[state, action])
            reward = -1 if next_state == state else 1

            if state == 0:
                self.state0_rewards[i] = reward
                self.state1_rewards[i] = None
            else:
                self.state1_rewards[i] = reward
                self.state0_rewards[i] = None
            
            # Update Q-value for current state-action pair
            
            self.Q[i][state, action] = self.Q[i-1][state, action]-self.alpha * (self.Q[i-1][state, action] - reward)
            self.Q[i+1] = self.Q[i]
            state = next_state # update state
            
    def plot(self):
        # Plot Q values and reward over time
        fig, axs = plt.subplots(2)
        axs[0].plot(self.Q[:, 0, 0], label='Q(0,A)')
        axs[0].plot(self.Q[:, 0, 1], label='Q(0,B)')
        axs[0].plot(self.Q[:, 1, 0], label='Q(1,A)')
        axs[0].plot(self.Q[:, 1, 1], label='Q(1,B)')
        axs[0].legend()

        axs[1].plot(self.state0_rewards, 'o', markersize = 2, label ="Rewards for state 0" )
        axs[1].plot(self.state1_rewards, 'o', markersize = 2, label = "Rewards for state 1")
        axs[1].legend()

        axs[0].set_xlabel('Time step')
        axs[0].set_ylabel('Q-value')
        plt.show()
        print(self.Q.mean(axis = 0))








### Confirmation bias

class Qlearning_CB:
    def __init__(self, alpha_plus=0.01, num_steps=10000, temperature=2, alpha_minus=0.01, p = .25):
        self.alpha_plus = alpha_plus # learning rate
        self.alpha_minus = alpha_minus # confirmation bias factor
        self.p = p
			# Define state transition probabilities
        self.P = np.array([[[p, 1-p], [1-p, p]],
                           [[1-p, p], [p, 1-p]]])
        



        self.num_steps = num_steps
        self.temperature = temperature 
        
        # List to store Q values and rewards
        self.state0_rewards = [0]*num_steps
        self.state1_rewards = [0]*num_steps
        self.total_reward = 0
        # Initialize Q-matrix with zeros
        self.Q = np.zeros((1+num_steps,2, 2))


        
    def run(self):
        # Initialize state randomly
        state = np.random.randint(0, 2)

        # Q-learning algorithm
        for i in range(self.num_steps):
            logits = self.Q[i][state]/self.temperature
            # Compute softmax probabilities
            probs = np.exp(logits) / np.sum(np.exp(logits))
            action = np.random.choice([0, 1], p=probs)
            next_state = np.random.choice([0, 1], p=self.P[state, action])
            reward = -1 if next_state == state else 1
            self.total_reward+= reward
            if state == 0:
                self.state0_rewards[i] = reward
                self.state1_rewards[i] = None
            else:
                self.state1_rewards[i] = reward
                self.state0_rewards[i] = None
                
            # Update Q-value for current state-action pair
            Q_old = self.Q[i-1][state, action]
            Q_new = Q_old + ((self.alpha_plus if reward - Q_old >= 0 else self.alpha_minus) * (reward-Q_old))
            self.Q[i][state, action] = Q_new

            self.Q[i+1] = self.Q[i]
            state = next_state # update state

    def plot(self):
            # Plot Q values and reward over time
            fig, axs = plt.subplots()
            axs.plot(self.Q[:, 0, 0], label='Q(0,A)')
            axs.plot(self.Q[:, 0, 1], label='Q(0,B)')
            axs.plot(self.Q[:, 1, 0], label='Q(1,A)')
            axs.plot(self.Q[:, 1, 1], label='Q(1,B)')
            
            y_minus = (self.p*self.alpha_plus - (1-self.p)*self.alpha_minus)/(
                    self.p*self.alpha_plus + (1-self.p)* self.alpha_minus)
            y_plus = ((1-self.p)*self.alpha_plus - self.p*self.alpha_minus)/(
                (1-self.p)*self.alpha_plus + self.p*self.alpha_minus)
            axs.axhline(y=y_plus, linestyle='--', color='green', label='Prediction for positive values')
            axs.axhline(y=y_minus, linestyle='--', color='red', label='Prediction for negatve values')
            axs.legend()

            # axs[1].plot(self.state0_rewards, 'o', markersize = 2, label ="Rewards for state 0" )
            # axs[1].plot(self.state1_rewards, 'o', markersize = 2, label = "Rewards for state 1")
            # axs[1].legend()
            plt.show()



def compute_reward(temperature, p, alpha_plus, alpha_minus, num_steps, seed):
    np.random.seed(seed)
    ql = Qlearning_CB(alpha_plus=alpha_plus, alpha_minus=alpha_minus, temperature=temperature, p=p)
    ql.run()
    return ql.total_reward








class ReinforcementLearning_SI:
    def __init__(self, num_actions=10, num_agents=2, num_steps=10000, temperature=.1, alpha_plus=.01, alpha_minus=.01, p=.75, influence = .1):
        def set_parameters(param, num_agents):
            if isinstance(param, (float, int)):
                return [param] * num_agents
            elif len(param) == num_agents:
                return param
            else:
                raise ValueError(f"Length of {param} array must be equal to num_agents.")
        
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.p = p
        self.q_values = np.zeros((num_actions, num_agents))
        self.q_values_history = np.zeros((num_actions, num_agents, num_steps))

        self.alpha_plus = set_parameters(alpha_plus, num_agents)
        self.alpha_minus = set_parameters(alpha_minus, num_agents)
        self.temperature = set_parameters(temperature, num_agents)
        self.influence = set_parameters(influence, num_agents)

        self.ps = np.linspace(p, 1-p, num_actions)
        self.count_songs = np.zeros(num_actions)


        

    def choose_action(self, agent):
        # Softmax decision policy
        exp_q_values = np.exp(self.q_values[:, agent] / self.temperature[agent])
        probs = exp_q_values / np.sum(exp_q_values)
        action = np.random.choice(range(self.num_actions), p=probs)
        return action

    def update_q_values(self, agent, action, reward):
        if reward == 1:
            self.q_values[action, agent] += self.alpha_plus[agent] * (reward - self.q_values[action, agent])
        else:
            self.q_values[action, agent] += self.alpha_minus[agent] * (reward - self.q_values[action, agent])

    def play_song(self):
        """ each agent chose a song then each agent draws another song from the array counting how many time each
            song is chosen, then each agent finally picks a song beteween the 2 previosu pick with probability of picking 
            the song drawn from the count of influence then the total count of each song is updated accordingly, and the rewards are drawn"""
        rewards = np.zeros(self.num_agents)        
        # Each agent chooses a song
        for agent in range(self.num_agents) :
            action = self.choose_action(agent)
            song_probs = softmax(self.count_songs)#/ temperature)
            #influence_song = np.random.choice(range(self.num_actions), p=song_probs)
            influence_song = np.argmax(self.count_songs)
            final_song = np.random.choice([influence_song, action], p=[self.influence[agent], 1 - self.influence[agent]])
            self.count_songs[final_song] += 1
            p_action = self.ps[final_song]
            reward = np.random.choice([1, 0], p=[p_action, 1 - p_action])
            rewards[agent] = reward
            self.update_q_values(agent, final_song, reward)
         # Calculate the final listenings percentage after all agents have made their choices
        total_listenings = np.sum(self.count_songs)
        self.final_listenings_percentage = self.count_songs / total_listenings

    def run_simulation(self):
        for step in tqdm(range(self.num_steps), leave = False, desc = "Step progress"):
            self.play_song()
            self.q_values_history[:, :, step] = self.q_values



    
