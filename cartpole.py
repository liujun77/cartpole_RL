import gym
import numpy as np
import tensorflow as tf
#%%
env = gym.make('CartPole-v0') 


#%%

# policy gradient

class actor:
    def __init__(self, env, discount=0.9, learning_rate=0.01):
        self.env = env
        self.discount = discount
        self.learning_rate = learning_rate
        self.action_dim = env.action_space.n
        self.observation_dim = env.observation_space.shape[0]
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(321)
            self.weight = tf.Variable(tf.random_normal((self.observation_dim, self.action_dim)))
            self.bias = tf.Variable(tf.random_normal((self.action_dim)))
            
            self.x = tf.placeholder("float", [None, self.observation_dim])
            self.y = tf.placeholder("float") #advantage
            self.action_input = tf.placeholder("float", [None, self.action_dim])
            
            self.policy = tf.nn.softmax(tf.mul(self.x, self.weight)+self.bias)
            self.log_act_prob = tf.reduce_sum(self.action_input * tf.log(self.policy))
            
            self.loss = - self.log_act_prob * self.y
            
            self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.init = tf.initialize_all_variables()
        
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def choose_action(self, state):
        state = np.asarray(state)
        state = state.reshape((1, self.observation_dim))
        softmax = self.sess.run(self.policy, feed_dict={self.x: state})
        action = np.random.choice([0, 1], 1, replace = True, p = softmax[0])[0]
        return action
        
    def evolution_policy(self, time_steps):
        total_reward = 0
        curr_state = self.env.reset()
        epsoid_state = []
        epsoid_reward = []
        epsoid_action = []
        epsoid_next_states = []
        epsoid_Q = []
        
        for time in xrange(time_steps):
            action = self.choose_action(curr_state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            if done or time> self.env.spec.timestep_limit:
                break;
            
            curr_state_l = curr_state.tolist()
            next_state_l = next_state.tolist()
            
            if curr_state_l not in epsoid_state:
                epsoid_state.append(curr_state_l)
                epsoid_action.append(action)
                epsoid_reward.append(reward)
                epsoid_next_states.append(next_state_l)
                epsoid_Q.append(reward)
                for i in xrange(len(epsoid_action)-1):
                    epsoid_Q[i] += np.power(self.discount, len(epsoid_action)-1-i)*reward
            else:
                for i in xrange(len(epsoid_action)):
                    epsoid_Q[i] += np.power(self.discount, len(epsoid_action)-i)*reward
            curr_state = next_state
        
        update_memory(self, epsoid_state, 
                      epsoid_action, epsoid_next_states, epsoid_reward, 
                      epsoid_Q)
        return total_reward, epsoid_state, epsoid_action, epsoid_next_states, epsoid_reward, epsoid_Q
    
    def update_memory(self, epsoid_state, 
                      epsoid_action, epsoid_next_state, epsoid_reward,
                      epsoid_Q):
        
            
        
        
class agent:
    def __init__(self):
        self.weight = np.random.rand(4)
        self.lr = 0.0001
        
    def choose_action(self, observation):
        p = 1/(np.exp(-np.dot(observation, self.weight))+1)
        if p>=0.5:
            return p, 1
        else:
            return p, 0
    
    def evaluate(self):
        total_reward = 0
        observation = env.reset()
        while 1:
            p, action = self.choose_action(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward
            
    def mc_policy_gradient(self):
        for i in xrange(200):
            observation = env.reset()
            while 1:
                p, action = self.choose_action(observation)
                observation, reward, done, info = env.step(action)
                self.weight -= self.lr * observation * (p-1) * reward
                if done:
                    break
        print 'i = ', i, 'total_reward = ', self.evaluate()

#%%

ag = agent()

ag.mc_policy_gradient()
            
                
        
        