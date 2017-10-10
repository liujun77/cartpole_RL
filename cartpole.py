import gym
import numpy as np
import tensorflow as tf
#%%


replay_states = []
replay_actions = []
replay_rewards = []
replay_next_states = []
replay_Q = []
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
            self.weight = tf.Variable(tf.random_normal([self.observation_dim, self.action_dim]))
            self.bias = tf.Variable(tf.random_normal([self.action_dim]))
            
            self.x = tf.placeholder("float", [None, self.observation_dim])
            self.y = tf.placeholder("float") #advantage
            self.action_input = tf.placeholder("float", [None, self.action_dim])
            
            self.policy = tf.nn.softmax(tf.matmul(self.x, self.weight)+self.bias)
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
            next_state, reward, done, info = self.env.step(action)
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
        
        self.update_memory(epsoid_state, 
                      epsoid_action, epsoid_next_states, epsoid_reward, 
                      epsoid_Q)
        return total_reward, epsoid_state, epsoid_action, epsoid_next_states, epsoid_reward, epsoid_Q
    
    def update_memory(self, epsoid_state, 
                      epsoid_action, epsoid_next_state, epsoid_reward,
                      epsoid_Q):
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_Q
        replay_states.append(epsoid_state)
        replay_actions.append(epsoid_action)
        replay_rewards.append(epsoid_reward)
        replay_next_states.append(epsoid_next_state)
        replay_Q.append(epsoid_Q)
    
    def reset_memory(self):
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_Q
        del replay_states[:], replay_actions[:], replay_rewards[:], replay_next_states[:], replay_Q[:]
            
    def to_action_input(self, action):
        action_input = [0]* self.action_dim
        action_input[action] = 1
        action_input = np.asarray(action_input)
        action_input = action_input.reshape(1, self.action_dim)
        return action_input
    
    def update_policy(self, advantage_vector):
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_Q
        for i in xrange(len(replay_actions)):
            states = replay_states[i]
            actions = replay_actions[i]
            advantage = advantage_vector[i]
            for j in xrange(len(states)):
                action = self.to_action_input(actions[j])
                state = np.asarray(states[j])
                state = state.reshape(1, self.observation_dim)
                _, err = self.sess.run([self.optim, self.loss], 
                                       feed_dict={self.x:state, self.action_input:action, self.y:advantage[j]})

class critic:
    def __init__(self, env, discount = 0.9, learning_rate = 0.008):
        self.env = env
        self.discount = discount
        self.learning_rate = learning_rate
        self.action_dim = env.action_space.n
        self.observation_dim = env.observation_space.shape[0]
        self.n_hidden_1 = 20
        self.n_epochs = 20
        self.batch_size = 32
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            tf.set_random_seed(312)
            self.weights = {'h1': tf.Variable(tf.random_normal([self.observation_dim, self.n_hidden_1])),
                            'out':tf.Variable(tf.random_normal([self.n_hidden_1, 1])) }
            self.bias = {'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                         'out':tf.Variable(tf.random_normal([1]))}
            self.state_input = tf.placeholder('float', [None, self.observation_dim])
            self.Q_input = tf.placeholder('float')
            self.value_pred = self.mlp(self.state_input, self.weights, self.bias)
            self.loss = tf.reduce_mean(tf.pow(self.Q_input-self.value_pred, 2))
            self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            init = tf.initialize_all_variables()
        
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(init)
        
    def mlp(self, x, weights, bias):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), bias['b1'])
        layer_1 = tf.nn.tanh(layer_1)
        out_layer = tf.matmul(layer_1, weights['out']) + bias['out']
        return out_layer

    def update_value_pred(self):
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_Q
        batch_size = self.batch_size
        if np.ma.size(replay_states)<batch_size:
            batch_size = np.ma.size(replay_states)
        
        for epoch in xrange(self.n_epochs):
            total_batch = np.ma.size(replay_states)/batch_size
            for i in xrange(total_batch):
                batch_state_input, batch_Q_input = self.next_batch(batch_size, 
                                                                        replay_states,
                                                                        replay_Q)
                self.sess.run(self.optim, feed_dict={self.state_input:batch_state_input, 
                                                     self.Q_input:batch_Q_input})
        
    def get_advantage_vector(self, states, rewards, next_states):
        advantage_vector = []
        for i in xrange(len(states)):
            state = np.asarray(states[i])
            state = state.reshape(1,self.observation_dim)
            next_state = np.asarray(next_states[i])
            next_state = next_state.reshape(1,self.observation_dim)
            reward = rewards[i]
            state_value = self.sess.run(self.value_pred, feed_dict={self.state_input:state})
            next_state_value = self.sess.run(self.value_pred, feed_dict={self.state_input:next_state})
            #This follows directly from the forula for TD(0)
            advantage = reward + self.discount*next_state_value - state_value
            advantage_vector.append(advantage)

        return advantage_vector
        
    def next_batch(self, batch_size, states_data, Q_data):
        all_states = []
        all_Qs = []
        for i in xrange(len(states_data)):
            episode_states = states_data[i]
            episode_Qs = Q_data[i]
            for j in xrange(len(episode_states)):
                all_states.append(episode_states[j])
                all_Qs.append(episode_Qs[j])
        all_states = np.asarray(all_states)
        all_Qs = np.asarray(all_Qs)
        randidx = np.random.randint(all_states.shape[0], size=batch_size)
        batch_states = all_states[randidx, :]
        batch_Qs = all_Qs[randidx]
        return batch_states, batch_Qs
        
class actor_critic:
    def __init__(self, env, max_episods, episod_before_update, discount):
        self.env = env
        self.actor1 = actor(env, discount)
        self.critic1 = critic(env, discount)
        
        self.max_episods = max_episods
        self.episods_before_update = episod_before_update
    
    def learn(self):
        advantage_vectors = []
        sum_reward = 0
        update = False
        for i in xrange(self.max_episods):
            episode_total_reward, episode_states, episode_actions, episode_next_states, episode_rewards, episode_Q = self.actor1.evolution_policy(200)
            advantage_vector = self.critic1.get_advantage_vector(episode_states, episode_rewards, episode_next_states)
            advantage_vectors.append(advantage_vector)
            sum_reward += episode_total_reward
            if (i+1)%self.episods_before_update==0:
                avg_reward = sum_reward/self.episods_before_update
                print "Current {} episode average reward: {}".format(self.episods_before_update, avg_reward)
                if avg_reward>195:
                    update = False
                else:
                    update = True
                if update:
                    print "Updating"
                    self.actor1.update_policy(advantage_vectors)
                    self.critic1.update_value_pred()
                else:
                    print "Good Solution, not updating"
                
                del advantage_vectors[:]
                self.actor1.reset_memory()
                sum_reward = 0

def main():
    env = gym.make('CartPole-v0')
    env.seed(1234)
    max_episodes = 3000
    episodes_before_update = 2
    discount = 0.85
    
    ac_learner = actor_critic(env, max_episodes, episodes_before_update, discount)
    ac_learner.learn()
    
if __name__=="__main__":
	main()