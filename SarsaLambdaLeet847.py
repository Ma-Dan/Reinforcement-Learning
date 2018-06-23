import random as rnd
import copy
import sys

class Env(object):
    def __init__(self):
        self.Connection = []
        self.Visited = []
        self.NodeIndex = -1

    def setConnection(self, con):
        self.Connection = con

    def reset(self):
        self.Visited = []
        for node in self.Connection:
            self.Visited.append(0)
        self.NodeIndex = -1
        return self.NodeIndex

    def actionSpace(self, state):
        node = state[len(state)-1]
        if -1 == node:
            space = range(0, len(self.Connection))
        else:
            space = self.Connection[node]

        return space

    def actionSample(self, state):
        node = state[len(state)-1]
        if -1 == node:
            Sample = rnd.randint(0, len(self.Connection)-1)
        else:
            Sample = self.Connection[node][rnd.randint(0, len(self.Connection[node])-1)]
        return Sample

    def isDone(self):
        is_done = True
        for visited in self.Visited:
            if 0 == visited:
                is_done = False
                break
        return is_done

    def step(self, state, action):
        state.append(action)

        self.NodeIndex = action
        reward = -10 * (self.Visited[self.NodeIndex] + 1) * (self.Visited[self.NodeIndex] + 1)
        self.Visited[self.NodeIndex] += 1

        is_done = self.isDone()
        if is_done:
            reward = 1.0e8

        return state, reward, is_done


class SarsaLambdaAgent(object):
    def __init__(self, env:Env):
        self.env = env
        self.Q = {}
        self.E = {}
        self.NodeHistory = []
        self.initAgent()

    def isStateInQ(self, state):
        statetuple = tuple(state)
        return self.Q.get(statetuple) is not None

    def initStateValue(self, state, randomized=True):
        statetuple = tuple(state)
        if not self.isStateInQ(statetuple):
            self.Q[statetuple] = {}
            for action in self.env.actionSpace(state):
                default_v = rnd.random() / 10 if randomized is True else 0.0
                self.Q[statetuple][action] = default_v

    def assertStateInQ(self, state, randomized=True):
        # 　cann't find the state
        if not self.isStateInQ(state):
            self.initStateValue(state, randomized)

    def getQ(self, state, action):
        self.assertStateInQ(state, randomized=False)
        statetuple = tuple(state)
        return self.Q[statetuple][action]

    def setQ(self, state, action, value):
        statetuple = tuple(state)
        self.Q[statetuple][action] = value

    def isStateInE(self, state):
        statetuple = tuple(state)
        return self.E.get(statetuple) is not None

    def initStateE(self, state):
        statetuple = tuple(state)
        if not self.isStateInE(statetuple):
            self.E[statetuple] = {}
            for action in self.env.actionSpace(state):
                self.E[statetuple][action] = 0.0

    def assertStateInE(self, state):
        # 　cann't find the state
        if not self.isStateInE(state):
            self.initStateE(state)

    def getE(self, state, action):
        self.assertStateInE(state)
        statetuple = tuple(state)
        return self.E[statetuple][action]

    def setE(self, state, action, value):
        statetuple = tuple(state)
        self.E[statetuple][action] = value

    def resetEValue(self):
        self.E = {}

    def initAgent(self):
        self.env.reset()
        self.NodeHistory = [-1]
        self.initStateValue(self.NodeHistory, randomized=False)

    # using simple decaying epsilon greedy exploration
    def curPolicy(self, state, episode_num, use_epsilon):
        epsilon = 1.00 / (episode_num+1)
        rand_value = rnd.random()
        action = None

        if use_epsilon and rand_value < epsilon:
            action = self.env.actionSample(state)
        else:
            self.assertStateInQ(state, randomized=True)
            Q_s = self.Q[tuple(state)]
            action = max(Q_s, key=Q_s.get)

        return action

    # Agent依据当前策略和状态决定下一步的动作
    def performPolicy(self, state, episode_num, use_epsilon=True):
        return self.curPolicy(state, episode_num, use_epsilon)

    def act(self, state, action):
        return self.env.step(state, action)

    # SARSA learning
    def learning(self, lambda_, gamma, alpha, max_episode_num):
        total_time, time_in_episode, num_episode = 0, 0, 0

        last_path = []

        while num_episode < max_episode_num:
            self.CurrentNode = self.env.reset()
            self.resetEValue()
            self.NodeHistory = [-1]

            s0 = copy.deepcopy(self.NodeHistory)
            a0 = self.performPolicy(s0, num_episode)

            time_in_episode = 0
            is_done = False
            while not is_done:
                print(a0, end="->")
                s1, r1, is_done = self.act(self.NodeHistory, a0)

                a1 = self.performPolicy(s1, num_episode)

                q = self.getQ(s0, a0)
                q_prime = self.getQ(s1, a1)
                delta = r1 + gamma * q_prime - q

                e = self.getE(s0, a0)
                e = e + 1
                self.setE(s0, a0, e) # set E before update E

                state_action_list = list(zip(self.E.keys(), self.E.values()))
                for s, a_es in state_action_list:
                    for a in self.env.actionSpace(s):
                        e_value = a_es[a]
                        old_q = self.getQ(s, a)
                        new_q = old_q + alpha * delta * e_value
                        new_e = gamma * lambda_ * e_value
                        self.setQ(s, a, new_q)
                        self.setE(s, a, new_e)

                s0 = copy.deepcopy(s1)
                a0 = a1
                time_in_episode += 1

            print("\nEpisode {0} takes {1} steps.".format(
                num_episode, time_in_episode))
            #print(self.Q)
            total_time += time_in_episode
            num_episode += 1
        return


def main(input):
    env = Env()
    env.setConnection(eval(input))
    agent = SarsaLambdaAgent(env)
    env.reset()

    print("Learning...")
    agent.learning(lambda_ = 0.1,
                   gamma=1.0,
                   alpha=0.1,
                   max_episode_num=5000)


if __name__ == "__main__":
    main(sys.argv[1])
