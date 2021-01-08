from random import uniform, choice, randint
from statistics import mean
from math import log, sqrt
import sys
import seaborn as sns
from math import inf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import colors

### Monte Carlo Tree Search

class Node:
    def __init__(self, **args):
        self.Q = args.get('Q', inf)
        self.N = args.get('N', 0)
        self.value = args.get('value', 0)
        self.left_child = args.get('left_child', None)
        self.right_child = args.get('right_child', None)
        self.parent = args.get('parent', None)
        self.depth = args.get('depth', 0)
        self.leaf = args.get('leaf', False)
        self.visited = args.get('visited', False)
        self.mean_Q = args.get('mean_Q', inf)
        self.ucb = None

class Tree:
    def __init__(self,  **args):
        self.max_depth = args.get('depth',12)
        self.leaf_values = []
        self.root = self.generateTree(Node(depth=0))
        self.resource = args.get('resource', 5)
        self.n_rollout = args.get('n_rollout', 15)
        self.c = args.get('c',2)
        
        
    def calculateUCB(self, node):
        try:
            ucb = float(node.mean_Q + self.c*sqrt(log(node.parent.N/node.N)))
        except ZeroDivisionError:
            ucb = inf
        node.ucb = ucb
        return ucb
    
    def selectNode(self, node):
        #select node that has max ucb value
        #if node already visited, select one of its child recusively
        
        selected = max([node.left_child, node.right_child], key=self.calculateUCB)
        if selected.leaf == True:
            return selected
        if selected.visited == True:
            return(self.selectNode(selected))
        
        return(selected)

    
    def rollout(self, node):
        if node.leaf == True:
            return node
        return self.rollout(choice([node.left_child,node.right_child]))
                        
    def updateNode(self, node, Q):
        if node.Q == inf:
            node.Q = Q
        else:
            node.Q+=Q
        node.N+=1
        node.mean_Q = float(node.Q/node.N)

                        
    def backPropogate(self, node, Q):
        self.updateNode(node,Q)
        if not node.parent:
            return(0)
        self.backPropogate(node.parent, Q)
        return(0)
    
    def chooseNextNode(self, node):
        return max([node.left_child, node.right_child], key=lambda x : x.mean_Q)
    
    def MCTS(self, root):
        flag = True
        next_node = root
        for i in range(self.resource):
            selected = self.selectNode(next_node)
            selected.visited = True
            rollout_values = []
            for n in range(self.n_rollout):
               rollout_values.append(self.rollout(selected).value)
            
            mean_rollout = mean(rollout_values)
            self.backPropogate(selected, mean_rollout)
        next_node = self.chooseNextNode(next_node)
        if next_node.leaf == True:
            return(next_node)
        return self.MCTS(next_node)
            
    def generateTree(self, node):
        if node.depth == self.max_depth:
            node.value = int(uniform(0,100))
            self.leaf_values.append(node.value)
            node.leaf = True
            return node
        current_depth = node.depth
        next_depth = current_depth+1
        node.left_child = self.generateTree(Node(depth = next_depth, parent = node))
        node.right_child = self.generateTree(Node(depth = next_depth, parent = node))
        return node
        
        
#### SARSE and QLearning
class Maze:
    def __init__(self, **args):
        GRID = [
               [0, 0, 0, 0, 0, 0,0,0,0],
               [0, 0, 1, 1, 1, 1,1,0,0],
               [0, 0, 0, 0, 0, 0,1,0,0],
               [0, 0, 0, 0, 0, 0,1,0,0],
               [0, 0, 0, 0, 0, 0,1,0,0],
               [0, 0, 0, 0, 0, 0,1,0,0],
                [0, 0, 0, 0, 0, 2,0,0,0],
                [0, 1, 1, 1, 1, 0,0,0,0],
                [0, 0, 0, 0, 0, 0,0,0,3],
        ]
        self.initial_grid = args.get('grid', GRID)
        self.rewards = args.get('rewards', {0:-1,1:-1,2:-50,3:50 })
        self.object_dict = args.get('object_dict', {'agent': -1,'wall':1, 
                                                    'space': 0 , 'pit': 2, 'goal': 3}
                                   )
        self.state = self.initial_grid
        self.agent_position = args.get('agent_position', [0,0])
        self.agent_move_history = []
        self.verifiyMove(self.agent_position)
        
        self.moves = args.get('moves',{'right': [0,1],
                                        'left': [0,-1],
                                        'up' : [-1,0],
                                        'down' : [1,0]}
                             )
        self.agent_reward = 0
        self.agent_state = 'playing'
        self.agent_steps = 0
        self.agent_reward_map = np.zeros(shape = (9,9))
        self.agent_reward_history = []
        self.data = args.get('data', ['agent_steps', 'agent_state', 'agent_reward',
                                      'agent_reward_map', 'agent_reward_history',
                                      'agent_move_history'
                                     ])
        
    def moveAgent(self, move):
        move = self.moves[move]
        new_pos = [i+j for i,j in zip(move, self.agent_position)]
        block = self.verifiyMove(new_pos)
        return (self.generateReward(block))
        
    def generateReward(self, block):
        reward = self.rewards[block]
        self.agent_reward+=reward
        self.agent_steps+=1
        self.agent_reward_map[self.agent_position[0]][self.agent_position[1]] = self.agent_reward
        self.agent_reward_history.append(reward)
        if reward != -1:
            if reward == 50:
                self.agent_state = 'WON'
            elif reward == -50:
                self.agent_state = 'LOST'

            self.collateData()
                
            self.state[self.agent_position[0]][ self.agent_position[1]] = 0
            
            return(-1)
        return(0)
        
    def collateData(self):
        _dict = {}
        for i in self.data:
            _dict[i] = getattr(self,i)

        self.data=_dict

    def verifiyMove(self, move):
        block = 0
        if( move[0] < 9 and move[0] > -1 ) and ( move[1] < 9 and move[1]> -1 ):
            block = self.state[move[0]][move[1]]

            if block == self.object_dict['space']:

                    self.state[self.agent_position[0]][self.agent_position[1]] = self.object_dict['space']
                    self.state[move[0]][move[1]] = self.object_dict['agent']
                    self.agent_position = move
                    self.agent_move_history.append(self.agent_position)
            
            elif block == self.object_dict['pit'] or block == self.object_dict['goal']:

                    self.state[move[0]][move[1]] = self.object_dict['agent']
                    self.agent_position = move
                    self.agent_move_history.append(self.agent_position)

        return(block)

        
    def displayGrid(self, grid = None):
        if not grid:
            grid = self.state
        cmap = colors.ListedColormap(['yellow','white', 'blue', 'red','green' ])
        bounds = [-1, 0,1,2,3,100]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        ax.imshow(self.state, cmap=cmap, norm=norm)

        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(0.5, 9, 1));
        ax.set_yticks(np.arange(0.5, 9, 1));

        plt.show()
        
class SARSA:
    def __init__(self,**args):
        self.alpha = args.get('alpha',0.3)
        self.gamma = args.get('gamma', 0.9)
        self.n_episodes = args.get('n_episodes', 10)
        self.q_table = args.get('q_table', np.zeros(shape=(9,9,4)))
        self.epsilon = args.get('epsilon', 0.25)
        self.max_steps = args.get('max_steps', 50)
        self.maze=None
        self.current_state = None
        self.action_mapping = {0:'right', 1:'left', 2:'up', 3:'down'}
        self.maze_state = 0
        self.data = []
        
    def printMazeState(self):
        maze = self.maze
        print(f'Agent {maze.agent_state} in {maze.agent_steps} steps, with reward: {maze.agent_reward}')
        
        
    def initMaze(self):
        del(self.maze)
        self.maze = Maze(agent_position=[0,0])
        self.current_state = self.maze.agent_position
        self.maze_state = 0
        
        
    def generateEpsilon(self):
        if uniform(0,1) <= self.epsilon:
            return True
        return False
    
    def generateRandomMove(self):
        return randint(0,3)
    
    def generateGreedyMove(self):
        state = self.current_state
        return np.argmax(self.q_table[state[0]][state[1]])
    
    def generateMoveFromPolicy(self):
        if self.generateEpsilon():
            selected_move = self.generateRandomMove()
        else:
            selected_move = self.generateGreedyMove()
            
        return selected_move
    
    def calculateQValue(self, new_q, old_q, reward):
        return ((1-self.alpha)*(old_q) + self.alpha*(reward + self.gamma*(new_q)-old_q))
    
    def updateQValue(self, reward, old_state, new_state, old_move, next_move):
        old_q = self.q_table[old_state[0]][old_state[1]][old_move]
        new_q = self.q_table[new_state[0]][new_state[1]][next_move]
        updated_q = self.calculateQValue(new_q,old_q,reward)
        self.q_table[old_state[0]][old_state[1]][old_move] = updated_q
        
    def moveAgent(self, selected_move):
        old_state = self.current_state

        
        self.maze_state = self.maze.moveAgent(self.action_mapping[selected_move])
        self.current_state = self.maze.agent_position
        reward = self.maze.agent_reward_history[-1]
        new_state = self.current_state
        old_move = selected_move

        return(reward, old_state, new_state, old_move)
        
        
    def train(self):
        
        for _ in range(self.n_episodes):
            
            self.initMaze()
            next_move = self.generateRandomMove()

            print(f'Episode {_}: ', end = '')
            for i in range(self.max_steps):
                if self.maze_state == -1:
                    self.printMazeState()
                    self.data.append(self.maze.data)
                    break
                

                reward, old_state, new_state, old_move = self.moveAgent(next_move)
                next_move = self.generateMoveFromPolicy()
                self.updateQValue(reward, old_state, new_state, old_move, next_move)
                
            if self.maze_state != -1:
                print('Did not find end state')
                self.maze.collateData()
                self.data.append(self.maze.data)
        self.data = pd.DataFrame(self.data)
        
class QLearning:
    def __init__(self,**args):
        self.alpha = args.get('alpha',0.3)
        self.gamma = args.get('gamme', 0.9)
        self.n_episodes = args.get('n_episodes', 10)
        self.q_table = args.get('q_table', np.zeros(shape=(9,9,4)))
        self.epsilon = args.get('epsilon', 0.25)
        self.max_steps = args.get('max_steps', 50)
        self.maze=None
        self.current_state = None
        self.action_mapping = {0:'right', 1:'left', 2:'up', 3:'down'}
        self.maze_state = 0
        self.data = []
        
    def printMazeState(self):
        maze = self.maze
        print(f'Agent {maze.agent_state} in {maze.agent_steps} steps, with reward: {maze.agent_reward}')
        
        
    def initMaze(self):
        del(self.maze)
        self.maze = Maze(agent_position=[0,0])
        self.current_state = self.maze.agent_position
        self.maze_state = 0
        
    def generateEpsilon(self):
        if uniform(0,1) <= self.epsilon:
            return True
        return False
    
    def generateRandomMove(self):
        return randint(0,3)
    
    def generateGreedyMove(self):
        state = self.current_state
        return np.argmax(self.q_table[state[0]][state[1]])
    
    def generateMoveFromPolicy(self):
        if self.generateEpsilon():
            selected_move = self.generateRandomMove()
        else:
            selected_move = self.generateGreedyMove()
            
        return selected_move
    
    def calculateQValue(self, max_q, old_q, reward):
        return ((1-self.alpha)*(old_q) + self.alpha*(reward + self.gamma*(max_q)-old_q))
    
    def moveAgent(self, selected_move):
        old_state = self.current_state
        
        
        self.maze_state = self.maze.moveAgent(self.action_mapping[selected_move])
        self.current_state = self.maze.agent_position
        
        next_state = self.current_state
        reward = self.maze.agent_reward_history[-1]
        old_q = self.q_table[old_state[0]][old_state[1]][selected_move]
        max_q = max(self.q_table[next_state[0]][next_state[1]])
        update_q = self.calculateQValue(max_q,old_q,reward)
        
        self.q_table[old_state[0]][old_state[1]][selected_move] = update_q
        
        
    def train(self):
        
        for _ in range(self.n_episodes):
            
            self.initMaze()
            first_move = self.generateRandomMove()
            self.moveAgent(first_move)
            print(f'Episode {_}: ', end = '')
            for i in range(self.max_steps):
                if self.maze_state == -1:
                    self.data.append(self.maze.data)
                    self.printMazeState()
                    break
                next_move = self.generateMoveFromPolicy()
                self.moveAgent(next_move)
                
            if self.maze_state != -1:
                print('Did not find end state')
                self.maze.collateData()
                self.data.append(self.maze.data)
        self.data=pd.DataFrame(self.data)
