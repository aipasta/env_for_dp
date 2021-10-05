from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, IntEnum

map_data_1 = '''\
######
#X   #
#    #
#    #
#   X#
######
'''
map_data_2 = '''\
###################
#                X#
#   ###########   #
#   #         #   #
#   # ####### #   #
#   # #     # #   #
#     #  #  #     #
#        #        #
#                 #
###             ###
#                 #
###################
'''


class Action(IntEnum):
  LEFT = 0
  DOWN = 1
  RIGHT = 2
  UP = 3


class CellType(Enum):
  WALL = 1
  EMPTY_CELL = 2
  GOAL_CELL = 3


def symbol_to_cell_type(symbol):
  return {'#': CellType.WALL,
          ' ': CellType.EMPTY_CELL,
          'X': CellType.GOAL_CELL}[symbol]


class Environment():
  """Class for grid world environment

  Attributes:
    map: The map consists of `cell_type`.
    states: States the environment has. States are represented as an int (i.e.,
      [0, num_state-1]).
    state_to_map: The mapping between state and map (e.g, state -> (row, col)).
    map_to_state: The mapping between map and state.
      (e.g, map_to_state[row][col] -> state).
    num_state: Size of state space.
    num_action: Size of action space.
    map_width: The width of map which includes cell and wall.
    map_height: The height of map which includes cell and wall.
  """

  def __init__(self, map_data: str):
    self.map = self._load_map_data(map_data)
    self.state_to_map, self.map_to_state = self._get_mapping_state_and_map()
    self.states = list(range(self.num_state))
    self.num_action = 4  # LEFT, DOWN, RIGHT, UP
    self.actions = list(range(self.num_action))
    self.reward_matrix = self.get_reward_matrix()
    self.transition_probability_matrix = (
      self.get_transition_probability_matrix())

  @property
  def num_state(self):
    return len(self.state_to_map)

  @property
  def map_width(self):
    return len(self.map[0])

  @property
  def map_height(self):
    return len(self.map)

  def _load_map_data(self, map_data: str) -> List:
    map = []
    for line in map_data.splitlines():
      cell_row = []
      for c in line:
        cell_row.append(symbol_to_cell_type(c))
      map.append(cell_row)
    return map

  def _get_mapping_state_and_map(self) -> Tuple[List, List]:
    state_to_map = []
    map_to_state = [[None]*self.map_width for _ in range(self.map_height)]

    for row in range(self.map_height):
      for col in range(self.map_width):
        if not self.map[row][col] == CellType.WALL:
          state_to_map.append((row, col))
          map_to_state[row][col] = len(state_to_map)-1

    return state_to_map, map_to_state

  def _get_cell_by_state(self, state: int):
    row, col = self.state_to_map[state]
    return self.map[row][col]

  def get_next_state(self, state: int, action: int) -> int:
    row, col = self.state_to_map[state]

    # If the agent achieves the goal, the episode terminates. So there is no
    # next state.
    if self.map[row][col] == CellType.GOAL_CELL:
      return state

    if action == Action.LEFT:
      col = col - 1
    elif action == Action.DOWN:
      row = row + 1
    elif action == Action.RIGHT:
      col = col + 1
    elif action == Action.UP:
      row = row - 1
    # If the agent is blocked by wall, it remain at the same state.
    if self.map[row][col] == CellType.WALL:
      return state
    else:
      return self.map_to_state[row][col]

  def get_transition_probability(self, state: int, action: int,
                                 next_state: int) -> float:
    if self.get_next_state(state, action) == next_state:
      return 1.0
    else:
      return 0.0

  def get_reward(self, state: int, action: int):
    """Return the reward for given sate and action.

    The reward for goal state is 0 and otherwise -1.
    """
    del action  # Not used in this environment.
    if self._get_cell_by_state(state) == CellType.GOAL_CELL:
      return 0
    else:
      return -1.0

  def get_reward_matrix(self):
    reward_matrix = np.zeros([self.num_state, self.num_action])
    for s in self.states:
      for a in self.actions:
        reward_matrix[s, a] = self.get_reward(s, a)
    return reward_matrix

  def get_transition_probability_matrix(self):
    transistioin_probability_matrix = np.zeros(
      [self.num_state, self.num_action, self.num_state])
    for s in self.states:
      for a in self.actions:
        for s_n in self.states:
          transistioin_probability_matrix[s, a, s_n] = (
            self.get_transition_probability(s, a, s_n))
    return transistioin_probability_matrix

  def value_function_with_structure(self,
                                    value_function: np.ndarray) -> np.ndarray:
    arr = np.empty((self.map_height-2, self.map_width-2)) * np.nan
    for state in range(len(value_function)):
      row, col = self.state_to_map[state]
      arr[row-1, col-1] = value_function[state]
    return arr

  def pretty_print_value_function(self, value_function: np.ndarray):
    arr = self.value_function_with_structure(value_function)

    with np.printoptions(precision=2, suppress=True):
      print(np.reshape(np.array(arr), (self.map_height-2, self.map_width-2)))

  def plot_value_function(self, value_function):
    arr = self.value_function_with_structure(value_function)
    fig, ax = plt.subplots()
    im = ax.imshow(np.reshape(arr, (-1, self.map_width-2)))
    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02,
                        ax.get_position().height])
    fig.colorbar(im, cax=cax)
    plt.show()


if __name__ == '__main__':

  env = Environment(map_data_2)
  env.get_next_state(10, Action.DOWN)
  v = [0] * env.num_state
  env.pretty_print_value_function(v)
  # env.plot_value_function(v)
  env = Environment(map_data_1)
  print(env.transition_probability_matrix.shape)
  print(env.reward_matrix.shape)