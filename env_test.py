import unittest
from parameterized import parameterized
import env
CellType = env.CellType
Action = env.Action

test_map_data = '''\
######
#X   #
# ## #
# #  #
# #  #
#   X#
######
'''


class TestEnv(unittest.TestCase):

  def setUp(self) -> None:
    self.env = env.Environment(test_map_data)

  def test_init(self):
    self.assertEqual(self.env.num_state, 16)
    self.assertEqual(self.env.map_width, 6)
    self.assertEqual(self.env.map_height, 7)

  def test_symbol_to_cell_type(self):
    self.assertEqual(env.symbol_to_cell_type('#'), CellType.WALL)
    self.assertEqual(env.symbol_to_cell_type(' '), CellType.EMPTY_CELL)
    self.assertEqual(env.symbol_to_cell_type('X'), CellType.GOAL_CELL)

  @parameterized.expand([[0, (1, 1)], [1, (1, 2)]])
  def test_state_to_map(self, state, map):
    self.assertEqual(self.env.state_to_map[state], map)

  @parameterized.expand([[1, Action.RIGHT, 2], 
                         [3, Action.RIGHT, 3],
                         [1, Action.LEFT, 0]])
  def test_get_next_state(self, state, action, expected_next_state):
    self.assertEqual(self.env.get_next_state(state, action),
                     expected_next_state)

  def test_get_next_state_for_goal_state(self):
    self.assertEqual(self.env.get_next_state(0, Action.DOWN), 0)

  @parameterized.expand([[1, Action.RIGHT, 2, 1.],
                         [3, Action.RIGHT, 3, 1.],
                         [8, Action.UP, 7, 0.]])
  def test_get_transition_probability(self, state, action, next_state,
                                      expected_probability):
    self.assertEqual(
      self.env.get_transition_probability(state, action, next_state),
      expected_probability)

  @parameterized.expand([[0, Action.RIGHT, 0.], [3, Action.RIGHT, -1.]])
  def test_get_reward(self, state, action, expected_reward):
    self.assertEqual(self.env.get_reward(state, action), expected_reward)


if __name__ == '__main__':
  unittest.main()
