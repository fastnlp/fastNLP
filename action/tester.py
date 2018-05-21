from action.action import Action


class Tester(Action):
    """docstring for Tester"""

    def __init__(self, arg):
        super(Tester, self).__init__()
        self.arg = arg
