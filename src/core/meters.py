import logging


class AverageEpochMeter(object):
    def __init__(self, name):
        """
        :param name: str, name of the meter
        """

        self.name = name
        self.reset()
        self.sum = 0
        self.avg = 0
        self.count = 0

    def reset(self):
        """
        resets avg, sum and count
        """

        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):

        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
