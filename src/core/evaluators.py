import numpy as np
from sklearn.metrics import r2_score, accuracy_score, f1_score, balanced_accuracy_score


class R2Evaluator(object):
    """
    R2 score evaluator

    Attributes
    ----------
    r2_score: float, the r2 score
    y_pred: numpy.ndarray, numpy array containing all predictions
    y_true: numpy.ndarray, numpy array containing all ground truth labels

    Methods
    -------
    reset(): resets r2_score, y_pred and y_true
    update(y_pred, y_true): updates the y_pred and y_true labels with input predictions and labels
    compute(): computes the r2 score using the information in y_pred and y_true
    """

    def __init__(self):

        self.r2_score = 0.0
        self.y_pred = np.array([])
        self.y_true = np.array([])

    def reset(self):
        """
        reset r2_score, y_pred and y_true
        """

        self.r2_score = 0.0
        self.y_pred = np.array([])
        self.y_true = np.array([])

    def update(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Update the predictions and labels arrays

        :param y_pred: numpy.ndarray, predictions to add to predictions array
        :param y_true: numpy.ndarray, predictions to add to labels array
        """

        try:
            # Append to y_preds
            self.y_pred = (
                np.concatenate((self.y_pred, y_pred), axis=0)
                if self.y_pred.size
                else y_pred
            )
        except ValueError:
            self.y_pred = np.hstack((self.y_pred, y_pred))

        try:
            # Append to y_true
            self.y_true = (
                np.concatenate((self.y_true, y_true), axis=0)
                if self.y_true.size
                else y_true
            )
        except ValueError:
            self.y_true = np.hstack((self.y_true, y_true))

    def compute(self) -> float:
        """
        Computes the r2 score using the internal predictions and labels arrays
        :return: r2 score
        """

        self.r2_score = r2_score(y_true=self.y_true, y_pred=self.y_pred)

        return self.r2_score


class MAEEvaluator(object):
    def __init__(self):

        self.mae = 0.0
        self.y_pred = np.array([])
        self.y_true = np.array([])

    def reset(self):
        """
        reset r2_score, y_pred and y_true
        """

        self.mae = 0.0
        self.y_pred = np.array([])
        self.y_true = np.array([])

    def update(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Update the predictions and labels arrays

        :param y_pred: numpy.ndarray, predictions to add to predictions array
        :param y_true: numpy.ndarray, predictions to add to labels array
        """

        try:
            # Append to y_preds
            self.y_pred = (
                np.concatenate((self.y_pred, y_pred), axis=0)
                if self.y_pred.size
                else y_pred
            )
        except ValueError:
            self.y_pred = np.hstack((self.y_pred, y_pred))

        try:
            # Append to y_true
            self.y_true = (
                np.concatenate((self.y_true, y_true), axis=0)
                if self.y_true.size
                else y_true
            )
        except ValueError:
            self.y_true = np.hstack((self.y_true, y_true))

    def compute(self) -> float:
        """
        Computes the r2 score using the internal predictions and labels arrays
        :return: r2 score
        """

        self.mae = np.mean(np.abs(self.y_true - self.y_pred.flatten())).item()

        return self.mae


class F1ScoreEvaluator(object):
    """
    F1 score evaluator

    Attributes
    ----------
    score: float, the F1 score
    y_pred: numpy.ndarray, numpy array containing all predictions
    y_true: numpy.ndarray, numpy array containing all ground truth labels
    threshold: float, the threshold used to binarize y_pred

    Methods
    -------
    reset(): resets score, y_pred and y_true
    update(y_pred, y_true): updates the y_pred and y_true labels with input predictions and labels
    compute(): computes the accuracy using the information in y_pred and y_true
    """

    def __init__(self, threshold: float = 0.4):
        """
        :param threshold: float, the threshold used to binarize y_pred
        """

        self.score = 0.0
        self.y_pred = np.array([])
        self.y_true = np.array([])
        self.threshold = threshold

    def reset(self):
        """
        reset score, y_pred and y_true
        """

        self.score = 0.0
        self.y_pred = np.array([])
        self.y_true = np.array([])

    def update(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Update the predictions and labels arrays

        :param y_pred: numpy.ndarray, predictions to add to predictions array
        :param y_true: numpy.ndarray, predictions to add to labels array
        """

        try:
            # Append to y_preds
            self.y_pred = (
                np.concatenate((self.y_pred, y_pred), axis=0)
                if self.y_pred.size
                else y_pred
            )
        except ValueError:
            self.y_pred = np.hstack((self.y_pred, y_pred))

        try:
            # Append to y_true
            self.y_true = (
                np.concatenate((self.y_true, y_true), axis=0)
                if self.y_true.size
                else y_true
            )
        except ValueError:
            self.y_true = np.hstack((self.y_true, y_true))

    def compute(self) -> float:
        """
        Computes the F1 score using the internal predictions and labels arrays
        :return: F1 score
        """

        self.score = f1_score(
            y_true=self.y_true < self.threshold, y_pred=self.y_pred < self.threshold
        )

        return self.score


class AccuracyEvaluator(object):
    def __init__(self):

        self.accuracy = 0.0
        self.y_pred = np.array([])
        self.y_true = np.array([])

    def reset(self):

        self.accuracy = 0.0
        self.y_pred = np.array([])
        self.y_true = np.array([])

    def update(self, y_pred: np.ndarray, y_true: np.ndarray):

        y_pred = np.argmax(y_pred, axis=1)

        try:
            # Append to y_preds
            self.y_pred = (
                np.concatenate((self.y_pred, y_pred), axis=0)
                if self.y_pred.size
                else y_pred
            )
        except ValueError:
            self.y_pred = np.hstack((self.y_pred, y_pred))

        try:
            # Append to y_true
            self.y_true = (
                np.concatenate((self.y_true, y_true), axis=0)
                if self.y_true.size
                else y_true
            )
        except ValueError:
            self.y_true = np.hstack((self.y_true, y_true))

    def compute(self) -> float:
        """
        Computes the r2 score using the internal predictions and labels arrays
        :return: r2 score
        """

        self.accuracy = accuracy_score(y_true=self.y_true, y_pred=self.y_pred)

        return self.accuracy


class BalancedAccuracyEvaluator(object):
    def __init__(self):

        self.accuracy = 0.0
        self.y_pred = np.array([])
        self.y_true = np.array([])

    def reset(self):

        self.accuracy = 0.0
        self.y_pred = np.array([])
        self.y_true = np.array([])

    def update(self, y_pred: np.ndarray, y_true: np.ndarray):

        y_pred = np.argmax(y_pred, axis=1)

        try:
            # Append to y_preds
            self.y_pred = (
                np.concatenate((self.y_pred, y_pred), axis=0)
                if self.y_pred.size
                else y_pred
            )
        except ValueError:
            self.y_pred = np.hstack((self.y_pred, y_pred))

        try:
            # Append to y_true
            self.y_true = (
                np.concatenate((self.y_true, y_true), axis=0)
                if self.y_true.size
                else y_true
            )
        except ValueError:
            self.y_true = np.hstack((self.y_true, y_true))

    def compute(self) -> float:
        """
        Computes the r2 score using the internal predictions and labels arrays
        :return: r2 score
        """

        self.accuracy = balanced_accuracy_score(y_true=self.y_true, y_pred=self.y_pred)

        return self.accuracy


class BalancedBinaryAccuracyEvaluator(object):
    def __init__(self):

        self.accuracy = 0.0
        self.y_pred = np.array([])
        self.y_true = np.array([])

    def reset(self):

        self.accuracy = 0.0
        self.y_pred = np.array([])
        self.y_true = np.array([])

    def update(self, y_pred: np.ndarray, y_true: np.ndarray):

        y_pred = y_pred > 0.5

        try:
            # Append to y_preds
            self.y_pred = (
                np.concatenate((self.y_pred, y_pred), axis=0)
                if self.y_pred.size
                else y_pred
            )
        except ValueError:
            self.y_pred = np.hstack((self.y_pred, y_pred))

        try:
            # Append to y_true
            self.y_true = (
                np.concatenate((self.y_true, y_true), axis=0)
                if self.y_true.size
                else y_true
            )
        except ValueError:
            self.y_true = np.hstack((self.y_true, y_true))

    def compute(self) -> float:
        """
        Computes the r2 score using the internal predictions and labels arrays
        :return: r2 score
        """

        self.accuracy = balanced_accuracy_score(y_true=self.y_true, y_pred=self.y_pred)

        return self.accuracy
