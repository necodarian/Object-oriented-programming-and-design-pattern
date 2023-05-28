import enum
from abc import ABC, abstractmethod

import numpy as np


class ImputerStrategy(ABC):
    """Imputer strategy user interface"""
    def __init__(self, axis: int) -> None:
        self.axis = axis

    """Each class will provide its implementation using these methods bellow"""
    @abstractmethod
    def fit(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def transform(self, data: np.ndarray, fitted_data: np.ndarray) -> np.ndarray:
        pass


"""These classes implement the calculation of the required tasks which is transform and fit"""
class Mean(ImputerStrategy):
    """"Concrete Mean strategy"""
    def __init__(self, axis: int = 0) -> None:
        super(Mean, self).__init__(axis)

    def fit(self, data: np.ndarray) -> np.ndarray:
        if self.axis == 0:
            return np.nanmean(data, axis=self.axis)
        else:
            print(f"`fit` method for axis={self.axis} is not implemented.")

    def transform(self, data: np.ndarray, fitted_data: np.ndarray) -> np.ndarray:
        if self.axis == 0:
            for i in range(data.shape[1]):
                d = data[:, i]
                data[:, i] = np.nan_to_num(d, nan=fitted_data[i])
            return data
        else:
            print(f"`transform` method for axis={self.axis} is not implemented.")


class Median(ImputerStrategy):
    """Concrete Median strategy"""
    def __init__(self, axis: int = 0) -> None:
        super(Median, self).__init__(axis=axis)

    def fit(self, data: np.ndarray) -> np.ndarray:
        if self.axis == 0:
            return np.nanmedian(data, axis=self.axis)
        else:
            print(f"`fit` method for axis={self.axis} is not implemented.")

    def transform(self, data: np.ndarray, fitted_data: np.ndarray) -> np.ndarray:
        if self.axis == 0:
            for i in range(data.shape[1]):
                d = data[:, i]
                data[:, i] = np.nan_to_num(d, nan=fitted_data[i])
            return data


class Mode(ImputerStrategy):
    """Concrete Mode strategy"""
    def __init__(self, axis: int = 0) -> None:
        super(Mode, self).__init__(axis=axis)

    def fit(self, data: np.ndarray) -> np.ndarray:
        if self.axis == 0:
            u, c = np.unique(data, axis=self.axis, return_counts=True)
            return u[c.argmax()]
        else:
            print(f"`fit` method for axis={self.axis} is not implemented.")

    def transform(self, data: np.ndarray, fitted_data: np.ndarray) -> np.ndarray:
        if self.axis == 0:
            for i in range(data.shape[1]):
                d = data[:, i]
                data[:, i] = np.nan_to_num(d, nan=fitted_data[i])
            return data


class Imputer:
    def __init__(self, strategy: ImputerStrategy) -> None:
        """The base class for imputer objects"""
        self._strategy = strategy
        self._data = None
        self._fitted_data = None

    def fit(self, data: np.ndarray) -> "Imputer":
        self._data = data.astype(float)
        self._fitted_data = self._strategy.fit(self._data)
        return self

    def transform(self) -> np.ndarray:
        return self._strategy.transform(self._data, self._fitted_data)


class Strategy(enum.Enum):
    """Keeps track of valid and available imputer strategies."""
    mean = "mean"
    median = "median"
    mode = "mode"
    unknown = "unknown"


def create_imputer_strategy(strategy: str, axis: int = 0) -> ImputerStrategy:
    """
    Creates an imputer strategy based on input strategy
    Args:
        strategy: User specifies strategy
        axis: axis (int, optional): column=0, row=1. Default: axis=0

    Returns:
        An instance of imputer strategy (Mean, Median, Mode)
    """
    try:
        strategy = Strategy(strategy)
    except ValueError:
        strategy = Strategy.unknown

    if strategy is Strategy.unknown:
        raise RuntimeError("Unknown strategy")

    if strategy is Strategy.mean:
        return Mean(axis=axis)
    elif strategy is Strategy.median:
        return Median(axis=axis)
    else:
        return Mode(axis=axis)


if __name__ == '__main__':
    arr = np.array([['France', 44.0, 72000.0],
                    ['Spain', 27.0, 48000.0],
                    ['Germany', 30.0, 54000.0],
                    ['Spain', 38.0, 61000.0],
                    ['Germany', 40.0, np.nan],
                    ['France', 35.0, 58000.0],
                    ['Spain', np.nan, 52000.0],
                    ['France', 48.0, 79000.0],
                    ['Germany', 50.0, 83000.0],
                    ['France', 37.0, 67000.0]], dtype=object)

    strategy = create_imputer_strategy("mean", axis=0)
    imputer = Imputer(strategy)
    x = imputer.fit(data=arr[:, 1:3])
    x = imputer.transform()
    print(x.astype(object))
