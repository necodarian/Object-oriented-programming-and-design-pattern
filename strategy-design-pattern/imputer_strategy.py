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
    """The base class for imputer objects"""
    """User will specify which imputation method"""
    """axis (int, optional): column=0, row=1. Default: axis=0"""
    def __init__(self, strategy: str = "mean", axis: int = 0) -> None:
        self._data = None
        self._fitted_data = None
        if strategy == "mean":
            self._strategy = Mean(axis=axis)
        elif strategy == "median":
            self._strategy = Median(axis=axis)
        elif strategy == "mode":
            self._strategy = Mode(axis=axis)
        else:
            raise RuntimeError(f"Unknown strategy: {strategy}.")

    def fit(self, data: np.ndarray) -> "Imputer":
        self._data = data.astype(float)
        self._fitted_data = self._strategy.fit(self._data)
        return self

    def transform(self) -> np.ndarray:
        return self._strategy.transform(self._data, self._fitted_data)


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


    imputer = Imputer(strategy="mean", axis=0)
    imputer = imputer.fit(data=arr[:, 1:3])
    x = imputer.transform()
    print(x.astype(object))


