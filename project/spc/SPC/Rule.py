import pandas as pd
from abc import ABC, abstractmethod
 
class Rule(ABC):

    @abstractmethod
    def execute(self, df: pd.DataFrame):
        pass