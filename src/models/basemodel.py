import abc

class BaseModel(abc.ABC):

    @abc.abstractmethod
    def run_training(self):
        pass
    
    @abc.abstractmethod
    def generate(self):
        pass