from src.models.zymctrl import ZymCTRL
from src.models.potts import PottsModel
from src.models.uniform import UniformRandomModel
from src.models.sedd import SEDDWrapper

def modelFactory(model_name: str, **kwargs):

    models = {
        "zymctrl": ZymCTRL,
        "potts": PottsModel,
        "random": UniformRandomModel,
        "sedd": SEDDWrapper,
    }

    return models[model_name](**kwargs)