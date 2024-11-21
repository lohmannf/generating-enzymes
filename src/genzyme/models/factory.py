from genzyme.models.zymctrl import ZymCTRL
from genzyme.models.potts import PottsModel
from genzyme.models.uniform import UniformRandomModel
from genzyme.models.sedd import SEDDWrapper
from genzyme.models.deep_ebm import DeepEBM

def modelFactory(model_name: str, **kwargs):

    models = {
        "zymctrl": ZymCTRL,
        "potts": PottsModel,
        "random": UniformRandomModel,
        "sedd": SEDDWrapper,
        "debm": DeepEBM,
    }

    return models[model_name](**kwargs)