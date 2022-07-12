# from inspect import isclass
# from pkgutil import iter_modules
# from pathlib import Path
# from importlib import import_module

# # iterate through the modules in the current package
# package_dir = Path(__file__).resolve().parent
# print(iter_modules([package_dir]))
# for (_, module_name, _) in iter_modules([package_dir]):
#     print(module_name)
#     # import the module and iterate through its attributes
#     module = import_module(f"{__name__}.{module_name}")
#     for attribute_name in dir(module):
#         attribute = getattr(module, attribute_name)

#         if isclass(attribute):
#             # Add the class to this package's variables
#             globals()[attribute_name] = attribute


from utils.dataloader import *
from utils.train_funcs import *
from utils.trainer_code import *
from utils.inkgeneration import *
from utils.register import ImageRegister
from utils.pairwise_patchextractor import Pairwise_Extractor