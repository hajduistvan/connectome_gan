"""
@ author István Hajdu at MTA TTK
https://github.com/hajduistvan/connectome_gan
"""
from .data_handler import Autism_handler, Age_handler, UKBioBankGenderHandler, UKBioBankHandler
def get_dataset(name):
    return {"autism": Autism_handler,
            "age_small": Age_handler,
            "biobank": UKBioBankHandler,
            "biobank_gender": UKBioBankGenderHandler}[name]