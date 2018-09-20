from .data_handler import Autism_handler, Age_handler


def get_dataset(name):
    return {"autism": Autism_handler, "age_small": Age_handler}[name]