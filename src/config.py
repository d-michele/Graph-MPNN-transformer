import yaml


class Config:
    def __init__(self):
        with open("hyp.yml", "r") as stream:
            yaml_params = yaml.safe_load(stream)
            for param, value in yaml_params.items():
                setattr(self, param, value)
