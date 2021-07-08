import json
import os

def get_config():
  with open( os.path.dirname(__file__) + "/rpn_quantize_config.json") as json_file:
    data_dict = json.load(json_file)
    return data_dict


if __name__ == "__main__":
  print(get_config())