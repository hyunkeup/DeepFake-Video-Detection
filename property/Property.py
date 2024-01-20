import json
import os

import dotenv

from utils.FileUtils import read_json_file

dotenv.load_dotenv()
ENV = os.environ.get("ENV").lower()

current_script_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.abspath(os.path.join(current_script_directory, '..'))
property_file_path = f"{root_directory}/appsettings_{ENV}.json"
properties = read_json_file(property_file_path)


def show_property():
    print(f"Show current properties: {property_file_path}")
    print(json.dumps(properties, indent=4))


def get_property(key):
    return properties[key]
