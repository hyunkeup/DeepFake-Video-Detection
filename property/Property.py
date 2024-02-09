import json
import os
import sys

from utils.FileUtils import read_json_file

# Python command line, `python3 extract_audios.py local`
ENV = sys.argv[1].lower()
if ENV not in ["local", "dev", "real"]:
    raise Exception("Please add the correct environment name in the python command line. [`local', 'dev', 'real']")

current_script_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.abspath(os.path.join(current_script_directory, '..'))
property_file_path = f"{root_directory}/appsettings_{ENV}.json"
properties = read_json_file(property_file_path)


def show_property():
    print(f"Show current properties: {property_file_path}")
    print(json.dumps(properties, indent=4))


def get_property(key):
    return properties[key]
