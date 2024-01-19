import csv
import json


class DatasetLoader:
    def __init__(self):
        pass

    @staticmethod
    def read_csv_file(file_path: str, has_header: bool = True):
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                if has_header:
                    next(reader, None)

                data = []
                for row in reader:
                    if row is not None:
                        data.append(tuple(row))

                return data

        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
        except Exception as e:
            print(f"Error: An unexpected error occurred - {str(e)}")

    @staticmethod
    def read_json_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                return data
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
        except json.JSONDecodeError:
            print(f"Error: JSON decoding failed for file - {file_path}")
        except Exception as e:
            print(f"Error: An unexpected error occurred - {str(e)}")

    def load(self, directory_path: str):
        pass
