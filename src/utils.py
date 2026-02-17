import os
import sys
import dill

from src.exception import CustomException

def save_object(file_Path, obj):
    try:
        dir_path = os.path.dirname(file_Path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_Path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
