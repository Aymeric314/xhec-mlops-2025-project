import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Union

import pandas as pd


def get_data():
    return pd.read_csv("../../data/abalone.csv")


# Use this module to code a `pickle_object` function. This will be useful to pickle the model (and encoder if need be).
def pickle_object(obj: Any, filepath: Union[str, Path]) -> None:
    """Serialize `obj` to `filepath` using pickle and an atomic move.

    Creates parent directories if needed, writes the pickle to a temporary file
    in the same directory, then moves the temp file into place to avoid partial
    files. Uses the highest available pickle protocol.

    Parameters
    - obj: Any picklable Python object.
    - filepath: Destination path (str or pathlib.Path).

    Raises
    - pickle.PicklingError, OSError, or other exceptions from IO/pickling.
    - Note: Never unpickle data from untrusted sources.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create a temp file in the same directory to allow atomic move
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        shutil.move(tmp_path, str(path))
    except Exception:
        # Clean up temp file on error
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise
