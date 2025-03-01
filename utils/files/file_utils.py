import zipfile
import os


def unzip_and_save(zip_file_path, destination_path="./"):
    """
    Unzips the specified zip file and saves its contents to the destination path.
    Removes the zip file after extraction.

    :param zip_file_path: Path to the zip file to be extracted.
    :param destination_path: Path where the contents of the zip file will be saved.
    """
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(destination_path)

    os.remove(zip_file_path)
