import sys
import os
import os.path as opath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def get_args():
    if len(sys.argv) < 2:
        raise Exception("Please provide a directory")
    root_directory = sys.argv[1]
    return root_directory


def get_directory_files(directory_path):
    if not os.path.isdir(directory_path):
        raise Exception("Please provide a valid directory")
    dir_files = get_recursive_files(directory_path)
    return filter_directory_files(dir_files)


def get_recursive_files(directory_path):

    directory_files = {}

    # Get all files in subdirectories
    for path, _, files in os.walk(directory_path):
        if path == directory_path:
            continue
        for name in files:
            path_name = path.split("/")[len(path.split("/")) - 1]
            if path_name not in directory_files:
                directory_files[path_name] = [name]
            else:
                directory_files[path_name].append(name)

    # Return directory_path files if no subdirectories
    if not directory_files:
        name = directory_path.split("/")[len(directory_path.split("/")) - 1]

        directory_files[name] = [f for f in os.listdir(directory_path)
                                 if opath.isfile(
                opath.join(directory_path, f))]

    return directory_files


def filter_directory_files(directory_files):
    for key in directory_files.keys():
        directory_files[key] = [file for file in directory_files[key]
                                if file.lower().endswith(".jpg")
                                or file.lower().endswith(".jpeg")
                                or file.lower().endswith(".png")]

    # Check if there are files in directory_files
    if not any(directory_files.values()):
        raise Exception("No image files found in the directory")

    return directory_files


def display_directory_files(directory_files):
    num_folders = len(directory_files)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, num_folders))
    folder_names = list(directory_files.keys())
    folder_sizes = [len(files) for files in directory_files.values()]
    color_map = dict(zip(folder_names, colors))

    fig, ax = plt.subplots(1, 2, figsize=(14, 10))
    ax[0].pie(folder_sizes,
              labels=folder_names,
              autopct='%1.1f%%',
              colors=[color_map[name] for name in folder_names])
    ax[0].set_title("Répartition des fichiers par dossier")

    ax[1].bar(folder_names, folder_sizes,
              color=[color_map[name] for name in folder_names])
    ax[1].set_xticklabels(folder_names,
                          rotation=45,
                          ha="right")
    ax[1].set_title("Nombre de fichiers par dossier")
    ax[1].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.show()


def main():
    directory_path = get_args()
    directory_files = get_directory_files(directory_path)
    display_directory_files(directory_files)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
