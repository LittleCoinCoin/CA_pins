import os

def Check_Make_Dir(path):
    """
    Checks if a directory exists at the given path, and if not, creates it.
    Args:
        path (str): The path where the directory should be checked/created.
    Returns:
        None
    """
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == "__main__":

    #paths for the data that will be generated
    root = "./data"
    patterns_path = root + "/patterns"
    trajectories_path = root + "/trajectories"
    pinned_trajectories_path = root + "/pinned_trajectories"
    pinned_results_path = root + "/pinned_results"
    patterns_1D_path = root + "/patterns/1D"
    patterns_1D_space_path = root + "/patterns/1D/space" # for patterns along the 1D space at a single time step
    patterns_1D_time_path = root + "/patterns/1D/time" # for patterns along the time axis at a single space cell

    all_paths = [root,
                 patterns_path,
                 trajectories_path,
                 pinned_trajectories_path,
                 pinned_results_path,
                 patterns_1D_path,
                 patterns_1D_space_path,
                 patterns_1D_time_path
                 ]

    # Check the directories and create them if they don't exist
    for _path in all_paths:
        Check_Make_Dir(_path)