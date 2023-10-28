import subprocess
if __name__ == '__main__':
    

    # Define the path to the requirements.txt file
    requirements_file = 'D:/Projects/Movie-Genre-Classification/requirements.txt'

    # Run pip install using subprocess to install packages
    try:
        subprocess.check_call(['pip', 'install', '-r', requirements_file])
        print("Packages installed successfully.")
    except subprocess.CalledProcessError:
        print("Failed to install packages.")