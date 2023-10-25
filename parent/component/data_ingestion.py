import os
import zipfile
import subprocess
import csv

def read_data():
   
    dataset_identifier = 'hijest/genre-classification-dataset-imdb'
    destination_folder = 'D:/Projects/Movie_Genre_Classification/datasets'

    # Run the Kaggle command to download the dataset
    command = f'kaggle datasets download -d {dataset_identifier} -p {destination_folder} --force'
    subprocess.call(command, shell=True)

   
    file_list = os.listdir(destination_folder)
    zip_files = [file for file in file_list if file.endswith('.zip')]
    for zip_file in zip_files:
        # Define the name of the downloaded zip file
        zip_file_name=zip_file
  
    # Unzip the downloaded file to the destination folder
    zip_file_path = os.path.join(destination_folder, zip_file_name)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)
    
    # List all items (files and subdirectories) in the specified folder
    items = os.listdir(destination_folder)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(destination_folder, item))]
    for subdirectory in subdirectories:
        convert_to_csv(os.path.join(destination_folder, subdirectory))
   


def convert_to_csv(destination_folder):
   
    
    file_list = os.listdir(destination_folder)
    text_files = [file for file in file_list if file.endswith('.txt')]

    # Specify the delimiter used in the text file (e.g., pipe "|")
    text_file_delimiter = ":::"
    
    # Specify the character encoding 
    input_encoding='ISO-8859-1'
    output_encoding = 'utf-8'

    for text_file in text_files:
        file_path = os.path.join(destination_folder, text_file)
        # Read the text file and convert it to a CSV file
        with open(file_path, 'r',encoding=input_encoding) as text_file, open(os.path.join(destination_folder,  os.path.splitext(os.path.basename(file_path))[0]+'.csv'), 'w', newline='',encoding=output_encoding) as csv_file:
            # file_name, file_extension = os.path.splitext(os.path.basename(file_path))
                   
                writer = csv.writer(csv_file)
                if os.path.splitext(os.path.basename(file_path))[0] !='test_data':
                    writer.writerow(["ID", "TITLE", "GENRE", "DESCRIPTION"])
                else:
                    writer.writerow(["ID", "TITLE","DESCRIPTION","GENRE"])

                # Read each line from the text file and split it by the specified delimiter (e.g., '|')
                for line in text_file:
                    # Remove leading/trailing whitespace and split the line
                    values = line.strip().split(text_file_delimiter)

                    # Write the split values as separate columns in the CSV file
                    writer.writerow(values)

def main():
    read_data()
 
if __name__ == "__main__":
    main()