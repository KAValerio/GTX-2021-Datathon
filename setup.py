import os

# set up file structure and unzip LAS files
os.system('mkdir "Data for Datathon/clean_las_files"')
os.system('mkdir "Data for Datathon/Structured Data"')
os.system('unzip "Data for Datathon/well_log_files.zip" -d "Data for Datathon/clean_las_files"')