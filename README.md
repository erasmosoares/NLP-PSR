# Computer Vision and NLP on OCR text extraction (Supervised ML)

# NLP Pay Stub Reader

The NLP PSR is a project aimed at developing an automated system for
extracting key information from pay stubs. The web app provides users with the ability to
upload their payement slips, automatically detect and extract relevant entities, and
visualize the extracted information. The project incorporates various techniques such as optical
character recognition (OCR), named entity recognition (NER), and image processing to achieve
its objectives.

# Project Structure

This projet is divided into 2 sub-projects
* Slips-app (flask-front-end)
* Slips-reader
    * 01_Pytesseract.ipynb
    * 02_Data_Preparation.ipynb
    * 03_Data_Preprocessing.ipynb
    * 04_Predictions.ipynb
    * 05_FinalPredictions.ipynb
 
# Instructions

Make sure to create a python environment with all dependencies and packages, if you use Anaconda, create a new environment on Anaconda.Navigator based on base root and add these packages:

  
1. pandas
2. cv2
3. pytesseract
4. os
5. glob
6. tqdm
7. spacy

On the terminal just replace myenv to your environement name:

```
conda activate myenv
```

## Slips-reader

1. Create a new folder called stubs into Slips-Reader directory, add a screenshot of a pay stub and name it Slip.png. You can also add multiples files, they are going to be considered on the data preparation process.
2. Run 01_Pytesseract.ipynb
3. Run 02_Data_Preparation.ipynb, this step will create a slips.csv file.
4. BIO Tag the slips.scv file and save a file as slips-tag.txt at the same level of your slips.csv. If you have some question on how to labelling your data, please check the article at the end of this readme.
5. Create a new folder called data into Slips-Reader directory and run 03_Data_Preprocessing.ipynb, thils will create two pickle format files in data directory TrainData.pickle and TestData.pickle
6. Before predictions, you have to train the NRE model, to do this, run the preprocess.py (python preprocess.py) file on the root of this project, make sure the training_data and TestData are correct on the directory, the execution will create a file called train.spacy.
7. Now you can train the model using the command:

Windows   
```
python -m spacy train .\config.cfg --output .\output --paths.train .\data\train.spacy --paths.dev .\data\test.spacy
```

MAC   
```
python -m spacy train ./config.cfg --output ./output --paths.train ./data/train.spacy --paths.dev ./data/test.spacy
```

8. This process wil take some time, after finish you will see in the output folder two subfolders, model-best and model-last.
9. Run 04_Prediction to see the final entity extracted, note the predictions file is configured to parse slips that contains theses attributes ( name, org, role, date, base, hours, ltd, gains, deductions, nette), you should adjust as necessary.
10. The 05_FinalPredictions.ipynb is just a test to read the predictions.py, which is a synthesized python file containing everything from 04_Predictions.

## Slips-App

This is a front-end app in Flask to upload an image and extract the text, to execute, copy the output folder of your Slips-Reader into the app directory and execute:

```
python main.py
```

