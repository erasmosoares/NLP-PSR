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

# Project Phases

## 1 - Data Preparation

In this data preparation phase, we are preparing the data for an automatic document extraction
text app.

```
01_Pytesseract.ipynb
02_Data_Preparation.ipynb
```

### 1.1 Model Entity

```
{
 'name': 'Erasmo'
 'org': 
 'date': '2023/09/14'
 'role': 'consultant'
 'base': 
 'hours': 
 'qtd': 
 'gains': 
 'deductions': 
 'nette': 
}
```

### 1.2 Obtaining Text Data

To extract text from the slip images, we utilize Pytesseract, an OCR (Optical Character
Recognition) tool developed by Google. Pytesseract allows us to convert the text within the
images into machine-readable format. Using Pytesseract, we extract the text from each of the business card images, enabling us to
obtain the necessary text data for further processing.

### 1.3 Text Cleaning and Organization

Once the text is extracted from the images, we perform basic cleaning procedures to prepare the
data for entity extraction. This may involve removing unwanted characters, correcting errors, or
standardizing formats.
After the cleaning process, we organize the extracted text by saving all the words or tokens in a
CSV (Comma-Separated Values) file. Each word or token is associated with the corresponding
filename, allowing us to maintain the connection between the extracted text and its source
slip.

### 1.4 Manual Labelling with BIO Tagging

To train a machine learning model for entity extraction, we require labeled data. In this project,
we perform manual labeling using the BIO (Begin, Inside, Outside) tagging scheme. BIO tagging
allows us to annotate each word or token in the text with a label indicating whether it is the
beginning of an entity, inside an entity, or outside any entity.
By manually labeling the data using BIO tagging, we create a ground truth dataset that serves as
the basis for training and evaluating our entity extraction model.

# 2 - Data Processing

After manually labeling the data using BIO tagging, the next step is data preprocessing by processing the data according to the Spacy training format for Named
Entity Recognition (NER).

```
03_Data_Preprocessing.ipynb
```

### 2.1 Loading and Converting the Data

1. We begin by loading the manually labeled data from the CSV or Excel file using Python's
open command. This allows us to access the data and perform further processing.
2. Once the data is loaded, we convert it into a dataframe, which provides a structured
representation of the data. The dataframe facilitates efficient manipulation and
preparation of the data for training the model.

### 2.2 Text Cleaning

To ensure high-quality training data, we perform text cleaning on the extracted text. This
involves removing any extra white spaces and eliminating special characters that are not
relevant for training the model. By cleaning the text, we enhance the accuracy and
effectiveness of the subsequent training process.

### 2.3 Converting to Spacy Training Format

 In this step, we convert the preprocessed data into the Spacy training format specifically
designed for Named Entity Recognition (NER). Spacy is a popular NLP library that
provides efficient tools and models for various natural language processing tasks.
o We take each business card text as an example and demonstrate the Spacy
training format for NER. The format typically consists of the original text and a
list of entities, where each entity is represented by its start and end indices, along
with the corresponding label.
o We follow the Spacy training format and transform the preprocessed data into this
structure for each business card text in the dataset.

### 2.4 Repetitions form all payement slips

Next, we repeat step 4 for all the business card data in our dataset. This ensures that each
business card text is processed and converted into the Spacy training format, allowing us
to include a diverse range of examples in our training data.


### 2.5 Train Test Data Splits

Finally, we split the processed data into training and testing sets. This division enables us
to evaluate the performance of the trained model on unseen data and assess its
generalization capabilities.

 We assign a portion of the processed data as the training set, which will be used to
train the NER model.
o The remaining portion is allocated as the test set, which serves as an independent
sample to evaluate the model's performance and gauge its ability to correctly
identify entities in new slips texts.

# 3 Training NER Model with Spacy

```
04_Predictions.ipynb
02_FinalPredictions.ipynb
```

In this step, we will train a Named Entity Recognition (NER) model using Spacy. The NER
model will learn to recognize and classify entities such as person names, designations,
organizations, phone numbers, emails, and websites from the labeled business card data.


### 3.1 Spacy Installation and Model Initialization

First, make sure you have Spacy installed.
Next, we need to download a pre-trained language model that provides the underlying
features for the NER task. For example, we can download the English language model.

### 3.2 Training Data Peparation

Before training the NER model, we need to prepare the training data in the Spacy format.
This format consists of a list of tuples, where each tuple represents a training example
containing the text and entity annotations.

### 3.3 Training and Testing the NER Model

With the training data prepared, we can now train the NER model using Spacy's ner
pipeline component. 

In this example, we retrieve the ner component from the loaded Spacy pipeline. We then specify
the number of training iterations (n_iter). During each iteration, we loop through the training data,
create a Doc object from the text, and convert the annotations into a Example object. We use the
nlp.update method to update the NER model with the training examples.

Once the training is complete, we can save the trained NER model for later use:

The model will be saved in the specified output directory, allowing you to load and use it for
entity extraction in the future.

# References

1. Li, J., Lu, Q., & Zhang, B. (2019). An efficient business card recognition system based
on OCR and NER. In 2019 International Conference on Robotics, Automation and
Artificial Intelligence (RAAI) (pp. 334-338). IEEE.
2. Sharma, S., & Sharma, A. (2020). Business Card Recognition using Convolutional
Neural Networks. In 2020 5th International Conference on Computing, Communication
and Security (ICCCS) (pp. 1-5). IEEE.
3. Spacy - Industrial-strength Natural Language Processing in Python. (n.d.). Retrieved
from https://spacy.io/
4. PyTesseract: Python-tesseract - OCR tool for Python. (n.d.). Retrieved from
https://pypi.org/project/pytesseract/
5. OpenCV: Open Source Computer Vision Library. (n.d.). Retrieved from
https://opencv.org/
6. Flask: A Python Microframework. (n.d.). Retrieved from
https://flask.palletsprojects.com/
7. Data Science Anyquehere & Visionview Team.

