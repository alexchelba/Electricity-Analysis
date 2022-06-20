This is my dissertation project! I did electricity consumption analysis. The report describes in detail what I have done and what I have achieved. Below is a description of each file and what you will find in each folder.

Main Code Files:

main.py - main skeleton for the application\
form.kv - design file for the application\
utils/featureParser_script.py - script that calls the feature parsing class on multiple datasets\
utils/home_categories_parse.py - categories parser class. The implementation of household categorization is found here, as well as data aggregation.\
utils/home_features_parse.py - feature parser class. The implementation of data cleaning is found here.\
ARIMA/EDA.py - Exploratory data analysis class. All the plots that are visible in Exploratory data analysis section were created using this class.\
ARIMA/EDA_script.py - script that calls the EDA class on multiple datasets\
ARIMA/modelHelper.py - ARIMA model builder class. The implementation of ARIMA model construction is found here.\
ARIMA/modelScript.py - script that calls the model builder classes on multiple datasets.\
ARIMA/naiveModel.py - Naive model builder class. The implementation of Naive model construction is found here.


Other Files:

Folder API/ was provided with the dataset files. The classes in there are used for reading electricity data from the dataset.\
You will see there are jupyter notebooks every here and there. Ignore them, they were for testing out code before putting it in scripts.

Image folders:\
EDA_*/ - contains the figures saved from Exploratory Data Analysis for the respective category\
imgs2/ - contains photos from aggregating datasets for all categories, as well as a screenshot of the application's UI.


Model folders:\
arima_nonseasonal/ - contains ARIMAX results and model files for all categories.\
sarimax/ - contains SARIMAX results for the 3 analysed categories.\
naive/ - contains Naive results for all categories.

Data folders:\
generated_data2/ - contains the final datasets, used in the analysis.\
regenerated/ - contains the datasets created after data aggregation.
