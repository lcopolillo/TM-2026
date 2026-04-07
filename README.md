# Text Mining Final Project | Sentiment Analysis
Text Mining project for the MSc curricular unit TMCD (2025/2026, 2nd semester).  

The goal is to explore multiple approaches to sentiment classification on a dataset of English texts.

Authors:
- Luiza Coelho
- Pedro Louro
- Tiago Vieira

## Setup

**Requirements:** 
- Python 3.13+
- git


## 1. Clone the repository:
   ```bash
   git clone https://github.com/lcopolillo/TM-2026
   cd TM-2026
   ```

## 2. Setup a virtual environment
Recommended to isolate dependencies. 

**Create the venv:**
```bash
python -m venv .venv
```
**Activate the venv:**
```bash
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Create a `data` directory for storing the dataset (source: Moodle)**

```bash
mkdir data
```

Place the datasets in the `data` directory. The expected structure is:

    data/
    ├── amazon_reviews_test.csv
    ├── amazon_reviews_train.csv
    ├── imdb_reviews_test.csv
    ├── imdb_reviews_train.csv
    ├── rotten_tomatoes_test.tsv
    ├── rotten_tomatoes_train.tsv
    ├── Tweets_EN_sentiment_test.csv
    └── Tweets_EN_sentiment_train.csv


## 3. Deactivate the venv:
When done, you can deactivate the virtual environment with the following command:
   ```bash
   deactivate
   ```


### Notebooks
- `00_dataset_selection.ipynb`: Preliminary and Statistical Analysis of the available datasets, and selection of the one to be used for the project.