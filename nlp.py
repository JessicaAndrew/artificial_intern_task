import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
nltk.download('punkt')


def handle_data(file_path, random_seed=2048):
    def process_text(text):
        tokens = nltk.word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        words = [token for token in tokens if token not in stop_words]
        return ' '.join(words)

    wine = pd.read_csv(file_path)  # load the csv as a dataframe
   
    wine.drop(wine.columns[0], axis=1, inplace=True)  # As column not going to be used
    
    # Change to all lowercase and remove stopwords
    wine['description'] = wine['description'].apply(process_text)
    wine['variety'] = wine['variety'].apply(process_text)
    wine['country'] = wine['country'].apply(process_text)

    y = wine.pop('country')  # output
    x = wine  # inputs

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_seed)

    return x_train, x_test, y_train, y_test


def evaluate_model(model, x_test, y_test, type):
    y_predict = model.predict(x_test)

    classification = classification_report(y_test, y_predict, output_dict=False)
    confusion = confusion_matrix(y_test, y_predict)

    print('--------------------------------------------------------')
    print(f'Classification report for {type}:\n{classification}')
    print(f'Confusion matrix for {type}:\n{confusion}')
    print('--------------------------------------------------------')


def train_model(x_train, y_train, type, max_iterations=1000):
    vectorizer_options = {
        'Bag of Words': lambda: CountVectorizer(max_features=max_features),
        'Bi-gram': lambda: CountVectorizer(max_features=max_features, ngram_range=(2, 2)),
        'Tri-gram': lambda: CountVectorizer(max_features=max_features, ngram_range=(3, 3)),
        'TF-IDF': lambda: TfidfVectorizer(max_features=max_features)
    }

    text_conversion_method = vectorizer_options.get(type, lambda: CountVectorizer(max_features=max_features))()

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', StandardScaler(), ['points', 'price']),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), ['variety']),
            ('text ', text_conversion_method, 'description')
        ]
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=max_iterations))
    ])

    model.fit(x_train, y_train)
    
    return model


def train_test_model(x_train, y_train, type):
    model = train_model(x_train, y_train, type)
    evaluate_model(model, x_test, y_test, type)


max_features = 5000

# Get data to use in training and testing
x_train, x_test, y_train, y_test = handle_data("wine_quality_1000.csv")

# Run for Bag of words
train_test_model(x_train, y_train, 'Bag of words')

# Run for Bi-gram
train_test_model(x_train, y_train, 'Bi-gram')

# Run for Tri-gram
train_test_model(x_train, y_train, 'Tri-gram')

# Run for TF-IDF
train_test_model(x_train, y_train, 'TF-IDF')


# References
# https://deysusovan93.medium.com/from-traditional-to-modern-a-comprehensive-guide-to-text-representation-techniques-in-nlp-369946f67497
# https://www.deeplearning.ai/resources/natural-language-processing/
# https://www.nltk.org/
# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# https://www.w3schools.com/python/pandas/ref_df_apply.asp
# https://scikit-learn.org/stable/index.html
