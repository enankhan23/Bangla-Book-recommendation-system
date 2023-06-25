from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

app = Flask(__name__)

# Load the necessary data and perform the calculations
dataset = pd.read_csv("books.csv", on_bad_lines='skip')
dataset = dataset.rename(columns={"urlID|Title|Author|Publisher|Price|DiscountedPrice|Discount|Category|ISBN|Edition|Pages|Country|Language|Ratings|RatingsNum|Reviews": "Column"})
dataset[['urlID', 'Title', 'Author', 'Publisher', 'Price', 'DiscountedPrice', 'Discount', 'Category', 'ISBN', 'Edition', 'Pages', 'Country', 'Language', 'Ratings', 'RatingsNum', 'Reviews']] = dataset['Column'].str.split('|', n=15, expand=True)
dataset = dataset.drop_duplicates(subset=['Title'])
dataset = dataset[dataset['Language'] == 'Bangla']
dataset = dataset.drop(columns=['Column', 'ISBN', 'Edition', 'Language', 'urlID', 'DiscountedPrice', 'Discount', 'Reviews'])
dataset = dataset.dropna()
stored_dataset = dataset.head(10000)
stored_dataset = stored_dataset.reset_index(drop=True)
columns_to_clean = ['Title', 'Author', 'Publisher', 'Category']
dataset[columns_to_clean] = dataset[columns_to_clean].apply(lambda x: x.str.replace(r'\s+', ''))
dataset = dataset.head(10000)
dataset = dataset.reset_index(drop=True)
textual_features = dataset[['Author', 'Category', 'Publisher', 'Country', 'Title']]
author_weight = 80
category_weight = 75
publisher_weight = 60
country_weight = 20
vectorizer_textual = TfidfVectorizer()
textual_features['Combined'] = textual_features.apply(lambda row: ' '.join(row), axis=1)
textual_vectors = vectorizer_textual.fit_transform(textual_features['Combined'])
dense_textual_vectors = textual_vectors.toarray()
dense_textual_vectors[:, :4] *= [author_weight, category_weight, publisher_weight, country_weight]
textual_vectors_weighted = sparse.csr_matrix(dense_textual_vectors)
cs = cosine_similarity(textual_vectors_weighted)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        book_title = request.form['book_title']

        if book_title in stored_dataset['Title'].values:
            book_index = stored_dataset[stored_dataset['Title'] == book_title].index[0]
            scores = list(enumerate(cs[book_index]))
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            recommendations = []
            for item in sorted_scores:
                if len(recommendations) >= 5:
                    break
                if item[0] != book_index:
                    book_title = stored_dataset.iloc[item[0]]['Title']
                    recommendations.append(book_title)
            return render_template('index.html', book_title=book_title, recommendations=recommendations)
        else:
            return render_template('index.html', book_title=book_title, recommendations=["Book not found"])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
