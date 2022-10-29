#importing libraries
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

books_filename='BX-Books.csv'
ratings_filename='BX-Book-Ratings.csv'


#imprt csv data into dataframes
df_books = pd.read_csv(
      books_filename,
      encoding='ISO-8859-1',
      sep=';',
      header = 0,
      names = ['isbn', 'title', 'author'],
      usecols = ['isbn', 'title', 'author'],
      dtype = {'isbn': 'str', 'tilte': 'str', 'author': 'str'}
)

df_ratings = pd.read_csv(
      ratings_filename,
      encoding='ISO-8859-1',
      sep = ';',
      header = 0,
      names = ['user', 'isbn', 'rating'],
      usecols = ['user', 'isbn', 'rating'],
      dtype = {'user': 'int32', 'isbn': 'str', 'ratomg': 'float32'}
)

df = df_ratings

counts1 = df['user'].value_counts()
counts2 = df['isbn'].value_counts()

df = df[~df['user'].isin(counts1[counts1 < 200].index)]
df = df[~df['isbn'].isin(counts1[counts1 < 200].index)]

df = pd.merge(right=df, left = df_books, on='isbn')
df = df.drop_duplicates(['title', 'user'])
piv = df.pivot(index='title', columns='user', values='rating').fillna(0)

matrix = piv.values

knn_model = NearestNeighbors(metric='cosine', algorithm='brute', p=2)
knn_model.fit(matrix)

titles = list(piv.index.values)

#function to return recommended books 
def get_recommended(book=''):
      distances, indices = knn_model.kneighbors(piv.loc[book].values.reshape(1, -1), len(titles), True)
      recommended_books = [book, sum([[[piv.index[indices.flatten()[i]], distances.flatten()[i]]] for i in range(5, 0, -1)], [])]

      return recommended_books

print(get_recommended('Where the Heart Is (Oprah\'s Book Club (Paperback))'))
# print(len(titles))

# print(matrix)
