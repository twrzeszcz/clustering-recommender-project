import pandas as pd
import streamlit as st
import numpy as np
import pickle
from sklearn.decomposition import PCA
import gc
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import cv2


st.set_option('deprecation.showPyplotGlobalUse', False)


def main_section():
    st.title('Books Clustering and Recommendation Project')
    st.write('')
    st.write('')
    st.image('https://images.pexels.com/photos/415071/pexels-photo-415071.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940',
             use_column_width=True)
    st.markdown('This is a simple book recommendation system based on a *cosine* similarity between book descriptions '
                'encoded as pretrained embeddings from the [TensorFlow Hub](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3). '
                'The multilingual model was used because there are book descriptions in various languages in the dataset and this allows to find '
                'similar books even if descriptions are in the different languages.')
    st.markdown('**Book recommender** section allows to choose a book from a list and to select a number of similar books with descriptions '
                'that will be displayed. In addition one can visualize similar books using TSNE or PCA in 2D or 3D. Colors of the points '
                'indicate clusters to which books belong. Books were clustered into 17 clusters and some model metrics '
                'are displayed in the **Model performance** section. Clusters can be visualized in the **Clustering** section '
                'where one can choose to display either the full data, chosen number of clusters or limited number of samples.')
    gc.collect()

@st.cache
def load_data():
    df = pd.read_csv('books_prep.csv')
    return df

@st.cache
def load_embeddings():
    embeddings = np.load('embedding.npy')
    return embeddings


def load_metrics_data():
    sil_scores = np.load('sil_scores.npy')
    inertia_scores = np.load('inertia_scores.npy')
    return sil_scores, inertia_scores


def book_recommender():
    st.title('Book recommendation')
    if st.sidebar.checkbox('Load dataset'):
        df = load_data()
        embeddings = load_embeddings()
        st.success('Data successfully loaded')
    book_title = st.sidebar.selectbox('Select book', df['book_title'].unique())
    book_index = df[df['book_title'] == book_title].index
    st.markdown('## **Selected book**')
    st.markdown('**Title**: *{}*'.format(book_title))
    st.markdown('**Author**: *{}*'.format(df['book_authors'][book_index[0]]))
    st.markdown('**Description**: *{}*'.format(df['book_desc_raw'][book_index[0]]))

    if st.sidebar.checkbox('Get similar books'):
        n_neighbors = st.sidebar.slider('Select number of similar books', min_value=1)
        nbrs = NearestNeighbors(metric='cosine', n_neighbors=n_neighbors + 1).fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings[book_index])
        if st.sidebar.checkbox('List similar books'):
            st.markdown('## **Similar books**')
            for i in indices[0][1:]:
                st.markdown('**Title**: *{}*'.format(df['book_title'][i]))
                st.markdown('**Author**: *{}*'.format(df['book_authors'][i]))
                st.markdown('**Description**: *{}*'.format(df['book_desc_raw'][i]))
                st.text('--' * 50)
        if st.sidebar.checkbox('Visualize similar books'):
            if st.sidebar.checkbox('TSNE'):
                if st.sidebar.checkbox('3D'):
                    fig = px.scatter_3d(df.loc[indices[0], :], x='tsne_0', y='tsne_1', z='tsne_2',
                                        color='cluster', hover_name='book_title')
                    fig.update_layout(title='TSNE decomposition', title_x=0.5)
                    st.plotly_chart(fig)
                    del fig
                if st.sidebar.checkbox('2D'):
                    fig = px.scatter(df.loc[indices[0], :], x="tsne_0_2d", y="tsne_1_2d", color="cluster", hover_name='book_title')
                    fig.update_layout(title='TSNE decomposition', title_x=0.5)
                    st.plotly_chart(fig)
                    del fig

            if st.sidebar.checkbox('PCA'):
                if st.sidebar.checkbox('3D'):
                        fig = px.scatter_3d(df.loc[indices[0], :], x='pca_0', y='pca_1', z='pca_2',
                                            color='cluster', hover_name='book_title')
                        fig.update_layout(title='PCA decomposition', title_x=0.5)
                        st.plotly_chart(fig)
                        del fig
                if st.sidebar.checkbox('2D'):
                        fig = px.scatter(df.loc[indices[0], :], x="pca_0_2d", y="pca_1_2d", color="cluster", hover_name='book_title')
                        fig.update_layout(title='PCA decomposition', title_x=0.5)
                        st.plotly_chart(fig)
                        del fig

def clustering_module():
    st.title('Clustering')
    if st.sidebar.checkbox('Load dataset'):
        df = load_data()
        st.success('Data successfully loaded')
    if st.sidebar.checkbox('TSNE'):
        if st.sidebar.checkbox('3D'):
            if st.sidebar.checkbox('Full data - all clusters'):
                fig = px.scatter_3d(df, x='tsne_0', y='tsne_1', z='tsne_2',
                                    color='cluster', hover_name='book_title')
                fig.update_layout(title='TSNE decomposition', title_x=0.5)
                st.plotly_chart(fig)
                del fig
            if st.sidebar.checkbox('Full data - choose clusters'):
                chosen_clusters = st.sidebar.multiselect('Select clusters', sorted(df['cluster'].unique()))
                df_concat = pd.concat([df.groupby('cluster').get_group(i) for i in chosen_clusters])
                fig = px.scatter_3d(df_concat, x='tsne_0', y='tsne_1', z='tsne_2',
                                    color='cluster', hover_name='book_title')
                fig.update_layout(title='TSNE decomposition', title_x=0.5)
                st.plotly_chart(fig)
                del fig
            if st.sidebar.checkbox('Sample randomly - all clusters'):
                n_samples = st.sidebar.slider('Number of samples', min_value=1, max_value=500)
                df_sampled = df.groupby('cluster').sample(n=n_samples, random_state=123)
                fig = px.scatter_3d(df_sampled, x='tsne_0', y='tsne_1', z='tsne_2',
                                    color='cluster', hover_name='book_title')
                fig.update_layout(title='TSNE decomposition', title_x=0.5)
                st.plotly_chart(fig)
                del fig

        if st.sidebar.checkbox('2D'):
            if st.sidebar.checkbox('Full data - all clusters'):
                fig = px.scatter(df, x="tsne_0_2d", y="tsne_1_2d", color="cluster", hover_name='book_title')
                fig.update_layout(title='TSNE decomposition', title_x=0.5)
                st.plotly_chart(fig)
                del fig
            if st.sidebar.checkbox('Full data - choose clusters'):
                chosen_clusters = st.sidebar.multiselect('Select clusters', sorted(df['cluster'].unique()))
                df_concat = pd.concat([df.groupby('cluster').get_group(i) for i in chosen_clusters])
                fig = px.scatter(df_concat, x="tsne_0_2d", y="tsne_1_2d", color="cluster", hover_name='book_title')
                fig.update_layout(title='TSNE decomposition', title_x=0.5)
                st.plotly_chart(fig)
                del fig
            if st.sidebar.checkbox('Sample randomly - all clusters'):
                n_samples = st.sidebar.slider('Number of samples', min_value=1, max_value=500)
                df_sampled = df.groupby('cluster').sample(n=n_samples, random_state=123)
                fig = px.scatter(df_sampled, x="tsne_0_2d", y="tsne_1_2d", color="cluster", hover_name='book_title')
                fig.update_layout(title='TSNE decomposition', title_x=0.5)
                st.plotly_chart(fig)
                del fig

    if st.sidebar.checkbox('PCA'):
        if st.sidebar.checkbox('3D'):
            if st.sidebar.checkbox('Full data - all clusters'):
                fig = px.scatter_3d(df, x='pca_0', y='pca_1', z='pca_2',
                                    color='cluster', hover_name='book_title')
                fig.update_layout(title='PCA decomposition', title_x=0.5)
                st.plotly_chart(fig)
                del fig
            if st.sidebar.checkbox('Full data - choose clusters'):
                chosen_clusters = st.sidebar.multiselect('Select clusters', sorted(df['cluster'].unique()))
                df_concat = pd.concat([df.groupby('cluster').get_group(i) for i in chosen_clusters])
                fig = px.scatter_3d(df_concat, x='pca_0', y='pca_1', z='pca_2',
                                    color='cluster', hover_name='book_title')
                fig.update_layout(title='PCA decomposition', title_x=0.5)
                st.plotly_chart(fig)
                del fig
            if st.sidebar.checkbox('Sample randomly - all clusters'):
                n_samples = st.sidebar.slider('Number of samples', min_value=1, max_value=500)
                df_sampled = df.groupby('cluster').sample(n=n_samples, random_state=123)
                fig = px.scatter_3d(df_sampled, x='pca_0', y='pca_1', z='pca_2',
                                    color='cluster', hover_name='book_title')
                fig.update_layout(title='PCA decomposition', title_x=0.5)
                st.plotly_chart(fig)
                del fig
        if st.sidebar.checkbox('2D'):
            if st.sidebar.checkbox('Full data - all clusters'):
                fig = px.scatter(df, x="pca_0_2d", y="pca_1_2d", color="cluster", hover_name='book_title')
                fig.update_layout(title='PCA decomposition', title_x=0.5)
                st.plotly_chart(fig)
                del fig
            if st.sidebar.checkbox('Full data - choose clusters'):
                chosen_clusters = st.sidebar.multiselect('Select clusters', sorted(df['cluster'].unique()))
                df_concat = pd.concat([df.groupby('cluster').get_group(i) for i in chosen_clusters])
                fig = px.scatter(df_concat, x="pca_0_2d", y="pca_1_2d", color="cluster", hover_name='book_title')
                fig.update_layout(title='PCA decomposition', title_x=0.5)
                st.plotly_chart(fig)
                del fig
            if st.sidebar.checkbox('Sample randomly - all clusters'):
                n_samples = st.sidebar.slider('Number of samples', min_value=1, max_value=500)
                df_sampled = df.groupby('cluster').sample(n=n_samples, random_state=123)
                fig = px.scatter(df_sampled, x="pca_0_2d", y="pca_1_2d", color="cluster", hover_name='book_title')
                fig.update_layout(title='PCA decomposition', title_x=0.5)
                st.plotly_chart(fig)
                del fig

def model_metrics():
    sil_scores, inertia_scores = load_metrics_data()

    fig = go.Figure(data=go.Scatter(
        x=list(range(2, 100)),
        y=inertia_scores,
        mode='lines'
    ))
    fig.update_layout(xaxis_title='Number of clusters', yaxis_title='Inertia')
    st.plotly_chart(fig)

    fig = go.Figure(data=go.Scatter(
        x=list(range(2, 100)),
        y=sil_scores,
        mode='lines'
    ))
    fig.update_layout(xaxis_title='Number of clusters', yaxis_title='Silhouette score')
    st.plotly_chart(fig)

    embedding = load_embeddings()
    pca = PCA(n_components=0.99999)
    pca.fit(embedding)
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    fig = go.Figure(data=go.Scatter(
        x=list(range(0, 512)),
        y=cumsum,
        mode='lines'
    ))
    fig.update_layout(xaxis_title='Dimensions', yaxis_title='Explained variance')
    st.plotly_chart(fig)

    sil_coeff = cv2.imread('images/silhouette_17.jpg')
    st.image(cv2.cvtColor(sil_coeff, cv2.COLOR_BGR2RGB), use_column_width=True)




activities = ['Main', 'Book recommender', 'Clustering', 'Model metrics']
option = st.sidebar.selectbox('Select Option', activities)

if option == 'Main':
    main_section()

if option == 'Book recommender':
    book_recommender()
    gc.collect()

if option == 'Clustering':
    clustering_module()
    gc.collect()

if option == 'Model metrics':
    model_metrics()
    gc.collect()
