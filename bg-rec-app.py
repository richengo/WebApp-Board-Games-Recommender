import streamlit as st
import SessionState
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import urllib

import tensorflow as tf
import tensorflow_recommenders as tfrs
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

## Import data and items
with open('./components/bgg_games_merged_topics.pkl', 'rb') as infile:
    df = pickle.load(infile)
with open('./components/ref_dictionaries.pkl', 'rb') as infile:
    ref_dicts = pickle.load(infile)
with open('./components/unique_user.pkl', 'rb') as infile:
    unique_user = pickle.load(infile)
with open('./components/user_similarity_matrix.pkl', 'rb') as infile:
    sparse_user_pivot = pickle.load(infile)
with open('./components/user_similarity_keys.pkl', 'rb') as infile:
    user_sim_dict = pickle.load(infile)
infile.close()

st.set_page_config(layout='wide')
tf.autograph.set_verbosity(2)

## Preprocessing
df['name_year'] = df['name'] + ' (' + df['year'].astype(str) + ')'
df['max_players'] = df['max_players'].map(lambda x: x.replace("+", "")).astype(int)

# Mapper (bgg_id -> name)
bg_mapper = {}
for i, name in zip(df['bgg_id'], df['name']):
    bg_mapper[str(i)] = name

# Custom function
def split_on_comma(input_data):
    return tf.strings.split(input_data, sep=',')

# Tensors
unique_user = tf.data.Dataset.from_tensor_slices(unique_user)
bgg_ids = tf.data.Dataset.from_tensor_slices(df['bgg_id'].values.astype(str))
complexity_buckets = np.linspace(1, 5, num=50)
complexity_tensor = tf.data.Dataset.from_tensor_slices(df['complexity']).batch(512)
game_type_tensor = tf.data.Dataset.from_tensor_slices(df['game_type']).batch(512)
category_tensor = tf.data.Dataset.from_tensor_slices(df['category']).batch(512)
mechanic_tensor = tf.data.Dataset.from_tensor_slices(df['mechanic']).batch(512)
tensor_candidates = tf.data.Dataset.from_tensor_slices({'bgg_id': df['bgg_id'].astype(str),
                                                        'complexity': df['complexity'].astype('float32'),
                                                        'game_type': df['game_type'],
                                                        'category': df['category'],
                                                        'mechanic': df['mechanic']
                                                       })

## Deep neural network
# User model
class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.user_lookup = tf.keras.layers.experimental.preprocessing.StringLookup()
        self.user_lookup.adapt(unique_user)
        self.user_embedding = tf.keras.Sequential([
            self.user_lookup,
            tf.keras.layers.Embedding(input_dim=self.user_lookup.vocab_size(), output_dim=32)
        ])
    def call(self, inputs):
        return tf.concat([
            self.user_embedding(inputs['bgg_user_name'])
        ], axis=1)

# Board game model
class BoardGameModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.bgg_id_lookup = tf.keras.layers.experimental.preprocessing.StringLookup()
        self.bgg_id_lookup.adapt(bgg_ids)
        self.bgg_id_embedding = tf.keras.Sequential([
            self.bgg_id_lookup,
            tf.keras.layers.Embedding(input_dim=self.bgg_id_lookup.vocab_size(), output_dim=32)
        ])
        self.complexity_normalization = tf.keras.layers.experimental.preprocessing.Normalization()
        self.complexity_normalization.adapt(complexity_tensor)
        self.complexity_embedding = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=len(complexity_buckets) + 1, output_dim=32),
            self.complexity_normalization
        ])
        self.vec_game_types = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=10_000, standardize=None, split=split_on_comma)
        self.vec_game_types.adapt(game_type_tensor)
        self.game_type_embedding = tf.keras.Sequential([
            self.vec_game_types,
            tf.keras.layers.Embedding(10_000, 32, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D()
        ])
        self.vec_categories = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=10_000, standardize=None, split=split_on_comma)
        self.vec_categories.adapt(category_tensor)
        self.category_embedding = tf.keras.Sequential([
            self.vec_categories,
            tf.keras.layers.Embedding(10_000, 32, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D()
        ])
        self.vec_mechanics = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=10_000, standardize=None, split=split_on_comma)
        self.vec_mechanics.adapt(mechanic_tensor)
        self.mechanic_embedding = tf.keras.Sequential([
            self.vec_mechanics,
            tf.keras.layers.Embedding(10_000, 32, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D()
        ])
    def call(self, inputs):
        return tf.concat([
            self.bgg_id_embedding(inputs['bgg_id']),
            self.complexity_embedding(inputs['complexity']),
            self.game_type_embedding(inputs['game_type']),
            self.category_embedding(inputs['category']),
            self.mechanic_embedding(inputs['mechanic'])
        ], axis=1)

# Query model
class QueryModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding_model = UserModel()
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(32)
        ])
    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)

# Candidate model
class CandidateModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding_model = BoardGameModel()
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(32)
        ])
    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)

# Full model
class BGDeepModel(tfrs.models.Model):
    def __init__(self):
        super().__init__()
        self.query_model = QueryModel()
        self.candidate_model = CandidateModel()
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        self.retrieval_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=tensor_candidates.batch(128).map(self.candidate_model)
            )
        )
        self.rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
    def compute_loss(self, features, training=False):
        query_embeddings = self.query_model({
            'bgg_user_name': features['bgg_user_name']
        })
        candidate_embeddings = self.candidate_model({
            'bgg_id': features['bgg_id'],
            'complexity': features['complexity'],
            'game_type': features['game_type'],
            'category': features['category'],
            'mechanic': features['mechanic']
        })
        ratings = features['bgg_user_rating']
        rating_predictions = self.rating_model(
            tf.concat([query_embeddings, candidate_embeddings], axis=1)
        )
        retrieval_loss = self.retrieval_task(query_embeddings, candidate_embeddings)
        rating_loss = self.rating_task(labels=ratings, predictions=rating_predictions)
        return (retrieval_loss + rating_loss)

# Instantiate model and download weights
model = BGDeepModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
model.load_weights("./components/deep_model_l1_weights")

# Index to take in queries and generate recommendations
index = tfrs.layers.factorized_top_k.BruteForce(model.query_model)
index.index(tensor_candidates.batch(128).map(model.candidate_model), bgg_ids)

#____________________________________________________________________________________
# Functions to import css styles
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
#
# def remote_css(url):
#     st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)
#
# def icon(icon_name):
#     st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

# Function to get similar users
@st.cache
def get_similar_users(inputs):
    selected_bg = [int(bgg_id) for name_year in inputs for bgg_id in df.loc[df['name_year']==name_year, 'bgg_id']]
    user_profile = pd.DataFrame(index=['new_user'], columns=user_sim_dict['bgg_id'])
    for bgg_id in selected_bg:
        user_profile.loc['new_user', bgg_id] = 10
    sparse_user_profile = sparse.csr_matrix(user_profile.fillna(0))
    user_similarities = pd.Series(cosine_similarity(sparse_user_profile, sparse_user_pivot)[0], index=user_sim_dict['bgg_user_name'])
    similar_users = user_similarities[user_similarities>0].sort_values(ascending=False)
    similar_users = list(similar_users.index[:10])   # Get top 10 similar users
    return similar_users, selected_bg

# Function to read image urls
def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return the image
    return cv2.resize(image, (2000,2000))

# Function to plot images
def subplot_images(image_urls, name_list):
    # nrows = int(np.ceil(len(name_list)/4))   # Makes sure you have enough rows
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    ax = ax.ravel()   # Ravel turns a matrix into a vector, which is easier to iterate
    for i, img_url in enumerate(image_urls):   # Gives us an index value to get into all our lists
        image = url_to_image(img_url)
        ax[i].set(xlim=(0,2000), ylim=(2000,0))
        ax[i].imshow(image)
        ax[i].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        txt = ax[i].text(100, 1850, name_list[i], wrap=True, ha='left', fontsize=35, family='Arial',
                   bbox=dict(boxstyle="square",
                             ec='lightgrey',
                             fc='lightgrey',
                             alpha=0.9
                   ))
        txt._get_wrap_line_width = lambda : 1300
    for axes in ax.flat[len(name_list):]:
        axes.axis('off')
    plt.margins(x=0, y=0)
    # plt.tight_layout()
    return fig

# Function to compute start and end of DataFrame
def compute_start_end(page=1):
    per_page = 12
    start = (page - 1) * per_page
    end = page * per_page
    return start, end

# Function to display the recommendations as list
def load_list(data, start, end):
    results_df = data.iloc[start:end, :]
    for i, game in zip(results_df.index, results_df['name_year']):
        st.write(f"#{i+1}. {game}")
    pass

# Function to display the recommendations as images
def load_images(data, start, end):
    results_df = data.iloc[start:end, :]
    boardgame_id = results_df['bgg_id'].values
    boardgame_imgs = [ref_dicts['images'][i] for i in boardgame_id]
    return subplot_images(boardgame_imgs, results_df['name_year'].values)
#____________________________________________________________________________________


# Start of app interface
# local_css("style.css")
st.image('./components/board-game-nomads.png', use_column_width=True)

############################
# User Inputs
############################

# Text area for user input
st.header("Enter Board Games You Like")
input_container = st.beta_container()
with input_container:
    input_col1, input_col2 = st.beta_columns([6,1])
with input_col1:
    user_input = st.multiselect("List of games", df.sort_values(by=['year', 'bayes_rating'], ascending=False)['name_year'].tolist())

# Need to use a try-except command in the case of empty input
try:
    similar_users, selected_bg = get_similar_users(user_input)

    # Genereate recommendations, put in dataframe
    _, board_games = index({'bgg_user_name': tf.constant(similar_users)}, k=100)
    unique_bg, idx, bg_count = tf.unique_with_counts(tf.reshape(board_games, [-1]))
    unique_bg = [bg.numpy().decode("utf-8") for bg in unique_bg]
    bg_count = [count.numpy() for count in bg_count]
    user_rec = pd.DataFrame({'bgg_id': unique_bg, 'freq': bg_count})
    user_rec['bgg_id'] = user_rec['bgg_id'].astype(int)
    user_rec = user_rec.merge(df, how='left', on='bgg_id')
    user_rec = user_rec[~user_rec['bgg_id'].isin(selected_bg)]
    user_rec = user_rec.drop(columns=['game_type', 'designer', 'artist', 'publisher', 'category', 'mechanic', 'num_votes', 'avg_rating', 'stddev_rating'])
    user_rec['topic_rank'] = user_rec.groupby('topic_no')['rank'].rank()
    user_rec['neg_topic_rank'] = -user_rec['topic_rank']
    empty_input = False

# ValueError due to tensorflow unable to read in empty list
except ValueError:
    user_rec = df
    empty_input = True

# Set session state to remember user actions
session_state = SessionState.get(rec_button=False, page=1, filter_options=False, filter_button=False, min_complexity=1.0, max_complexity=5.0,
                                min_year=user_rec['year'].min(), max_year=user_rec['year'].max(), max_players=10, min_age=user_rec['min_age'].min(),
                                max_time=1200.0, user_cond="year>=1995")

# Checks if user has empty input box
if empty_input:
    session_state.rec_button = False

############################
# User Interface
############################

# User buttons
with input_col2:
    rec_type = st.radio('Select recommendation type', ['Default', 'By popularity'])
    if st.button('Recommend!'):
        # If user presses without input
        if len(user_input) == 0:
            with input_col1:
                st.info("Please enter your board games!")
        else:
            session_state.rec_button = True
            session_state.page = 1
    if st.button('Cancel'):
        session_state.rec_button = False
        session_state.page = 1
with input_col1:
    if st.button('Advanced filter options'):
        session_state.filter_options = True
st.write("___________")

# Filter options
if session_state.filter_options:
    st.sidebar.write("## Filter Options")
    low_age, high_age = (user_rec['min_age'].min(), user_rec['min_age'].max())
    min_age = st.sidebar.slider("Player age (min)", min_value=low_age, max_value=high_age, value=low_age, step=1.0)
    low_player, high_player = (2, user_rec['max_players'].max())
    max_players = st.sidebar.slider("Player count (max)", min_value=low_player, max_value=high_player, value=high_player, step=1)
    max_time = st.sidebar.slider("Play time (max)", min_value=30.0, max_value=1200.0, value=1200.0, step=30.0)
    min_complexity, max_complexity = st.sidebar.slider("Complexity", min_value=1.0, max_value=5.0, value=(1.0,5.0), step=0.1)
    low_year, high_year = (user_rec['year'].min(), user_rec['year'].max())
    min_year, max_year = st.sidebar.slider("Year", min_value=low_year, max_value=high_year, value=(low_year, high_year), step=1)
    if st.sidebar.button('Filter!'):
        session_state.filter_button = True
        session_state.min_age = min_age
        session_state.max_players = max_players
        session_state.max_time = max_time
        session_state.min_complexity, session_state.max_complexity = (min_complexity, max_complexity)
        session_state.min_year, session_state.max_year = (min_year, max_year)
        session_state.user_cond = f"min_age>={session_state.min_age} & max_players<={session_state.max_players} & max_time<={session_state.max_time} & complexity>={session_state.min_complexity} & complexity<={session_state.max_complexity} & year>={session_state.min_year} & year<={session_state.max_year}"
        session_state.page = 1
    if st.sidebar.button('Clear filters'):
        session_state.filter_button = False
        session_state.page = 1
else:
    pass
with input_col1:
    if session_state.filter_button:
        st.write("Filter is: **ON**")
    elif session_state.filter_options:
        st.write("Filter is: **OFF**")


# Containers
output_title_container = st.beta_container()
output_list_container = st.beta_container()
page_container = st.beta_container()

# Items to show depending on whether user clicked rec_button
if not session_state.rec_button:
    max_page = int(np.ceil(len(df['bgg_id'])/12))  # 12 recommendations per page
    if session_state.filter_button:
        rec_df = df.query(session_state.user_cond).sort_values(by='rank').reset_index(drop=True)
    elif not session_state.filter_button:
        rec_df = df.sort_values(by='rank').reset_index(drop=True)
    with output_title_container:
        st.header("Popular board games")
        st.write("Updated as of *06 Jan 2021*.")
    with page_container:
        page_col1, page_col2, page_col3, page_col4, page_col5 = st.beta_columns([1,2,1,2,1])
    with page_col1:
        prev_page = st.button('prev')
    with page_col5:
        next_page = st.button('next')
    if prev_page:
        session_state.page -= 1
        session_state.page = max(session_state.page, 1)
    if next_page:
        session_state.page += 1
        session_state.page = min(session_state.page, max_page)
    with page_col3:
        st.write(f'page: {session_state.page} of {max_page}')

    start_page, end_page = compute_start_end(session_state.page)
    with output_list_container:
        list_col1, list_col2 = st.beta_columns(2)
    with list_col1:
        load_list(rec_df, start_page, end_page-6)
    with list_col2:
        load_list(rec_df, start_page+6, end_page)
    with st.beta_expander("Show as images"):
        st.pyplot(load_images(rec_df, start_page, end_page))

elif session_state.rec_button:
    max_page = int(np.ceil(len(user_rec['bgg_id'])/12))  # 12 recommendations per page
    if session_state.filter_button:
        rec_df = user_rec.query(session_state.user_cond)
    elif not session_state.filter_button:
        rec_df = user_rec
    with output_title_container:
        output_title_col1, output_title_col2 = st.beta_columns([6,1])
    with output_title_col1:
        st.header("Recommended games")
    with output_title_col2:
        diversify_games = st.checkbox("Diversify games!")
    with page_container:
        page_col1, page_col2, page_col3, page_col4, page_col5 = st.beta_columns([1,2,1,2,1])
    with page_col1:
        prev_page = st.button('prev')
    with page_col5:
        next_page = st.button('next')
    if prev_page:
        session_state.page -= 1
        session_state.page = max(session_state.page, 1)
    if next_page:
        session_state.page += 1
        session_state.page = min(session_state.page, max_page)
    with page_col3:
        st.write(f'page: {session_state.page} of {max_page}')

    start_page, end_page = compute_start_end(session_state.page)
    if rec_type=='Default':
        if diversify_games:
            rec_df = rec_df.sort_values(by=['neg_topic_rank', 'freq', 'bayes_rating'], ascending=False).reset_index(drop=True)
        else:
            rec_df = rec_df.sort_values(by=['freq', 'bayes_rating'], ascending=False).reset_index(drop=True)
        with output_list_container:
            list_col1, list_col2 = st.beta_columns(2)
        with list_col1:
            load_list(rec_df, start_page, end_page-6)
        with list_col2:
            load_list(rec_df, start_page+6, end_page)
        with st.beta_expander("Show as images"):
            st.pyplot(load_images(rec_df, start_page, end_page))
    elif rec_type=='By popularity':
        if diversify_games:
            rec_df = rec_df.sort_values(by=['topic_rank', 'rank']).reset_index(drop=True)
        else:
            rec_df = rec_df.sort_values(by='rank').reset_index(drop=True)
        with output_list_container:
            list_col1, list_col2 = st.beta_columns(2)
        with list_col1:
            load_list(rec_df, start_page, end_page-6)
        with list_col2:
            load_list(rec_df, start_page+6, end_page)
        with st.beta_expander("Show as images"):
            st.pyplot(load_images(rec_df, start_page, end_page))
