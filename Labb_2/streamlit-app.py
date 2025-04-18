import streamlit as st
import time
import pandas as pd 

# uploaded_file = st.file_uploader("streamlit-data/movies_cosine_sim.csv")
# if uploaded_file is not None:
#     movies_cosine_sim = pd.read_csv(uploaded_file).set_index("movieId", drop=True)
#     movies_cosine_sim = movies_cosine_sim.iloc[0:1300]

movies_cosine_sim = pd.read_csv("Labb2/streamlit-data/movies_cosine_sim.csv").set_index("movieId", drop=True)
movies_cosine_sim = movies_cosine_sim.iloc[0:1300]

# uploaded_file = st.file_uploader("streamlit-data/movies_cosine_sim.csv")
# if uploaded_file is not None:
#     tf_cosine_sim = pd.read_csv(uploaded_file).set_index("movieId", drop=True)
#     tf_cosine_sim = tf_cosine_sim.iloc[0:1300]

tf_cosine_sim = pd.read_csv("Labb_2/streamlit-data/tf_cosine_sim.csv").set_index("movieId", drop=True)
tf_cosine_sim = tf_cosine_sim.iloc[0:1300]

# uploaded_file = st.file_uploader("streamlit-data/movies_cosine_sim.csv")
# if uploaded_file is not None:
#     tf_idf_cosine_sim = pd.read_csv(uploaded_file).set_index("movieId", drop=True)
#     tf_idf_cosine_sim = tf_idf_cosine_sim.iloc[0:1300]

tf_idf_cosine_sim = pd.read_csv("Labb_2/streamlit-data/tf_idf_cosine_sim.csv").set_index("movieId", drop=True)
tf_idf_cosine_sim = tf_idf_cosine_sim.iloc[0:1300]

titles = pd.read_csv("Labb_2/streamlit-data/titles.csv").set_index("movieId", drop=True)

popularity_weights = pd.read_csv("Labb_2/streamlit-data/popularity.csv").set_index("movieId", drop=True)
#score_weights = pd.read_csv("streamlit-data/score.csv").set_index("movieId", drop=True)

custom_css = f"""
<style>
.stApp {{
    background-color: #181818;
    color: #cfcfcf;
}}


div.stButton>button {{
    margin-top: 28px !important;
}}


</style>
"""

# div.stButton>button {{
#     margin-top: 28px !important;
# }}

# div.stTextInput>div>div>input {{
#     border: 3px solid #181818;
#     border-radius: 20px;
#     padding: 0,5rem;
# }}

def stream_text(text, stream_word=False):
    
    if stream_word:
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.08)
    else:
        for char in text:
            yield char
            time.sleep(0.02)

def stream_list(li_st):
    for idx, st in enumerate(li_st, start=1):
        st = "["+str(idx)+"] "+st+"\n\n"
        for char in st:
            yield char
            time.sleep(0.04)
        #time.sleep(0.18)

st.markdown(custom_css, unsafe_allow_html=True)

st.markdown("# The Movie Kernel 🍿")
st.markdown("")
st.markdown("")
#st.markdown("*- A Movie Recommendation Application*\n\n\n\n")



text_input = st.text_input(""":gray[**Type the name of a movie to get recommendations**]         ||         :gray[**Type info for more information**]""").strip()
advanced_mode = st.toggle(":gray[Advanced Mode]")





col1, col2 = st.columns([1,1])
popularity_text = ":gray[Impact of Popularity on Recomendations]"
score_text = ":grey[Impact of Ratings on Recomendations]"
if advanced_mode:
    with col1: 
            popularity_slider = st.slider(f"{popularity_text}", -1.0, 1.0, 0.0)
  

    with col2:
        #score_slider = st.slider(f"{score_text}", -1.0, 1.0, 0.0)
        radio_weights = st.radio(
        ":gray[Select Type of Tag Weights]",
        [":red[Standard]",":red[TF]",":red[TF-IDF]", ],
        index=0,
        horizontal=True,
        )



if text_input: 
    if text_input in titles["title"].values:
    
        movieId = str(titles[titles["title"]==text_input].index[0])
     
        switch = { 
            ":red[Standard]":movies_cosine_sim, 
            ":red[TF]":tf_cosine_sim,
            ":red[TF-IDF]":tf_idf_cosine_sim,
        }
        #similare_movies = movies_cosine_sim[movieId].copy()
        #similare_movies = tf_cosine_sim[movieId].copy()
        #similare_movies = tf_idf_cosine_sim[movieId].copy()
        similare_movies = switch[radio_weights if advanced_mode else ":red[Standard]"][movieId].copy()

        similare_movies.drop(int(movieId), inplace=True)
        similare_movies += popularity_weights["rating"]*(popularity_slider+0.01 if advanced_mode else 0.01)
        #similare_movies += score_weights["rating"]*(score_slider+0.01)
        similare_movies.sort_values(ascending=False, inplace=True)
        #movies_cosine_sim.sort_values(movieId, ascending=False, inplace=True)
        
        top_five = list(titles.loc[[int(val) for val in similare_movies.index[1:6]],"title"])

        st.write_stream( stream_list(  [title.lower() for title in top_five] ))
       

    elif text_input.lower() == "info":
        text_help="""        
        - Make sure you have spelled the name of the movie correctly
        - Make sure you typed in the full name of the movie
        - Some alternative tiles can work
        - The search is not case sensitive so both 'Toy Story' and 'toy story' will work.
        - Prefixes 'The' and 'A' are ignored, so they can be added or omitted, the reult will remain the same
        - If you input a year make sure the formating is correct 'title (yyyy)' example 'Toy Story (1995)'
        - This movie database is limited and might not contain the movie you are looking for, in this case you might want to try the name of a similar movie
        """
        st.write_stream(stream_text("Troubelshoting"))
        st.write(text_help)
    else: 
        text_error="""
        The movie could not be found in our database,
        please check your spelling.
        """
        st.write_stream(stream_text(text_error))
        


# col1, col2 = st.columns([1,1])
# with col1: 
#     #st.write(":blue[****Coll1****]")
#     st.text_input(":red[Type in the name of the movie to get recommendations]")
# with col2:
#     st.toggle("Help")
#     # if st.button("Help"):
#     #     st.write("Type in the name")


# with st.sidebar:
#     st.write("Sidebar")

