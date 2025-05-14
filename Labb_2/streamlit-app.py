import streamlit as st
import time
import pandas as pd 
import re

#path = "C:/Code/AT24-Maskininlärning-Adam-Teodorsson/Labb_2/streamlit-data/"
path = "streamlit-data/"

"""
li_csv = ["movies_cosine_sim", "tf_cosine_sim", "tf_idf_cosine_sim", "titles", "popularity" ]
di_csv = { 
    csv:pd.read_csv(path+csv+".csv").set_index("movieId", drop=True)
    for csv 
    in li_csv
    } 
"""


# titles = li_csv['titles'] pd.read_csv(path+"titles.csv").set_index("movieId", drop=True)
popularity_weights = pd.read_csv(path+"popularity.csv").set_index("movieId", drop=True)


custom_css = f"<style>.stApp {{ background-color: #181818; color: #cfcfcf; }}</style>"

# Used to give a streaming effect to text
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
     

st.markdown(custom_css, unsafe_allow_html=True)

st.markdown("# The Movie Kernel 🍿")
st.markdown("")
st.markdown("")

# Used to clean up the user inupt from 'a' and 'the'
def clean_input(str_input): 
    str_input = str_input.strip().lower()
    str_input = re.findall(r"(^a |^the )?(.*)", str_input)
    str_input = str_input[0][1].strip()
    return str_input

user_input_label ="""
:gray[**Type the name of a movie to get recommendations**] 
|| :gray[**Type info for more information**] """

user_input = clean_input( st.text_input(user_input_label) )
advanced_mode = st.toggle(":gray[Advanced Mode]")

# Contians the advanced widgets 
col1, col2 = st.columns([1,1])
popularity_text = ":gray[Impact of Popularity on Recomendations]"
score_text = ":grey[Impact of Ratings on Recomendations]"
if advanced_mode:
    with col1: 
            popularity_slider = st.slider(f"{popularity_text}", -1.0, 1.0, 0.0)
  
    with col2:
        radio_weights = st.radio(
        ":gray[Select Type of Tag Weights]",
        [":red[Standard]",":red[TF]",":red[TF-IDF]", ],
        index=0,
        horizontal=True,
        )

# Check a bunch of clauses for the title ex is it 
# the standard title, alternative, does it include
# the year 
# (!) title duplicates are not implmented
def check_title(str_input):
   # if str_input in titles["title duplicate"].values:
        #st.markdown("title duplicates")
   #    return True, "title duplicates"
    if str_input in titles["title"].values:
        #st.markdown("title")
        return True, "title"
    elif str_input in titles["alt"].values:
        #st.markdown("alt")
        return True, "alt"
    elif str_input in titles["title year"].values:
        #st.markdown("title year")
        return True, "title year"
    elif str_input in titles["alt year"].values:
        #st.markdown("alt year")
        return True, "alt year"
    return False, "-"


# Takes the user input and recommends movies
if user_input: 
    if check_title(user_input)[0]:
    
        movieId = str(titles[titles[check_title(user_input)[1]]==user_input].index[0])
     
        switch = { 
            ":red[Standard]":di_csv["movies_cosine_sim"], 
            ":red[TF]":di_csv["tf_cosine_sim"],
            ":red[TF-IDF]":di_csv["tf_idf_cosine_sim"],
        }

        # Switch between types of weights
        similare_movies = switch[radio_weights if advanced_mode else ":red[Standard]"][movieId].copy()

        # drop the target movie so that it don't accidentally shows up in the recommendation
        similare_movies.drop(int(movieId), inplace=True)

        # Grabs the series of similarity of the user input movie
        # and applice popularity to the similarity series
        similare_movies += popularity_weights["rating"]*(popularity_slider+0.01 if advanced_mode else 0.01)
        
        # Sort the series of similar movies
        similare_movies.sort_values(ascending=False, inplace=True)
        
        # Selects the top five movies using a slize
        top_five = list(titles.loc[[int(val) for val in similare_movies.index[1:6]],"title"])

        # Prints the top five recommended movies
        st.write_stream( stream_list(  [title.lower() for title in top_five] ))
       
# Info function
    elif user_input.lower() == "info":
        text_help="""        
        - Make sure you have spelled the name of the movie correctly
        - Make sure you typed in the full name of the movie
        - Some alternative tiles can work
        - The search is not case sensitive so both 'Toy Story' and 'toy story' will work.
        - Prefixes 'The' and 'A' are ignored, so they can be added or omitted, the reult will remain the same
        - If you input a year make sure the formating is correct 'title (yyyy)' example 'Toy Story (1995)'
        - This movie database is limited and might not contain the movie you are looking for, in this case you might want to try the name of a similar movie
        ___ 
        Advanced Mode
        - The 'Impact of Popularity Slider' controls how much the popularity of movies should effect the recomendations
        - Use 'Select Type of Tag Weights' to switch between different ways to weight the tags
        """
        st.write_stream(stream_text("Troubelshoting"))
        st.write(text_help)
    else: 
        text_error="""
        The movie could not be found in our database,
        please check your spelling.
        """
        st.write_stream(stream_text(text_error))
        
