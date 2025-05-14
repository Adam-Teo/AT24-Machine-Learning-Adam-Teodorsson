import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re 

# Load Datasets
# --------------
# Read in files and drop unused columns
def readin(file, drop=[]):
    return pd.read_csv(f"data/{file}.csv").drop(columns=drop)
    
ratings = readin("ratings", drop=["timestamp"])
movies  = readin("movies")
tags    = readin("tags", drop=["timestamp","userId"]) 

# print(
#     f"ratings : {ratings.shape}\n"
#     f"movies  : {movies.shape}\n"
#     f"tags    : {tags.shape}"
# )

# Redduce movies.csv
# ------------------
# 1) Remove the movies that have only gotten a few reviews
mask = ratings.groupby("movieId")["rating"].count() > 100
mask = mask[mask]
movies_r = movies[ movies["movieId"].isin(mask.index) ].copy()

# 2) Remove the movies that have a very low ratings
mask = ratings.groupby("movieId")["rating"].mean() > 3.8
mask = mask[mask]
movies_r = movies_r[ movies_r["movieId"].isin(mask.index) ] 


# Reduce tags.csv
# ---------------
# 1) Remove all movies from tags.csv that was removed from movies.csv
tags_temp = tags[ tags["movieId"].isin(movies_r["movieId"]) ].copy()

# 2) Set all tags to lower case
tags_temp["tag"] = tags_temp["tag"].astype(str)
tags_temp["tag"] = tags_temp["tag"].map(lambda x: x.lower())

# 3) Remove the least frequent tags 
mask = tags_temp.groupby("tag")["tag"].count() > 1000
mask = mask[mask]
tags_r = tags_temp[ tags_temp["tag"].isin(mask.index) ]

# Reduce ratings.csv
# ------------------
# 1) Reduce ratings
ratings_r = ratings[ ratings["movieId"].isin(movies_r["movieId"])].copy()


# Split movies.csv
# ----------------
# 1) Split movies in to titles and genres 
titles_r = movies_r.copy()
titles_r.drop(columns=["genres"], inplace=True)

genres_r = movies_r.copy()
genres_r.drop(columns=["title"], inplace=True)

# 2) Explode the genres column in movies_r
def explode(df, id_column, explode_column, split_marker):
    expand = list()
  
    # Transforms the DataFrame in to a list with the exploded columns
    for index, row in df.iterrows(): 
        expand.extend( (row[id_column], gen.lower()) for gen in row[explode_column].split(split_marker)  )

    # Transforms the list back in to a DataFrame
    df_new = pd.DataFrame({id_column:[tu[0] for tu in expand], explode_column:[tu[1] for tu in expand]})
    return df_new

genres_r = explode(genres_r, "movieId", "genres", "|")

# 3) Som movies have been taged with genres, 
#    to avoid duplicates 'tags = genres' are removed from tags  
mask = tags_r["tag"].isin(genres_r["genres"].unique())
tags_r = tags_r[~mask]

# Cosine For Tags
# ---------------   
# 1) Combine genres and tags
genres_rc = genres_r.copy()
tags_rc = tags_r.copy()
tags_genres_combined = pd.concat([tags_rc, genres_rc.rename(columns={"genres":"tag"})])
tags_genres_combined.reset_index(drop=True, inplace=True)
tags_genres_combined.drop_duplicates(inplace=True)

# 2) Create Pivot Table
tags_genres_combined["ones"]=1
tags_genres_pivot = tags_genres_combined.pivot(index="movieId", columns="tag", values="ones")
tags_genres_pivot.fillna(0, inplace=True)

# 3) Create cosine matrix
cosine_sim_matrix = cosine_similarity(tags_genres_pivot)
cosine_sim_id = pd.DataFrame(cosine_sim_matrix, index=tags_genres_pivot.index, columns=tags_genres_pivot.index )

# 4) Export
cosine_sim_id.to_csv("data/movies_cosine_sim.csv")



# TF and TF-IDF Frequency
# -----------------------

# 1) Create Groups
total_tags_per_movie = tags_r.groupby("movieId")["tag"].count()
type_of_tag_per_movie = tags_r.groupby(["movieId","tag"])["tag"]

# TF
# To get the tag frequency, divide each tag in the movie with 
# the total number of tags in the movie. 
tf = type_of_tag_per_movie.apply(lambda x: x.count() / total_tags_per_movie[x.name[0]])

# TF-IDF
# !) More TF-IDF is added after the genres are add
tf_idf = tags_r.groupby(["movieId","tag"])["tag"].count()


# 2) Add genres and adjust there weight
#    combined_weight = True  : The weight will be divided between the genres in the movie
#    combined_weight = False : The genres in the movie will be set to weight 
def add_genres(weight, combined_weight=True):
    genres_per_movie = genres_r.groupby("movieId")["genres"].count()
    genres_tf_idf = genres_r.copy()
    
    # !) .get() is needed because all movies have at aleast one 
    #    gnerea to there name, but not all movies have tags  
    if combined_weight:
        genres_tf_idf["tf"]  = genres_r["movieId"].apply(lambda x: weight/genres_per_movie[x])
        genres_tf_idf["idf"] = genres_r["movieId"].apply(lambda x: (total_tags_per_movie.get(x,1)*weight)/genres_per_movie[x] )
    else:
        genres_tf_idf["tf"]  = genres_r["movieId"].apply(lambda x: weight)
        genres_tf_idf["idf"] = genres_r["movieId"].apply(lambda x: total_tags_per_movie.get(x,1)*weight)


    for row in genres_tf_idf.itertuples():
        tf[row.movieId, row.genres] = row.tf
        tf_idf[row.movieId, row.genres] = row.idf

add_genres(1, False)


# 3) Create tf-idf
# Note that type_of_tag_per_movie_gbo is used as framework because it has the 
# groupby structure so one can use .apply and get both the x.count() and the 
# associated index aka the movieId and tag
documents = tf_idf.index.shape[0]
idf = tf_idf.unstack().count().apply(lambda x: np.log(documents/x))

terms_per_document = tf_idf.groupby("movieId").sum()
tf_ = type_of_tag_per_movie.apply(lambda x: tf_idf[x.name[0], x.name[1]] / terms_per_document[x.name[0]])

tf_idf = type_of_tag_per_movie.apply(lambda x: tf_[x.name[0], x.name[1]] * idf[x.name[1]])


# 4) Remove tags with low weight
#    At < 0.0099 it removes 15% of all tags
def remove_tags(tags, threshold, verbose=False):
    pre = tags.shape[0]
    mask = tags < 0.0099
    tags = tags.loc[~mask] 
    if verbose: 
        print(tags.shape[0]/pre)
    return tags

tf = remove_tags(tf, 0.0099)
tf_idf = remove_tags(tf_idf, 0.0099)


# 5) Cosine Similarity
#    Use the TF or TF_IDF weight adjusted tags as a base for cosine similarity
#
# !) If you want to use reset_index then make sure that the index names  
#    don't clash with the Series name. Set .pivot index to the Series
#    index and .pivot values to the Series name
def cosin_sim(tags):
    tags.name = "weights"
    tags = tags.reset_index()
    tags_pivot = tags.pivot(index="movieId", columns="tag", values="weights")
    tags_pivot.fillna(0, inplace=True)
    return cosine_similarity(tags_pivot), tags_pivot

tf_cs_matrix, tf_pivot = cosin_sim(tf)
tf_idf_cs_matrix, tf_idf_pivot = cosin_sim(tf_idf)


tf_cs_matrix = pd.DataFrame(tf_cs_matrix, index=tf_pivot.index, columns=tf_pivot.index )
tf_idf_cs_matrix = pd.DataFrame(tf_idf_cs_matrix, index=tf_idf_pivot.index, columns=tf_idf_pivot.index )

# 6) Export
tf_cs_matrix.to_csv("data/tf_cosine_sim.csv")
tf_idf_cs_matrix.to_csv("data/tf_idf_cosine_sim.csv")


# User Input Interpreter
# ----------------------
# Creates a Data Frame that is used when the app deals with user inputs

# 1) Prep titles_r
titles_to_exp = titles_r.copy().set_index("movieId", drop=True)

# 2) Regulare expression used to split up the movie titles
#    Technically there are more prefixs then 'a' and 'the'
#    The last group (.)$ is required to catch movies without a year it 
pattern = r"(.*?)(, a |, the )?(\(.*\) )?(\(\d{4})?(.)$"

# 3) Creat the Dataframe
#    Note that there are movies with more then one alternative title,
#    A function should be implemented in the streamlit app that converts
#    non-english characters like é into e
titles_exp = pd.DataFrame(index=titles_to_exp.index, columns=["prefix", "title", "alt", "title year", "alt year", "title duplicate"])
titles_exp["title duplicate"] = "-"
titles_exp["alt duplicate"] = "-"

# 4) Split up the titles and add them to a Data Frame
for row in titles_to_exp.itertuples():
    
    # Preforms the regular expression group capture
    regex_groups = re.findall(pattern, row[1].lower())

    # This handles movies that lack a year 
    regex_groups[0] = list(regex_groups[0])  
    last_character = regex_groups[0].pop()
    if last_character != ')':     
        regex_groups[0][0] += last_character
        regex_groups[0].append('(?)')  
    else : 
        regex_groups[0][-1] += ')'

    
    # !) the if else statment is needed because there is a bug with one of the movies
    #    maybe not needed
    titles_exp.loc[row[0], "title"]=regex_groups[0][0].strip() if regex_groups[0] else str()
    
    # This cleans up the string contained in the group and preforms
    # a little hack to add an empty string if regex don't capture anything
    prefix = re.findall(r"a|the" ,regex_groups[0][1])
    titles_exp.loc[row[0], "prefix"]=prefix[0] if prefix else "-"
    
    # Grabs the alternative title of the movie
    alt = re.findall(r"\(a.k.a. (.*?)\)|\((.*?)\)" ,regex_groups[0][2])
    if alt == []:
        alt = "-" 
        alt_year = "-"
    elif alt[0][0] == "":
        alt=alt[0][1]
        alt_year = f"{alt} {regex_groups[0][3]}"
    else: 
        alt=alt[0][0]
        alt_year = f"{alt} {regex_groups[0][3]}"

    # (!) Safty Probably not needed 
    year = regex_groups[0][3] if regex_groups[0][3] else str()
    titles_exp.loc[row[0], "alt"]= alt
    titles_exp.loc[row[0], "title year"]=f"{regex_groups[0][0]}{regex_groups[0][3]}"
    titles_exp.loc[row[0], "alt year"]=alt_year
    
titles_exp

# 5)
# !) Not implemented
#    Some films have the same tilte, these columns whould 
#    have been used to prompted user to add a date if they 
#    entered a title one of these movies. 
duplicate = titles_exp[titles_exp["title"].duplicated(keep=False)].index
for movieId in duplicate:
    titles_exp.loc[movieId, "title duplicate"] = titles_exp.loc[movieId, "title"]
# for movieId in duplicate:
#     print( titles_exp.loc[movieId, "title duplicate"] )

duplicate = titles_exp[titles_exp["alt"].duplicated(keep=False)].index
for movieId in duplicate:
    titles_exp.loc[movieId, "alt duplicate"] = titles_exp.loc[movieId, "alt"]
# for movieId in duplicate:
#     print( titles_exp.loc[movieId, "alt duplicate"] )

# 6) Export
titles_exp.to_csv("data/titles.csv")


# Popularity 
# ----------
# Calculate a popularity score for each movie that can be used 
# in combination with the cosine tags
# 1) Calculate Popularity 
master_index = titles_exp.index
ratings_temp = ratings_r[ ratings_r["movieId"].isin(master_index) ].copy()
ratings_temp.drop(columns="userId", inplace=True)
popularity = ratings_temp.groupby("movieId").count()

# 2) Scale
popularity_reshaped = popularity["rating"].values.reshape(-1,1)
scaler = MinMaxScaler()
scaler.fit(popularity_reshaped)
popularity_scaled = popularity.copy()
popularity_scaled["rating"] = scaler.transform(popularity_reshaped)

# 3) Export
popularity_scaled.to_csv("data/popularity.csv")