import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re 

# Load Data Sets
# --------------
# Read in the files and drop unused columns
def readin(file, drop=[]):
    return pd.read_csv(f"data/{file}.csv").drop(columns=drop)
    
ratings = readin("ratings", drop=["timestamp"])
movies  = readin("movies")
tags    = readin("tags", drop=["timestamp","userId"]) 

print(
    f"ratings : {ratings.shape}\n"
    f"movies  : {movies.shape}\n"
    f"tags    : {tags.shape}"
)


# Redduce movies.csv
# ------------------
# 1) Remove the movies that have few reviews
mask = ratings.groupby("movieId")["rating"].count() > 100
mask = mask[mask]
movies_r = movies[ movies["movieId"].isin(mask.index) ].copy()

# 2) Remove the movies that have a low ratings
mask = ratings.groupby("movieId")["rating"].mean() > 3.8
mask = mask[mask]
movies_r = movies_r[ movies_r["movieId"].isin(mask.index) ] 
movies_r.shape


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
# 1) Split movies in to titels and genres 
titles_r = movies_r.copy()
titles_r.drop(columns=["genres"], inplace=True)

genres_r = movies_r.copy()
genres_r.drop(columns=["title"], inplace=True)

# 2) Explode the genres column in Movies
def explode(df, id_column, explode_column, split_marker):
    expand = list()
  
    # Transforms the DataFrame in to a list with the exploded columns
    for index, row in df.iterrows(): 
        expand.extend( (row[id_column], gen.lower()) for gen in row[explode_column].split(split_marker)  )

    # Transforms the list back in to a DataFrame
    df_new = pd.DataFrame({id_column:[tu[0] for tu in expand], explode_column:[tu[1] for tu in expand]})
    return df_new

genres_r = explode(genres_r, "movieId", "genres", "|")

# 3) Remove all genres that also exists in tags
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

# 2) Create Pivote Table
tags_genres_combined["ones"]=1
tags_genres_pivot = tags_genres_combined.pivot(index="movieId", columns="tag", values="ones")
tags_genres_pivot.fillna(0, inplace=True)

# 3) Create cosine matrix
cosine_sim_matrix = cosine_similarity(tags_genres_pivot)
cosine_sim_id = pd.DataFrame(cosine_sim_matrix, index=tags_genres_pivot.index, columns=tags_genres_pivot.index )

# 4) Export
cosine_sim_id.to_csv("data/movies_cosine_sim.csv")


# TF Frequency
# ------------
# 1) Create Groups
tags_r.groupby("movieId")["movieId"].count().sort_values()
total_tags_per_movie = tags_r.groupby("movieId")["tag"].count()
type_of_tag_per_movie = tags_r.groupby(["movieId","tag"])["tag"]

# 2) Divide the number of each specific tag in a movie with the total number of tags in
#    the movie to get the tag frequencey for each tag
tag_frequency_per_movie = type_of_tag_per_movie.apply(lambda x: x.count() / total_tags_per_movie[x.name[0]])


# 3) Add genres and set there weight
# combined_weight = True  : The weight will be divided between the genres in the movie
# combined_weight = False : The genres in the movie will be set to weight 
def add_genres_tf(weight, combined_weight=True):
    genres_per_movie = genres_r.groupby("movieId")["genres"].count()

    if combined_weight:
        genres_r["weight"] = genres_r["movieId"].apply(lambda x: weight/genres_per_movie[x])
    else:
        genres_r["weight"] = genres_r["movieId"].apply(lambda x: weight)

    for row in genres_r.itertuples():
        tag_frequency_per_movie[row.movieId, row.genres] = row.weight
    
    return genres_r
add_genres_tf(1, True)

# 4) Remove tags with little weight
mask = tag_frequency_per_movie < 0.0099
tag_frequency_per_movie = tag_frequency_per_movie.loc[~mask]

# 5) TF Cosine 
tf = tag_frequency_per_movie.copy()
# (!) If you want to use reset_index then can not be the same as one of the index names  
#     because the name becomes the name of the value column 
tf.name = "weights"
tf = tf.reset_index()
tf_pivot = tf.pivot(index="movieId", columns="tag", values="weights")
tf_pivot.fillna(0, inplace=True)

tf_cs_matrix = cosine_similarity(tf_pivot)

tf_cs_matrix = pd.DataFrame(tf_cs_matrix, index=tags_genres_pivot.index, columns=tags_genres_pivot.index )
tf_cs_matrix.to_csv("data/tf_cosine_sim.csv")


# TF-IDF Frequency
# ----------------
# 1) Create Groups
# Not needed as they are created in TF Frequency
# tags_r.groupby("movieId")["movieId"].count().sort_values()
# total_tags_per_movie = tags_r.groupby("movieId")["tag"].count()
# type_of_tag_per_movie = tags_r.groupby(["movieId","tag"])["tag"]

# (!) This is probably a roundabout way of doing it but 
#     this is needed to maintain the groupby structure at a later stage
type_of_tag_per_movie_gbo = tags_r.groupby(["movieId","tag"])["tag"].count()


# 2) Add genres and set there weight
def add_genres_tf_idf(weight, combined_weight=True):

    # (!) don't know why geners_r[movieId].unique() is larger then total_tags_per_movie.index
    #     investigate this att some point
    mask = genres_r["movieId"].isin(total_tags_per_movie.index)
    genres_temp = genres_r[mask].copy()
    genres_temp

    genres_per_movie = genres_temp.groupby("movieId")["genres"].count()

    if combined_weight:
        genres_temp["n"] = genres_temp["movieId"].apply(lambda x: np.ceil( (total_tags_per_movie[x]*weight)/genres_per_movie[x] ) )
    else:
        genres_temp["n"] = genres_temp["movieId"].apply(lambda x: np.ceil( (total_tags_per_movie[x]*weight) ) )


    for row in genres_temp.itertuples():
        type_of_tag_per_movie_gbo[row.movieId, row.genres] = row.n

    return genres_temp
add_genres_tf_idf(1, False)
type_of_tag_per_movie_gbo


# 3) Create tf-idf
# Note that type_of_tag_per_movie_gbo is used as frame work because it has the groupby structure
# so one can use .apply and get both the x.count() and the associated index aka the movieId and tag
documents = type_of_tag_per_movie_gbo.index.shape[0]
idf = type_of_tag_per_movie_gbo.unstack().count().apply(lambda x: np.log(documents/x))

terms_per_document = type_of_tag_per_movie_gbo.groupby("movieId").sum()
tf = type_of_tag_per_movie.apply(lambda x: type_of_tag_per_movie_gbo[x.name[0], x.name[1]] /  terms_per_document[x.name[0]])

tf_idf = type_of_tag_per_movie.apply(lambda x: tf[x.name[0], x.name[1]] * idf[x.name[1]] )


# # tf-idf Cosine
# tf_idf_copy = tf_idf.copy()
# # (!) If you want to use reset_index then can not be the same as one of the index names  
# #     because the name becomes the name of the value column 
# tf_idf_copy.name = "weights"
# tf_idf_copy = tf_idf_copy.reset_index()
# tf_idf_pivot = tf_idf_copy.pivot(index="movieId", columns="tag", values="weights")
# tf_idf_pivot.fillna(0, inplace=True)

# tf_idf_cs_matrix = cosine_similarity(tf_idf_pivot)
# tf_idf_cs_matrix.shape
# tf_idf_cs_matrix = pd.DataFrame(tf_idf_cs_matrix, index=tags_genres_pivot.index, columns=tags_genres_pivot.index )
# tf_idf_cs_matrix.to_csv("data/tf_idf_cosine_sim.csv")



# User Input Interpreter
# ----------------------
# Creates a Data Frame that is used when the app deals with user inputs

# 1) Bug Fix
# (!) There was a bug I couldent figure out so I hade to reduce the 
#     movies to a total of 1300 
test_2 = titles_r.loc[titles_r.index[0:1300]].copy()
test_2 = test_2.set_index("movieId", drop=True)

# 2) Regulare expression used to split up the movie titles in to 
#    differnt parts and combinations. 
#    Technically there are more prefixs then 'a' and 'the'
pattern = r"(.*?)(, a |, the )?(\(.*\) )?(\(\d{4}\))"


# 3) Creat the Dataframe
# Note that there can be more then one alternative title, by I couldn't be bothered
# so there is max one alternative title per movie
# Technically one probelbly should make a title without non english characters like é
titles_exp = pd.DataFrame(index=test_2.index, columns=["prefix", "title", "alt", "title year", "alt year", "title duplicate"])
titles_exp["title duplicate"] = "-"
titles_exp["alt duplicate"] = "-"

# 4) Split up the titles and add them to a Data Frame
for row in test_2.itertuples():
    
    regex_groups = re.findall(pattern, row[1].lower())
    
    # (!) the if else statment is needed because there is a bug with one of the movies
    titles_exp.loc[row[0], "title"]=regex_groups[0][0].strip() if regex_groups[0] else str()
    
    # This cleans up the string contained in the gruop and preforms
    # a little hack to add an empty string if regex don't capture anything
    prefix = re.findall(r"a|the" ,regex_groups[0][1])
    titles_exp.loc[row[0], "prefix"]=prefix[0] if prefix else "-"
    
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
# (!) Not implemented
# Some films have the same tilte, these columns where going to 
# be used in a scheme where the user would have been prompted 
# to add the year
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