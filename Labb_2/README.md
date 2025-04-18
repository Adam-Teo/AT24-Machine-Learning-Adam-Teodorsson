### Movie Kernel 
##### *- A Movie Recommendation Application*

### Preprocessing
This application uses Collaborative Filtering instead of Content Filtering mainly because the user does not have a profile, this does not mean that Content Filtering couldn't work as profiles could created for the users already present in the data sets or even for the movies them selves but it seemed more natural to look at what other users have liked and tagged in the past.

Three data sets where used: ratings.csv, movies.csv, tags.csv
These where pruned heavily for a variety of reasons: 
- ratings.csv
	The number of users where reduced in order to free up memory and speed up processing, the users removed where the ones that had rated the fewest movies because theoretically (have not actually checked) people with few ratings tend to rate more popular movies and thus removing them will not reduce the variety of movies that would be recommended.
	This gives the the ratings a certain bias towards people who spend a lot of time rating movies, and if this group enjoys certain type of movie more than the casual movie rater then this will effect the recommendations.

- tags.csv
	Tags had a low frequency where removed because they might be a bit arbitrary there inclusion might have been the result of a person making a mistake as they tagged the movie. Although if this was not the case then they might have made them very valuable as if a very specific tag where charred amongst only a couple of movies this would heavily link them to gather in the eyes of the algorithm. Also single tags where removed in this process because in this context they are basically usles as they can't link to another movie.

 - movies.csv
	This set was reduced by removing movies that had a low rating, because this is a movie recommendation app, low rated movies are not that interesting, the majority of people likely want to be recommended movies that people enjoy al though there most likely are exceptions to this like people with strange taste.

### Cosine Similarity
 To find movies that are similar to the movie the user inputs the tags and genres of the movies where combined in to a pivot table. The pivot table where then used in combination with the `sklearn` Cosine similarity functions, this created a matrix from which a column, representing a movie, could be selected and sorted thus the most similar movies would be at the top of the column. 

### Popularity Score
A popularity score was calculated for each movie based on how many ratings it had received, these numbers where then scaled from 0 to 1. This "score" is then applied once the movie column had been selected from the matrix aka the target movie. The user can adjust the effect of the score via a slide that goes from -1 to +1, if the score closer to +1 more generically popular movies will be recommended if the slide is closer to -1 more obscure movies, that have fewer ratings will get recommended.


### Weighted Tags
Three different methods where implemented where tried and implemented allowing the user to switch between the methods.

- Standard
	No initial weights where added to the tags/genres, this worked well it gave more expected answers and it paired well with the popularity slider.

- TF
	This the Term Frequency version, it's not the exactly TF, but its similar, the point being it does not take in to account the tags in all documents/movies, it only care about the tags in the current document/movie. The more one type of tag a movie has the higher its impact on the recommendations. Because the geners are never more then one, there weight are adjusted manually so they don't get overshadowed by the other tags.
	These TF weights seem to produce recommendations where the movies are a little more obscure, similar to turning the popularity sliders in to the negative.

- TF-IDF
	This is the classic TF-IDF, it takes in to account the frequency of the tags/genres across all movies so that tags that are less frequent over all but more frequent in a few movies have greater impact than tags that are spread out across a lot of movies. It does seem to give a little more varied recommendations right out of the books but changing the popularity slider don't give that varied of a result, in a way it seems a little more stable.

Ultimately its hard to say which recommendations are better and which are worse as it by definition is subjective, does one want to be recommended popular movies that other people have liked or more obscure movies on the same theme. A solution to this is simply to give the users the tools so they them selves can adjust what type off recommendations they want to get.


### Tried and Discarded
-  Second Slider
	In addition to the popularity slide a ratings slide where also introduced so the user could adjust how much impact the ratings would have on the recommendations. But this seemed to produce the same result as popularity slider probably because there correlated, movies that get high ratings often become popular. This slider might also have had a greater effect if movies with low ratings where left in.

- Proportion Scheme
	The first thought was not to have a slider but to have scheme where each of the five movies recommended would have a different proportion of tag, popularity and rating applied to it so that one movie might be more heavily influenced by the popularity then tags. But giving the user more control seemed like a much beter solution and one would also have to explain more how each of the recommendations where selected.

| Nr  | Popularity | Tags | Rating |
| :-: | :--------: | :--: | :----: |
|  1  |    33%     | 33%  |  33%   |
|  2  |    70%     | 15%  |  15%   |
|  3  |    15%     | 70%  |  15%   |
|  4  |    15%     | 15%  |  70%   |
|  5  |    rng     | rng  |  rng   |

- Group Ratings
	A scheme where also considered where the most controversial movies would be selected aka the ones with the moster 5, 4.5, 2, 1.5 and 1 would have been selected and then used in combination with K Mean Clustering or perhaps a Decision Tree in order to group the users and then use the Cluster scores to give recommendations  but it was rejected for three reasons
	 1. It was needlessly complex 
	 2. the ratings where not balanced, it was hard to find movies 50/50 split between high and low
	 3. no movies where universally rated, the ratings where very fragmented
	This might still have worked but it wouldn't have been as easy as splitting group in to group in to group like a game of guess who.