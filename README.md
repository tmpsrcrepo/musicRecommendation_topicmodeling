# musicRecommendation_topicmodeling
Personal project: music recommendation using topic modeling.

Since music is my only passion out of school, I like spending lots of time on reading music critics and listening to various albums. I also used those popular web music player, e.g. Spotify and iHeartRadio. However they recommend very limited songs and artists to me. If I want some out-of-box recommendations, then I have to do my own research. 

Therefore, I'm motivated to create a content-based recommendation algorithm which captures essense from music critics. Additionally this algorithm can pick similar songs beloning to different genres.

Basically the project consists of three components: 

1. A web scrpaer extracting reviews from Pitchfork.com (Scraping_Ultimate.py)

2. Retrieving contents from last.fm API (lastfm_scrape.py)

3. Applying topic modeling and TF/IDF on review contents (topic_model.py)

The algorithm is evaluated by a couple of music lovers. 

In the future, I'll collect more user data and combine this algorithm with collaborative filtering to better serve users' needs.

