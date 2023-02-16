import streamlit as st
import movies_recommender as mr
recommender = mr.movies_recommender()
#movies_df = pd.DataFrame()
st.header("MOVIES RECOMMENDER")
st.caption("Welcome!, choose movie genres you would prefer...") 
options = st.multiselect('What are your favourite movie genres: ',
                         ['action', 'adventure', 'animation', 'childrens', 'comedy', 'crime',
                          'documentary', 'drama', 'fantasy', 'horror', 'mystery', 'romance', 'scifi', 'thriller'],
                         ['action', 'adventure'])
if len(options) > 0:
    for option in options:
        rating = st.slider(option,0.0,5.0,0.0,0.5)
        recommender.set_user_data(option,rating)
    
IsClick = st.button("Generate Movies","click the button to get recommended movies")
if IsClick:
    IsClick = False
    movies_df = recommender.get_recommended_movies()
    st.table(movies_df[['title','genres']].style.set_properties(['title'],**{'color':'blue'}))