import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
# 读取数据
with open('电影推荐/filmratings.json', 'r') as f:
    data = json.load(f)
# 转换为DataFrame
ratings_df = pd.DataFrame(data).T.fillna(0)
ratings_matrix = ratings_df.fillna(0)
similarity_matrix = cosine_similarity(ratings_matrix)
# 推荐函数
def collaborative_filtering_predict(user, ratings_matrix, similarity_matrix, n_recommendations=3):
    user_index = ratings_matrix.index.tolist().index(user)
    similar_users = list(enumerate(similarity_matrix[user_index]))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:]  # 排除自己
    similar_user_indices = [i[0] for i in similar_users]
    # 获取推荐电影
    recommendations = {}
    for idx in similar_user_indices:
        for movie, rating in ratings_matrix.iloc[idx].items():
            if ratings_matrix.loc[user, movie] == 0:  # 未评分电影
                if movie not in recommendations:
                    recommendations[movie] = 0
                recommendations[movie] += rating * similarity_matrix[user_index][idx]
    recommended_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    return recommended_movies
def content_based_recommend(user, ratings_matrix, n_recommendations=3):
    user_ratings = ratings_matrix.loc[user]
    rated_movies = user_ratings[user_ratings > 0].index.tolist()
    # 计算用户的平均评分
    user_mean_rating = user_ratings.mean()
    recommendations = {}
    for movie in ratings_matrix.columns:
        if movie not in rated_movies:
            # 计算基于均值的推荐
            recommendations[movie] = ratings_matrix[movie].mean() - user_mean_rating
    recommended_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    return recommended_movies
# 流行度推荐
def popularity_recommend(n_recommendations=3):
    # 计算每部电影的平均评分
    movie_means = ratings_matrix.mean().sort_values(ascending=False)
    return list(movie_means.items())[:n_recommendations]
# 页面配置
st.set_page_config(
    page_title="电影推荐系统",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)
# 添加 CSS 来调整字体样式
st.markdown(
    """  
    <style>  
    .big-font {  
        font-size: 20px !important;  /* 设置大字体 */  
    }  
    .header-font {  
        font-size: 35px; /* 主标题字体大小 */  
        font-weight: bold;  
    }  
    </style>  
    """,
    unsafe_allow_html=True
)

# 标题
st.markdown("<h1 class='header-font'>🎬 电影推荐系统</h1>", unsafe_allow_html=True)
st.markdown("<p class='big-font'>欢迎使用我们的电影推荐系统，基于协同过滤和内容推荐算法为您提供最佳观影建议！</p>", unsafe_allow_html=True)
# 侧边栏
st.sidebar.header("用户操作")
user_input = st.sidebar.selectbox("选择用户", ratings_df.index.tolist())
# 显示推荐
def display_recommendations(recommendations, title):
    st.subheader(title)
    if recommendations:
        for movie, score in recommendations:
            color = "green" if score > 0 else "red"
            st.markdown(f"<div class='big-font' style='color: {color};'>{movie}: **{score:.2f}**</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='big-font' style='color: red;'>没有推荐可用。</div>", unsafe_allow_html=True)

if st.sidebar.button("生成推荐"):
    collaborative_recommendations = collaborative_filtering_predict(user_input, ratings_matrix, similarity_matrix)
    content_recommendations = content_based_recommend(user_input, ratings_matrix)
    # 显示协同过滤推荐和基于内容的推荐
    display_recommendations(collaborative_recommendations, "🤝 协同过滤推荐:")
    display_recommendations(content_recommendations, "📖 基于内容的推荐:")
    # 如果没有推荐，则使用流行度推荐
    if not collaborative_recommendations and not content_recommendations:
        st.subheader("🔄 推荐电影（基于流行度）:")
        popularity_recommendations = popularity_recommend()
        display_recommendations(popularity_recommendations, "🎉 流行度推荐:")