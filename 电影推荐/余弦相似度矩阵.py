import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# 读取数据
with open('filmratings.json', 'r') as f:
    data = json.load(f)

# 转换为DataFrame
ratings_df = pd.DataFrame(data).T.fillna(0)
ratings_matrix = ratings_df.fillna(0)

# 计算余弦相似度
similarity_matrix = cosine_similarity(ratings_matrix)

# 将相似度矩阵转换为 DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=ratings_matrix.index, columns=ratings_matrix.index)

# 创建热图并保存
plt.figure(figsize=(10, 8))  # 调整图形大小
sns.heatmap(similarity_df, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
plt.title('Cosine Similarity Matrix')
plt.xlabel('Users')
plt.ylabel('Users')

# 保存为图片
plt.savefig('cosine_similarity_matrix.png', bbox_inches='tight', dpi=300)  # 可以修改文件名和参数
plt.close()  # 关闭图形以释放资源

print("余弦相似度矩阵已保存为 'cosine_similarity_matrix.png'")