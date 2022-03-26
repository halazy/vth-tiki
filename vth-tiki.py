from secrets import choice
from cv2 import threshold
import streamlit as st
import numpy as np
import pandas as pd
import pandas_profiling as pp
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from underthesea import word_tokenize, pos_tag, sent_tokenize
from gensim import corpora, models, similarities
import jieba
import re
import regex
from wordcloud import WordCloud

from math import sqrt
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import scipy
from numpy import dot
from numpy.linalg import norm

import pickle


# 1. Read data

# def data_loading():
#     # read product data
#     df = pd.read_csv('Product_new.csv')
#     df = df.drop_duplicates()
#     df = df.dropna()
#     df.reset_index(inplace=True)

#     # create a stop words list
#     with open('resources/vietnamese-stopwords.txt', 'r', encoding='utf-8') as file:
#         stop_words = file.read()
#         stop_words = stop_words.split('\n')
#     for word in ['thông_tin', 'chi_tiết', 'sản_phẩm', 'tiki']:
#         stop_words.append(word)

#     # overall preprocessing
#     for col in ['name', 'description', 'brand', 'group']:
#       df[col] = df[col].apply(lambda x: str(x).lower().replace('_', ' ').replace('\n', ' ').replace('.', ' '))

#     # concate the information
#     df['products'] = df.name + ' ' + df.description + ' ' + df.brand + ' ' + df.group

#     # tokenize words
#     df['products_wt'] = df['products'].apply(lambda x: word_tokenize(x, format='text'))

#     return df, stop_words

# def content_model():

#     df, stop_words = data_loading()

#     # Vectorize data
#     tf = TfidfVectorizer(analyzer='word', min_df=0.1, stop_words=stop_words)
#     tfidf_matrix = tf.fit_transform(df.products_wt)
#     cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#     return df, cosine_sim

# def content_model_predict(product_list, top_n=5):
#     cosine_sim = np.load("resources/cosine_sim.dat", allow_pickle=True)
#     df = pd.read_csv('Product_new.csv')
#     df = df.drop_duplicates()
#     df = df.dropna()
#     df.reset_index(inplace=True)
#     indices = pd.Series(df['name'])

#     # Getting the index of the products that matches the name
#     idx_1 = indices[indices == product_list[0]].index[0]
#     idx_2 = indices[indices == product_list[1]].index[0]
#     idx_3 = indices[indices == product_list[2]].index[0]

#     # Creating a Series with the similarity scores in descending order
#     rank_1 = cosine_sim[idx_1]
#     rank_2 = cosine_sim[idx_2]
#     rank_3 = cosine_sim[idx_3]
#     # Calculating the scores
#     score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
#     score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
#     score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
#     # Getting the indexes of the 10 most similar products
#     listings = score_series_1.append(score_series_1).append(score_series_3).\
#                sort_values(ascending = False)

#     # Store products' names
#     recommended_products = []
#     # Appending the names of movies
#     top_50_indexes = list(listings.iloc[1:50].index)
#     # Removing chosen movies
#     top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
#     for i in top_indexes[:top_n]:
#         recommended_products.append(list(df['name'])[i])

#     return recommended_products


# def collab_model_predict(user_id):
#     df_product = pd.read_csv('Product_new.csv')
#     user_recs = pd.read_parquet("resources/CF_recommendations.parquet", engine='pyarrow')
#     id = user_recs[user_recs.customer_id == user_id].index[0]
#     recommended_products_id = re.findall('product_id\'\:\s(\d+)',
#                                          str(user_recs['recommendations'][id]))

#     recommended_products = []
#     for i in recommended_products_id:
#         recommended_products.append(df_product['name'][df_product.item_id == int(i)].item())
#     return recommended_products    

df_product = pd.read_csv('Product_new.csv',encoding = 'utf-8')
df_review = pd.read_csv('Review_new.csv',encoding = 'utf8', engine = 'python')

# 2. Data pre-processing
# Content-based filtering
# chuan hoa text
STOP_WORD_FILE = "vietnamese-stopwords.txt"
with open(STOP_WORD_FILE, 'r', encoding = 'utf-8') as file:
  stop_words = file.read()
  
stop_words = stop_words.split('\n')

# Chuan hoa unicode
import regex as re
 
uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
 
 
def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
 
 
dicchar = loaddicchar()
 
 
def covert_unicode(txt):
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)


def chuan_hoa_dau_cau_tieng_viet(sentence):
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
        # print(cw)
        if len(cw) == 3:
            cw[1] = chuan_hoa_dau_tu_tieng_viet(cw[1])
        words[index] = ''.join(cw)
    return ' '.join(words)

# Loại bỏ NaN trong description
df_product = df_product[~df_product['description'].isnull()].reset_index()
df_product = df_product.drop('index', axis=1)

df_product['recommend'] = df_product['name'] + df_product['description']

# Cosine-similarity
X =  [1,2]
Y = [2,2]
cos_sim = dot(X, Y)/(norm(X)*norm(Y))
tf = TfidfVectorizer(analyzer="word", min_df = 0, stop_words = stop_words)
tfidf_matrix = tf.fit_transform(df_product['recommend'].values.astype('U'))
cosine_similarities = cosine_similarity(tfidf_matrix,tfidf_matrix)
# với mỗi sản phẩm, lấy 10 sản phẩm tương quan nhất
results = {}

for idx, row in df_product.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-11:-1] #argsort() là sắp xếp tăng dần
    similar_items = [(cosine_similarities[idx][i], df_product['item_id'][i]) for i in similar_indices]
    results[row['item_id']] = similar_items[1:] #bỏ đi phần tử 0 là chính nó

# Xây dựng hàm lấy thông tin sản phẩm
def item(product_id):
    return df_product.loc[df_product['item_id'] == product_id]['name'].tolist()[0].split('-')[0]

def recommend(product_id, num):
    print('Recommending products similar to ' + item(product_id) + ':')
    recs = results[product_id][:num]
    for rec in recs:
        print(rec[1])
        print('Recommend: product id:' + str(rec[1]) +', ' + item(rec[1]) + ' (score:' + str(rec[0]) + ")")  


# Save model
info = []
for p_id, v in results.items():
    for item in v:
        info.append({
            'product_id': p_id,
            'rcmd_product_id': item[1],
            'score': item[0]
        })
content_based_cosine_df = pd.DataFrame(info)

content_based_cosine_df.to_csv('Content_based_cosine_info.csv')


#--------------
# GUI
st.title("Data Science Project")
st.header("Tiki Products and Reviews Recommendation")

# GUI
menu = ['Business Objective', 'Build Project', 'Recommender']
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':
    st.title("I. Tổng quan về Recommendation system")
    st.image("DS.jpeg")
    st.write("""
    Là một subclass của information filtering system tìm cách dự đoán **"xếp hạng"** hoặc **"ưu tiên"** mà người dùng sẽ dành cho một mục.
    """)  
    st.write(""" **Recommender system** thực sự quan trọng trong một số lĩnh vực vì chúng có thể tạo ra một khoản thu nhập khổng lồ hoặc cũng là một cách để nổi bật đáng kể so với các đối thủ cạnh tranh.
    """)
    st.write("""Có hai recommender system phổ biến nhất là **Collaborative Filtering** (CF) và **Content-Based**""")
    st.image("recommendation-overview.jpeg")
    st.title("II. Bussiness Objective")
    st.write("""##### Tiki là một hệ sinh thái thương mại “all in one”, trong đó có tiki.vn, là một website thương mại điện tử đứng top 2 của Việt Nam, top 6 khu vực Đông Nam Á.""")
    st.write("""###### Trên trang này đã triển khai nhiều tiện ích hỗ trợ nâng cao trải nghiệm người dùng và họ muốn xây dựng nhiều tiện ích hơn nữa. Bài toán được đưa ra là: Xây dựng Recommendation System cho một hoặc một số nhóm hàng hoá trên tiki.vn giúp đề xuất và gợi ý cho khách hàng trong quá trình lựa chọn sản phẩm""")
    st.title("III. Đề xuất phương pháp")
    st.write("""- Hệ thống gợi ý dựa trên nội dung - Content based recommender systems: tức là hệ thống sẽ quan tâm đến nội dung, đặc điểm của mục tin hiện tại và sau đó gợi ý cho người dùng các mục tin tương tự. Đó chính là trường hợp thứ nhất""")
    st.write("""- Hệ thống gợi ý dựa trên các user - lọc cộng tác - Collaborative filtering recommender systems: tức là hệ thống sẽ phân tích các user có cùng đánh giá, cùng mua mục tin hiện tại. Sau đó tìm ra danh sách các mục tin khác cũng được đánh gía bởi các user này, xếp hạng và gợi ý cho người dùng. Tư tưởng của phương pháp này chính là dựa trên sự tương đồng về sở thích giữa các người dùng để đưa ra các gợi ý.
    """)
    st.image("tiki.jpeg")

elif choice == 'Build Project':
    st.title('Build Project')
    st.subheader('I. Data Overview')
    st.write('Product')
    st.dataframe(df_product.head())
    st.write('Review')
    st.dataframe(df_review.head())

    st.subheader('II. Visualization')
    st.write('Product')
    fig1, ax = plt.subplots(1,2, figsize=(15,8))
    df_product.list_price.plot(kind='box', ax=ax[0])
    df_product.list_price.plot(kind='hist', bins=20, ax=ax[1])
    st.pyplot(fig1)

    st.write('Review')
    fig2, ax = plt.subplots(1,2, figsize=(15,8))
    df_review.rating.plot(kind='box', ax=ax[0])
    df_review.rating.plot(kind='hist', bins=20, ax=ax[1])
    st.pyplot(fig2)

    st.write('Top 20 sản phẩm được khách hàng đánh giá nhiều nhất')
    st.image('20_product.png')
    st.write('Top 20 khách hàng đánh giá nhiều nhất')
    st.image('20_customers.png')
    st.write('Top 10 thương hiệu có nhiều sản phẩm nhất')
    st.image('number_product_by_brand.png')
    st.write('Giá trung bình xếp theo thương hiệu')
    st.image('average_price.png')

    st.subheader('III. Build Model')
    st.write('1. Gensim')
    st.write('Các bước thực hiện:')
    st.write("""- Tokensize (tách) các câu thành các từ.""")
    st.write("""- Xoá những ký tự đặc biệt.""")
    st.write("""- Sử dụng Mô hình TF-IDF để xử lý kho dữ liệu.""")
    st.write("""- Xây dựng hàm dự đoán sản phẩm tương tự.""")
    st.write("""- Đối với mỗi sản phẩm, chúng ta có thể nhận được một số sản phẩm tương tự.""")

    st.write('2. Cosine Similarity')
    st.write('Các bước thực hiện:')
    st.write("""- Sử dụng thư viện underthesea để chuẩn hoá văn bản.""")
    st.write("""- Sử dụng Mô hình TF-IDF để xử lý kho dữ liệu.""")
    st.write("""- Xây dựng hàm lấy thông tin sản phẩm tương tự, bao gồm: ID sản phẩm, tên sản phẩm, score sản phẩm.""")
    st.write("""- Trực quan hoá bằng Word-Cloud cho sản phẩm và các sản phẩm tương tự.""")
    st.image("word-cloud.png")

    st.write('3. ALS Model')
    st.write('Các bước thực hiện:')
    st.write("""- Tính sparsity, chia tệp train, test.""")
    st.write("""- Tạo ALS model.""")
    st.write("""- Tính RMSE và đánh giá model.""")
    st.write("""- Cải thiện RMSE (nếu có).""")
    st.write("""- Đề xuất sản phẩm tương tự dựa trên người dùng tương tự, lựa chọn sản phẩm có rating >= 4.""")

    st.write('4. SVD Model')
    st.write('Các bước thực hiện:')
    st.write("""- Chuyển dữ liệu thành dataframe.""")
    st.write("""- Lọc ra những khách hàng có đánh giá tối thiểu 5 sản phẩm trở lên và các sản phẩm có đánh giá bởi ít nhất 100 khách hàng.""")
    st.write("""- Tạo ma trận, sử dụng reduced_SVD.""")
    st.write("""- Đề xuất sản phẩm tương tự dựa trên người dùng tương tự, lựa chọn 5 sản phẩm liên quan nhất.""")

    st.subheader('IV. Model Evaluation')
    st.write("""- Đối với dự đoán dựa trên Content:  Lựa chọn Cosine-Similarity cho kết quả tốt hơn.""")
    st.write("""- Đối với dự đoán dựa trên User:  Lựa chọn ALS Model cho kết quả khá tốt với RMSE = 1.11 ở mức chấp nhận được.""")
    

elif choice == 'Recommender':
    st.subheader('Make new Prediction')

    name1 = st.text_input('Customer ID (6177374, 1827148...)')
    if name1!="":
        recommend = content_model_predict(name1)
    st.write("## We think you'd like to see more of these items:")
    for i,j in enumerate(recommend):
        st.write(str(i+1)+'. '+j)
    name2 = st.text_input('Product ID (3792857, 1060082...)')  
    pclass  = st.selectbox("Product Keywords", options=['Tivi', 'Tủ lạnh','Loa','Laptop','Camera','Khác'])
    st.write('#### Suggested Products')



    # st.write('## Please Input Product ID:')
    # chosen_product = st.text_input('Product ID')

    # if st.button("Recommend (CB)"):
    #     try:
    #         with st.spinner('Crunching the numbers...'):
    #             top_recommendations = content_model_predict(chosen_product)
    #         st.write("## We think you'd like to see more of these items:")
    #         for i,j in enumerate(top_recommendations):
    #             st.write(str(i+1)+'. '+j)
    #     except:
    #         st.error("Oops! Looks like this algorithm does't work.\
    #                     We'll need to fix it!")

    # st.write('## Please Input User ID:')
    # chosen_user = st.text_input('User ID')

    # if st.button("See Profile"):
    #     try:
    #         with st.spinner('Loading...'):
    #             user_name = load_user_name(chosen_user)
    #             reviewed_products, rating = load_user_reviews(chosen_user)
    #         st.write('User Name: ' + user_name)
    #         for i,j in enumerate(reviewed_products):
    #             st.write(str(i+1) + '. ' + j + ' | Given rating: '+ str(rating[i]))
    #     except:
    #         st.error("Oops! Looks like something does't work.\
    #                         Please choose another one.")

    # if st.button("Recommend (UB)"):
    #     try:
    #         with st.spinner('Crunching the numbers...'):
    #             top_recommendations = collab_model_predict(user_id)
    #         st.write("## We think you'd like to see more of these items:")
    #         for i,j in enumerate(top_recommendations):
    #             st.write(str(i+1)+'. '+j)
    #     except:
    #         st.error("Oops! Looks like this algorithm does't work.\
    #                         We'll need to fix it!")








