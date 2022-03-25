from secrets import choice
from cv2 import threshold
import pickle
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

from math import sqrt
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import scipy
from numpy import dot
from numpy.linalg import norm

# 1. Read data
df_product = pd.read_csv('Product_new.csv',encoding = 'utf-8')
df_review = pd.read_csv('Review_new.csv',encoding = 'utf8', engine = 'python')

# 2. Data pre-processing




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
    st.subheader('Select data')


