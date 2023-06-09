import streamlit as st
import pandas as pd
import pickle
import requests
import numpy as np
import base64
st.title('Heart Disease Prediction')
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background.png')

#MODEL--------------------------------------------------------------------------------------------
df = pd.read_csv(r"D:\2.DS heart disease\heart_disease_data.csv")
y = df['target']
x = df.drop(['target'],axis=1)
import statsmodels.api as sm
reg_log = sm.Logit(y,x)
res = reg_log.fit()

#INPUT PARAMS-------------------------------------------------------------------------------------
age = st.number_input('Age', min_value=1, step=1)
sex = st.selectbox(
    'Sex',
    ['0.Female','1.Male'])
cp = st.selectbox(
    'Chest Pain Type',
    ['0.Asymptomatic','1.Atypical angina','2.Non-anginal pain','3.Typical angina'])
trestbps = st.number_input('Resting blood pressure(mm Hg)', min_value=0, step=1)
chol = st.number_input('Cholesterol(md/dl)', min_value=0, step=1)
fbs = st.number_input('Fasting blood sugar', min_value=0, step=1)
restecg = st.selectbox(
    'Resting electrocardiographic results',
    ['0.Left ventricular hypertrophy','1.Normal','3.ST-T wave abnormality'])
thalach = st.number_input('Maximum heart rate achieved', min_value=0, step=1)
exang = st.selectbox(
    'Exercise induced angina',
    ['0.No','1.Yes'])
oldpeak = st.number_input(' ST depression induced by exercise', min_value=0.0, step=0.1)
slope = st.selectbox(
    'Slope of the peak exercise ST segment',
    ['0.Downsloping','1.Flat','2.Upsloping'])
ca = st.number_input('Number of major vessels', min_value=0,max_value=3,step=1)
thal = st.selectbox(
    'Thalassemia',
    ['1.Fixed defect','2.Normal blood flow','3.Reversible defect'])

#PREDICTION-------------------------------------------------------------------------------------------------
sex = int(sex[0])
cp = int(cp[0])
restecg = int(restecg[0])
exang = int(exang[0])
slope = int(slope[0])
thal = int(thal[0])

ip = (age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
ip_data = np.asarray(ip)
if st.button('Predict'):
    st.write('Chances of heart disease: ',round(res.predict(ip_data)[0],2)*100,'%')