import streamlit as st
import numpy as np
import sklearn
import pickle


@st.cache
def load_model(file):
    model = pickle.load(open(file,"rb"))
    return model

lrmodel = load_model('model.pkl')



st.title('Simple Linear Regression')
st.write('Experience Vs Salary')



exp=st.number_input("Enter experience in years",min_value=0.0,max_value=50.0,step=0.5)
prediction = lrmodel.predict([[exp]])
salary = np.round(prediction[0], 2)
status=st.button("Click to See Salary")
if status:
	st.write("Predicted Salary for " + str(exp) + " years is Rs "+str(salary))





        


