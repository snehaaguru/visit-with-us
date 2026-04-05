

import streamlit as st
import pandas as pd, pickle, json
from huggingface_hub import hf_hub_download


HF_USERNAME = "snehaguru"
MODEL_REPO   = f"{HF_USERNAME}/visit-with-us-model"

@st.cache_resource
def load_model():
    m  = hf_hub_download(repo_id=MODEL_REPO, filename="best_model.pkl")
    fn = hf_hub_download(repo_id=MODEL_REPO, filename="feature_names.json")
    with open(m,  "rb") as f: model    = pickle.load(f)
    with open(fn, "r")  as f: features = json.load(f)
    return model, features

model, feature_names = load_model()


occupation_map  = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
designation_map = {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}
marital_map     = {"Divorced": 0, "Married": 1, "Single": 2, "Unmarried": 3}
product_map     = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}
gender_map      = {"Female": 0, "Male": 1}
contact_map     = {"Company Invited": 0, "Self Inquiry": 1}


st.sidebar.title("Visit With Us")

st.title("Wellness Tourism Package Purchase Predictor")
st.markdown("Enter customer details below to predict purchase likelihood.")

col1, col2, col3 = st.columns(3)

with col1:
    age            = st.number_input("Age",value=35)
    monthly_income = st.number_input("Monthly Income", value=40000)
    num_trips      = st.number_input("Number of Trips/Year", value= 3)
    city_tier      = st.selectbox("City Tier", [1, 2, 3])
    passport       = st.selectbox("Has Passport?", ["No", "Yes"])

with col2:
    occupation     = st.selectbox("Occupation",     list(occupation_map.keys()))
    gender         = st.selectbox("Gender",         list(gender_map.keys()))
    marital_status = st.selectbox("Marital Status", list(marital_map.keys()))
    designation    = st.selectbox("Designation",    list(designation_map.keys()))
    own_car        = st.selectbox("Owns a Car?",    ["No", "Yes"])

with col3:
    type_contact   = st.selectbox("Type of Contact",  list(contact_map.keys()))
    product_pitched= st.selectbox("Product Pitched",  list(product_map.keys()))
    pitch_score    = st.slider("Pitch Satisfaction Score", value= 3)
    num_followups  = st.slider("Number of Follow-ups", value= 3)
    duration_pitch = st.number_input("Duration of Pitch (mins)", value= 20)
    num_persons    = st.number_input("Persons Visiting", value= 2)
    num_children   = st.number_input("Children (<5) Visiting", value= 0)
    property_star  = st.selectbox("Preferred Property Star", [3, 4, 5])

if st.button("Predict Purchase"):


    input_dict = {
        "Age"                     : age,
        "TypeofContact"           : contact_map[type_contact],
        "CityTier"                : city_tier,
        "DurationOfPitch"         : duration_pitch,
        "Occupation"              : occupation_map[occupation],
        "Gender"                  : gender_map[gender],
        "NumberOfPersonVisiting"  : num_persons,
        "NumberOfFollowups"       : num_followups,
        "ProductPitched"          : product_map[product_pitched],
        "PreferredPropertyStar"   : property_star,
        "MaritalStatus"           : marital_map[marital_status],
        "NumberOfTrips"           : num_trips,
        "Passport"                : 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore"  : pitch_score,
        "OwnCar"                  : 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children,
        "Designation"             : designation_map[designation],
        "MonthlyIncome"           : monthly_income,
    }

    input_df   = pd.DataFrame([input_dict])[feature_names]
    prediction = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Customer is LIKELY to purchase the Wellness Package! (Probability: {proba:.2%})")
    else:
        st.warning(f"❌ Customer is UNLIKELY to purchase. (Probability: {proba:.2%})")
