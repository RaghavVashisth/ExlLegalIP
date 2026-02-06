import streamlit as st
import pandas as pd
import ast

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Similar Claims Explorer",
    layout="wide"
)

st.title("🔍 Similar Claims Explorer")

# ----------------------------------
# Load Data
# ----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("model_output_with_top5_claims.csv")

    # Convert string lists to actual Python lists
    df["top_5_similar_claim_ids"] = df["top_5_similar_claim_ids"].apply(ast.literal_eval)
    df["top_5_similarity_scores"] = df["top_5_similarity_scores"].apply(ast.literal_eval)

    return df

df = load_data()

# ----------------------------------
# Claim Selection
# ----------------------------------
selected_claim_id = st.selectbox(
    "Select Claim ID",
    sorted(df["claim_id"].unique())
)

selected_claim = df[df["claim_id"] == selected_claim_id].iloc[0]

# ----------------------------------
# Primary Claim Summary
# ----------------------------------
st.subheader("📄 Selected Claim Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Claim Type", selected_claim["Claim"])
c2.metric("State", selected_claim["FTR_JRSDTN_ST_ABBR"])
c3.metric("Injury", selected_claim["BODY_PART_INJD_DESC"])
c4.metric("Law Firm", selected_claim["Law Firm Name"])

st.markdown("**Claim Notes**")
st.info(selected_claim["Claim_Notes"])

# ----------------------------------
# Similar Claims Section
# ----------------------------------
st.subheader("🧠 Top 5 Similar Claims")

similar_ids = selected_claim["top_5_similar_claim_ids"]
similar_scores = selected_claim["top_5_similarity_scores"]

similar_claims = df[df["claim_id"].astype(str).isin(similar_ids)].copy()

# Map similarity scores
score_map = dict(zip(similar_ids, similar_scores))
similar_claims["Similarity Score"] = similar_claims["claim_id"].astype(str).map(score_map)

# Sort by similarity
similar_claims = similar_claims.sort_values(
    "Similarity Score",
    ascending=False
)

# ----------------------------------
# Display Similar Claims
# ----------------------------------
for _, row in similar_claims.iterrows():
    with st.container():
        st.markdown(
            f"""
            <div style="padding:15px; border-radius:12px; background-color:#f5f7fa; margin-bottom:15px;">
                <h4>Claim ID: {row['claim_id']} | Similarity: {row['Similarity Score']:.3f}</h4>
                <b>Claim Type:</b> {row['Claim']} <br>
                <b>State:</b> {row['FTR_JRSDTN_ST_ABBR']} <br>
                <b>Injury:</b> {row['BODY_PART_INJD_DESC']} <br>
                <b>Law Firm:</b> {row['Law Firm Name']} <br>
                <b>Match Scope:</b> {row['match_scope']} <br><br>
                <b>Claim Notes:</b><br>
                <span style="font-size:14px;">{row['Claim_Notes']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
