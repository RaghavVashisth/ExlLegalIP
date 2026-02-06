import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Page Config
# -------------------------
st.set_page_config("Law Firm Assignment", layout="wide")



@st.cache_data
def load_data():
    claims = pd.read_csv("model_output_with_predicted_cluster.csv")
    firms = pd.read_csv("synthetic_litigation_dataset_with_firms_and_cluster.csv")
    return claims, firms

claims_df, firms_df = load_data()



# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
.metric-card {
    background-color: #f8f9fa;
    padding: 16px;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
}
.title-card {
    background-color: #1f3c88;
    padding: 14px;
    border-radius: 8px;
    color: white;
}
.small-text {
    font-size: 13px;
    color: #555;
}
.badge {
    padding: 4px 10px;
    border-radius: 20px;
    background-color: #e6f0ff;
    display: inline-block;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.markdown(
    "<div class='title-card'><h3>Law Firm Assignment</h3></div>",
    unsafe_allow_html=True
)

# -------------------------
# Claim Context Bar
# -------------------------
c1, c2, c3, c4, c5 = st.columns([1,1,3,1,1])

with c1: 
        # -------------------------
    # Claim Selection
    # -------------------------
    claim_idx = st.selectbox(
        "Select Claim",
        claims_df.index,
        format_func=lambda x: f"Claim {x}"
    )

    claim = claims_df.loc[claim_idx]
    # st.selectbox("Claim Context", ["CLM-10245", "CLM-10246"])

with c2:
    st.metric("Jurisdiction", claim["FTR_JRSDTN_ST_ABBR"])
    # st.markdown("**State**")
    # st.write("TX")

with c3:
    st.metric("Injury Severity", claim["BODY_PART_INJD_DESC"])

    # st.markdown("**Injury**")
    # st.write("Severe Injury")

with c4:
    st.metric("Demand", claim["DEMAND"])

    # st.markdown("**Exposure**")
    # st.write("$250K - $500K")

with c5:
    st.metric("Offer", claim["OFFER"])
    # st.markdown("**Stage**")
    # st.write("OFFER")

st.divider()

# -------------------------
# Main Section
# -------------------------
left, right = st.columns([2.5, 1.5])

    
with left:
    st.subheader("Legal Market Segment Identification")

    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import numpy as np

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv("firm_level_cluster_map_output.csv")

    # Separate firms and centroids
    firms = df[df["entity_type"] == "firm"]
    centroids = df[df["entity_type"] == "centroid"]

    # -----------------------------
    # Cluster color mapping
    # -----------------------------
    cluster_colors = {
        "Efficient Volume Handlers": "#F4B400",      # yellow
        "Outcome Specialists": "#6A5ACD",            # purple
        "High-Value Core Firms": "#66C2A5",           # green
        "High-Cost/ Underperformers": "#5DA5DA"      # blue
    }

    # -----------------------------
    # Create plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    # -----------------------------
    # Scatter points (firms)
    # -----------------------------
    for cluster, color in cluster_colors.items():
        subset = firms[firms["cluster_id"] == cluster]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=50,
            alpha=0.8,
            color=color,
            label=cluster
        )

    # -----------------------------
    # Draw cluster ellipses
    # -----------------------------
    for _, row in centroids.iterrows():
        cluster = row["cluster_id"]
        color = cluster_colors.get(cluster, "#999999")

        cluster_points = firms[firms["cluster_id"] == cluster]

        x_mean = cluster_points["PC1"].mean()
        y_mean = cluster_points["PC2"].mean()
        x_std = cluster_points["PC1"].std()
        y_std = cluster_points["PC2"].std()

        ellipse = Ellipse(
            (x_mean, y_mean),
            width=4 * x_std,
            height=4 * y_std,
            facecolor=color,
            edgecolor=color,
            alpha=0.18
        )

        ax.add_patch(ellipse)

    # -----------------------------
    # Plot centroids
    # -----------------------------
    ax.scatter(
        centroids["PC1"],
        centroids["PC2"],
        color="black",
        s=120,
        marker="X",
        label="Cluster Centroid"
    )

# -----------------------------
# Highlight "This Claim" (dynamic firm)
# -----------------------------
    # a = "Phillips & Garcia, LLP"
    a = claim["Cluster_name"]

    this_claim = firms[firms["firm_name"] == claim["Firm Name"]]

    if not this_claim.empty:
        x = this_claim.iloc[0]["PC1"]
        y = this_claim.iloc[0]["PC2"]

        ax.scatter(
            x,
            y,
            s=300,
            facecolor="white",
            edgecolor="black",
            linewidth=2.5,
            zorder=6
        )

        ax.text(
            x,
            y + 0.35,
            "This Claim",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.4",
                fc="#4B0082",
                ec="none",
                alpha=0.9
            ),
            color="white",
            zorder=7
        )
    else:
        st.warning(f"Firm '{a}' not found in data.")






    # -----------------------------
    # Labels & styling
    # -----------------------------
    # ax.set_title("Legal Market Segment Identification\nClaim Clustering Map", fontsize=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax.axhline(0, linestyle="--", alpha=0.3)
    ax.axvline(0, linestyle="--", alpha=0.3)

    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    # plt.show()
    st.pyplot(fig)








# -------------------------
# Assigned Firm Cluster
# -------------------------
with right:
    # a = "Phillips & Garcia, LLP"

    Lit_data_firm_df = pd.read_csv("synthetic_litigation_dataset_with_firms_and_cluster.csv")
    Cluster_id = Lit_data_firm_df[Lit_data_firm_df["Cluster_name"] == a]["Cluster_name"].values[0]
    Avg_Cycle_Time = int(Lit_data_firm_df[Lit_data_firm_df["Cluster_name"] == Cluster_id]["Cycle time"].mean())
    Win_Rate = round(Lit_data_firm_df[Lit_data_firm_df["Cluster_name"] == Cluster_id]["Win rate proxy"].mean()*100,1)
    Avg_Cost = int(Lit_data_firm_df[Lit_data_firm_df["Cluster_name"] == Cluster_id]["Cost per case"].mean())


    st.markdown(f"""
    <div class="metric-card">
        <h4>Assigned Firm Cluster</h4>
        <h5>{Cluster_id}</h5>
        <hr>
        <b>Avg Cycle Time:</b> {Avg_Cycle_Time} days<br>
        <b>Win Rate:</b> {Win_Rate}%<br>
        <b>Avg Cost:</b> ${Avg_Cost}K<br>
        <hr>
        <b>Typical Claim Profile</b>
        <ul>
            <li>Complex Injury Cases</li>
            <li>Plaintiff-Friendly Venues</li>
        </ul>
        <span class="small-text">22
        Best suited for long-duration, high-exposure litigation.
        </span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# -------------------------
# Recommended Firms
# -------------------------
st.subheader("Recommended Firms (Within Cluster)")
col1, col2, col3 = st.columns(3)
with col1:
    cycle_time_weight = st.slider(
        "Cycle Time Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )
with col2:
    win_rate_weight = st.slider(
        "Win Rate Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )
with col3:
    Cost_per_case_weight = st.slider(
        "Cost Per Case Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )




Lit_data_firm_filter_df = Lit_data_firm_df[Lit_data_firm_df["Cluster_name"] == Cluster_id]
Lit_data_firm_filter_df.shape



state = claim["FTR_JRSDTN_ST_ABBR"]


Lit_data_firm_filter_df = Lit_data_firm_filter_df[
    Lit_data_firm_filter_df["state_list"].str.contains(
        fr"\b{state}\b", na=False
    )
]



Lit_data_firm_filter_df.shape


# ---- Compute weighted score ----
Lit_data_firm_filter_df['Weighted_Score'] = (
    cycle_time_weight * (10 - Lit_data_firm_filter_df['Cycle time']) +
    win_rate_weight * Lit_data_firm_filter_df['Win rate proxy'] +
    Cost_per_case_weight * (10 - Lit_data_firm_filter_df['Cost per case'])
)

top_firms = Lit_data_firm_filter_df.sort_values(
    "Weighted_Score", ascending=False
).head(3)

# ---- Layout ----
f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1])

# ---- Build firms list SAFELY ----
firms = []

for _, row in top_firms.iterrows():
    firms.append(
        (
            row["Firm Name"],
            row["state_list"],
            f"{round(row['Win rate proxy'] * 100, 1)}%",
            f"${int(row['Cost per case'])}K",
            f"{int(row['Cycle time'])} Days"
        )
    )

# ---- Guard: no firms found ----
if not firms:
    st.warning("No recommended firms found for the selected criteria.")
else:
    # ---- Render firm cards ----
    for col, firm in zip([f1, f2, f3], firms):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h5>{firm[0]} ✅</h5>
                    <span class="small-text">{state}</span>
                    <hr>
                    <b>Win Rate:</b> {firm[2]}<br>
                    <b>Avg Cost:</b> {firm[3]}<br>
                    <b>Cycle Time:</b> {firm[4]}<br>
                    <b>Stage:</b> Discovery
                </div>
                """,
                unsafe_allow_html=True
            )
# -------------------------
# Decision Support Summary
# -------------------------
with f4:
    st.markdown("""
    <div class="metric-card">
        <h5>Decision Support Summary</h5>
        <b>Model Confidence</b>
    </div>
    """, unsafe_allow_html=True)

    st.progress(0.84)

    st.markdown("""
    <span class="small-text">
    Decision support only.<br>
    Final assignment at adjuster discretion.
    </span>
    """, unsafe_allow_html=True)

    st.radio(
        "Optimization Strategy",
        ["Outcome Focused", "Cost Focused"],
        horizontal=True
    )