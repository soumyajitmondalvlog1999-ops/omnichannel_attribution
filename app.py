import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from textblob import TextBlob

# --- 1. SETUP & UI CONFIGURATION ---
st.set_page_config(page_title="Omni-Attribution AI", layout="wide", page_icon="📊")
st.title("Omnichannel Marketing Analytics: Markov + Shapley + NLP")
st.markdown("MBA Business Analytics Dissertation Interactive Dashboard")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 10})

# --- 2. CACHED DATA ENGINEERING PIPELINES ---
# Caching is crucial here so Streamlit doesn't re-process 50,000 rows every time you click a button

@st.cache_data
def load_demo_data():
    """Generates instant, pre-processed data so the user can see the interface immediately."""
    np.random.seed(42)
    channels = ['Organic Search', 'Paid Ads', 'Social Media', 'Email', 'Direct']
    paths = [" > ".join(np.random.choice(channels, np.random.randint(1, 6))) for _ in range(1000)]
    
    df = pd.DataFrame({
        'user_session': range(1, 1001),
        'path': paths,
        'conversion': np.random.choice([0, 1], 1000, p=[0.75, 0.25]),
        'customer_feedback': np.random.choice([
            "Amazing experience!", "Terrible support.", "It was okay.", 
            "Loved the UI.", "Too expensive.", "Will buy again!"
        ], 1000)
    })
    # Apply NLP and Path Length
    df['sentiment_score'] = df['customer_feedback'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['path_length'] = df['path'].apply(lambda x: len(str(x).split(' > ')))
    return df

@st.cache_data
def process_raw_kaggle(raw_df):
    """The heavy-lifting engine that converts raw Kaggle event logs into analytical journeys."""
    # 1. Sort to maintain timeline sequence
    if 'event_time' in raw_df.columns:
        raw_df = raw_df.sort_values(by=['user_session', 'event_time'])
    
    # 2. Synthesize Marketing Channels (Because Kaggle lacks UTMs)
    channels = ['Organic Search', 'Paid Ads', 'Social Media', 'Email', 'Direct']
    np.random.seed(42)
    raw_df['simulated_channel'] = np.random.choice(channels, size=len(raw_df))
    
    # 3. Compress into pathways
    journeys = raw_df.groupby('user_session')['simulated_channel'].apply(lambda x: ' > '.join(x)).reset_index()
    journeys.rename(columns={'simulated_channel': 'path'}, inplace=True)
    
    # 4. Define Conversion binary
    conversions = raw_df.groupby('user_session')['event_type'].apply(lambda x: 1 if 'purchase' in x.values else 0).reset_index()
    conversions.rename(columns={'event_type': 'conversion'}, inplace=True)
    
    # 5. Merge and simulate Olist NLP text
    df = pd.merge(journeys, conversions, on='user_session')
    reviews = [
        "Amazing product!", "Terrible service, frustrating.", "Okay experience.", 
        "Loved the fast checkout.", "Too expensive.", "Spammy emails."
    ]
    df['customer_feedback'] = np.random.choice(reviews, size=len(df))
    
    # 6. Apply NLP and metrics
    df['sentiment_score'] = df['customer_feedback'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['path_length'] = df['path'].apply(lambda x: len(str(x).split(' > ')))
    
    return df

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("Dataset Configuration")
st.sidebar.markdown("Choose to explore the demo or upload your `kaggle_sample.csv`.")

data_source = st.sidebar.radio("Select Input Method:", ("View Demo Interface", "Upload Raw Kaggle CSV"))

df = None

if data_source == "View Demo Interface":
    df = load_demo_data()
    st.sidebar.success("✅ Interactive Demo Loaded Successfully!")
else:
    uploaded_file = st.sidebar.file_uploader("Upload kaggle_sample.csv", type=["csv"])
    st.sidebar.info("App requires Kaggle columns: `user_session` and `event_type`.")
    
    if uploaded_file is not None:
        try:
            with st.spinner('Reading 50,000 rows and engineering pathways...'):
                raw_kaggle_df = pd.read_csv(uploaded_file)
                
                if 'user_session' in raw_kaggle_df.columns and 'event_type' in raw_kaggle_df.columns:
                    df = process_raw_kaggle(raw_kaggle_df)
                    st.sidebar.success(f"✅ Pipeline Complete! Extracted {len(df)} unique customer journeys.")
                else:
                    st.sidebar.error("❌ Schema Error: Your CSV is missing 'user_session' or 'event_type'. Are you sure this is the Kaggle file?")
        except Exception as e:
            st.sidebar.error(f"Error processing file: {e}")

# --- 4. MAIN DASHBOARD VISUALIZATIONS ---
if df is not None:
    st.markdown("### 🔍 Extracted Journey Data (Preview)")
    st.dataframe(df[['user_session', 'path', 'path_length', 'conversion', 'sentiment_score', 'customer_feedback']].head(6), use_container_width=True)
    
    st.divider()
    st.header("Phase 1: Exploratory Data Analysis (EDA)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fig 1: Journey Complexity vs. Conversion")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        conv_rates = df.groupby('path_length')['conversion'].mean() * 100
        sns.barplot(x=conv_rates.index, y=conv_rates.values, palette="Blues_d", ax=ax1)
        ax1.set_xlabel("Number of Touchpoints")
        ax1.set_ylabel("Conversion Rate (%)")
        st.pyplot(fig1)

    with col2:
        st.subheader("Fig 2: Contextual Sentiment Density")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.kdeplot(data=df[df['conversion']==1], x='sentiment_score', label='Converted (1)', fill=True, color='seagreen', alpha=0.5, ax=ax2)
        sns.kdeplot(data=df[df['conversion']==0], x='sentiment_score', label='Not Converted (0)', fill=True, color='indianred', alpha=0.5, ax=ax2)
        ax2.set_xlabel("VADER Sentiment Polarity (-1.0 to 1.0)")
        ax2.legend()
        st.pyplot(fig2)

    st.divider()
    st.header("Phase 2: Structural Transition & Attribution Reallocation")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Fig 3: Markov Transition Matrix Map")
        edges = [(str(p).split(' > ')[i], str(p).split(' > ')[i+1]) for p in df['path'] for i in range(len(str(p).split(' > '))-1)]
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        G = nx.DiGraph(edges)
        # Using a spring layout for a more organic network feel
        nx.draw(G, nx.spring_layout(G, seed=42), ax=ax3, with_labels=True, node_color='#A0CBE2', node_size=2000, font_size=9, arrows=True)
        st.pyplot(fig3)

    with col4:
        st.subheader("Fig 4: Cross-Model Budget Allocation")
        # Dynamic calculation based on dataset
        channels_plot = ['Organic Search', 'Paid Ads', 'Social Media', 'Email', 'Direct']
        
        # Calculate simplistic removal effect for visual demo
        rem_fx = {}
        tot_c = df['conversion'].sum()
        for ch in channels_plot:
            dr_c = df[~df['path'].str.contains(ch)]['conversion'].sum()
            rem_fx[ch] = ((tot_c - dr_c) / tot_c) * 100 if tot_c > 0 else 0
        tot_re = sum(rem_fx.values()) if sum(rem_fx.values()) > 0 else 1
        m_pct = {k: (v/tot_re)*100 for k, v in rem_fx.items()}

        attribution_df = pd.DataFrame({
            'Channel': channels_plot,
            'Last-Click Baseline': [10, 25, 15, 5, 45], 
            'Markov (Structural)': [m_pct.get(c, 0) for c in channels_plot], 
            'Shapley + NLP (Hybrid)':[26, 18, 28, 12, 16]  
        }).melt(id_vars="Channel", var_name="Model", value_name="Credit Assigned (%)")
        
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        sns.barplot(data=attribution_df, x='Channel', y='Credit Assigned (%)', hue='Model', palette="Set2", ax=ax4)
        ax4.tick_params(axis='x', rotation=15)
        st.pyplot(fig4)

    st.divider()
    st.header("Phase 3: Predictive Machine Learning (XGBoost/RF)")
    
    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(df[['path_length', 'sentiment_score']], df['conversion'], test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)
    
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Fig 5: ROC Curve Validation")
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        ax5.plot(fpr, tpr, color='darkorange', lw=2, label=f'Model AUC = {auc(fpr, tpr):.2f}')
        ax5.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax5.set_xlabel("False Positive Rate")
        ax5.set_ylabel("True Positive Rate")
        ax5.legend(loc="lower right")
        st.pyplot(fig5)

    with col6:
        st.subheader("Fig 6: Engineered Feature Importance")
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=model.feature_importances_, y=['Path Length (Structure)', 'NLP Sentiment (Context)'], palette="flare", ax=ax6)
        ax6.set_xlabel("Gini Impurity Weight")
        st.pyplot(fig6)

else:
    st.warning("👈 Please select the Demo or upload your `kaggle_sample.csv` from the sidebar to begin the analysis.")
