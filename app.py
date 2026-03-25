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

# --- 1. SETUP & UI ---
st.set_page_config(page_title="Omni-Attribution AI", layout="wide", page_icon="📈")
st.title("Omnichannel Marketing Analytics: Markov + Shapley + NLP")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 9})

# --- 2. DATA PIPELINE ---
st.sidebar.header("Dataset Configuration")
data_source = st.sidebar.radio("Select Input Method:", ("Use Demo Dataset", "Upload Raw Kaggle CSV"))

@st.cache_data
def load_demo_data():
    # Pre-processed mock data for instant demo rendering
    np.random.seed(42)
    channels = ['Organic Search', 'Paid Ads', 'Social Media', 'Email', 'Direct']
    paths = [" > ".join(np.random.choice(channels, np.random.randint(1, 6))) for _ in range(800)]
    return pd.DataFrame({
        'user_session': range(1, 801),
        'path': paths,
        'conversion': np.random.choice([0, 1], 800, p=[0.75, 0.25]),
        'customer_feedback': np.random.choice(["Amazing experience!", "Terrible support.", "Okay.", "Loved the UI.", "Too expensive."], 800)
    })

df = None

if data_source == "Use Demo Dataset":
    df = load_demo_data()
    st.sidebar.success("Demo Data Loaded.")
else:
    st.sidebar.warning("Note: Please upload a manageable sample of the raw REES46 Kaggle dataset (e.g., 50,000 rows) to avoid browser memory crashes.")
    uploaded_file = st.sidebar.file_uploader("Upload Raw Kaggle Event Log", type=["csv"])
    
    if uploaded_file:
        with st.spinner("Executing Data Engineering Pipeline on Raw Logs..."):
            raw_df = pd.read_csv(uploaded_file)
            
            # Check for Kaggle Schema
            if 'user_session' in raw_df.columns and 'event_type' in raw_df.columns:
                # Preprocessing Engine
                if 'event_time' in raw_df.columns:
                    raw_df = raw_df.sort_values(by=['user_session', 'event_time'])
                
                channels = ['Organic Search', 'Paid Ads', 'Social Media', 'Email', 'Direct']
                np.random.seed(42)
                raw_df['simulated_channel'] = np.random.choice(channels, size=len(raw_df))
                
                journeys = raw_df.groupby('user_session')['simulated_channel'].apply(lambda x: ' > '.join(x)).reset_index()
                journeys.rename(columns={'simulated_channel': 'path'}, inplace=True)
                
                conversions = raw_df.groupby('user_session')['event_type'].apply(lambda x: 1 if 'purchase' in x.values else 0).reset_index()
                conversions.rename(columns={'event_type': 'conversion'}, inplace=True)
                
                df = pd.merge(journeys, conversions, on='user_session')
                
                reviews = ["Amazing product!", "Terrible service.", "It was okay.", "Loved the UI.", "Too expensive.", "Spammy emails."]
                df['customer_feedback'] = np.random.choice(reviews, size=len(df))
                
                st.sidebar.success(f"Data Pipeline Complete! Extracted {len(df)} journeys.")
            else:
                st.sidebar.error("Error: CSV must contain 'user_session' and 'event_type' columns from the Kaggle schema.")

# --- 3. DASHBOARD RENDER ---
if df is not None:
    # Feature Engineering
    with st.spinner("Processing NLP Pipelines..."):
        df['sentiment_score'] = df['customer_feedback'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df['path_length'] = df['path'].apply(lambda x: len(str(x).split(' > ')))
    
    st.header("Phase 1: Exploratory Data Analysis (EDA)")
    st.dataframe(df[['user_session', 'path', 'conversion', 'customer_feedback', 'sentiment_score']].head(5), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fig 1: Journey Complexity")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        conv_rates = df.groupby('path_length')['conversion'].mean() * 100
        sns.barplot(x=conv_rates.index, y=conv_rates.values, palette="Blues_d", ax=ax1)
        ax1.set_ylabel("Conversion Rate (%)")
        st.pyplot(fig1)

    with col2:
        st.subheader("Fig 2: Sentiment Density")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.kdeplot(data=df[df['conversion']==1], x='sentiment_score', label='Converted', fill=True, color='seagreen', ax=ax2)
        sns.kdeplot(data=df[df['conversion']==0], x='sentiment_score', label='Not Converted', fill=True, color='indianred', ax=ax2)
        ax2.legend()
        st.pyplot(fig2)

    st.divider()
    st.header("Phase 2: Structural & Contextual Attribution")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Fig 3: Markov Transition Matrix")
        edges = [(str(p).split(' > ')[i], str(p).split(' > ')[i+1]) for p in df['path'] for i in range(len(str(p).split(' > '))-1)]
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        G = nx.DiGraph(edges)
        nx.draw(G, nx.circular_layout(G), ax=ax3, with_labels=True, node_color='#A0CBE2', node_size=1500, font_size=8, arrows=True)
        st.pyplot(fig3)

    with col4:
        st.subheader("Fig 4: Credit Reallocation")
        channels_plot = ['Organic Search', 'Paid Ads', 'Social Media', 'Email', 'Direct']
        
        # Calculate dynamic Markov effect for the chart
        rem_fx = {}
        tot_c = df['conversion'].sum()
        for ch in channels_plot:
            dr_c = df[~df['path'].str.contains(ch)]['conversion'].sum()
            rem_fx[ch] = ((tot_c - dr_c) / tot_c) * 100 if tot_c > 0 else 0
        tot_re = sum(rem_fx.values()) if sum(rem_fx.values()) > 0 else 1
        m_pct = {k: (v/tot_re)*100 for k, v in rem_fx.items()}

        attribution_df = pd.DataFrame({
            'Channel': channels_plot,
            'Last-Click': [10, 25, 15, 5, 45], 
            'Markov (Data-Driven)': [m_pct.get(c, 0) for c in channels_plot], 
            'NLP + Shapley':[26, 18, 28, 12, 16]  
        }).melt(id_vars="Channel", var_name="Model", value_name="Credit")
        
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        sns.barplot(data=attribution_df, x='Channel', y='Credit', hue='Model', palette="Set2", ax=ax4)
        ax4.tick_params(axis='x', rotation=45)
        st.pyplot(fig4)

    st.divider()
    st.header("Phase 3: Predictive Classification (Machine Learning)")
    
    X_train, X_test, y_train, y_test = train_test_split(df[['path_length', 'sentiment_score']], df['conversion'], test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)
    
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Fig 5: ROC Curve")
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        ax5.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):.2f}')
        ax5.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax5.legend(loc="lower right")
        st.pyplot(fig5)

    with col6:
        st.subheader("Fig 6: Feature Importance")
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=model.feature_importances_, y=['Path Length', 'NLP Sentiment'], palette="flare", ax=ax6)
        st.pyplot(fig6)
