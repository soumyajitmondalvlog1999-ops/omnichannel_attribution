📊 Omni-Attribution AI: Context-Aware Marketing Attribution Framework
📌 Project Overview
Modern digital marketing suffers from a multi-million-dollar blindspot: Heuristic bias and emotional vacuum. Legacy attribution models (like Last-Click) falsely award total conversion credit to the final navigational action (e.g., Direct Search) while completely ignoring the human friction or delight experienced during the journey.

This project is a complete data engineering and machine learning pipeline that dismantles legacy attribution. By fusing Discrete-Time Stochastic Processes (Markov Chains), Cooperative Game Theory (Shapley Value), and Natural Language Processing (NLP), this framework dynamically reallocates marketing budgets based on both structural path probability and qualitative customer sentiment.

🚀 Key Innovations & Features
Big Data Engineering: Processed a 5GB e-commerce clickstream dataset (REES46, 42M+ rows) into 12,000+ chronological customer journeys.

Sentiment-Weighted Markov Chains: Integrated VADER NLP polarity scores (-1.0 to +1.0) directly into Markov transition matrices to mathematically penalize channels that drive negative customer experiences (CX).

Algorithmic Fairness: Utilized the Shapley Value to distribute fractional financial credit fairly across the omnichannel ecosystem.

Predictive Forecasting: Trained a Random Forest classifier (0.76 AUC) to predict conversion intent based on sequence complexity and real-time emotion.

Interactive Dashboard: Deployed a fully functional Streamlit web application allowing stakeholders to upload raw logs and instantly visualize budget reallocation.

🛠️ Technology Stack
Language: Python

Data Processing: Pandas, NumPy

Machine Learning: Scikit-Learn (Random Forest, XGBoost)

Network Graphing & Math: NetworkX, Itertools

Natural Language Processing: TextBlob, VADER

Data Visualization: Matplotlib, Seaborn

Frontend/UI: Streamlit

📈 Strategic Business Results
When applied to the 12,798-session sample, the Hybrid NLP + Shapley framework revealed severe inefficiencies in standard heuristic tracking:

Corrected Bottom-Funnel Bias: The model proved that "Direct" traffic was historically overvalued by 29%, triggering a massive budget reallocation.

Rewarded Discovery: Upper-funnel acquisition channels like "Social Media" and "Organic Search" saw a combined valuation increase of over 25% once negative cross-channel friction was accounted for.
