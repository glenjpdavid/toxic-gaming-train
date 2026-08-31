🎮 Networked Dynamics of Toxicity in Reddit Gaming Communities

📌 Project Overview

This repository contains the code, data pipelines, and analytical models for the research project: "Networked Dynamics of Toxicity in Reddit Gaming Communities." This work was accepted for presentation at NetSci '26 and CySoc '26.

The primary objective of this project is to analyze how toxic behavior spreads through online gaming communities. Moving beyond generalized claims about "toxic gaming culture," this research introduces a novel $2\times2$ comparative framework that crosses game genre (MMORPG vs. Soulslike) with developer scale (AAA/AA vs. Indie).

By extracting and scoring over 550,000 Reddit comments using multi-class NLP models, we constructed thread-level reply trees and user-level networks to mathematically map toxicity cascades and contagion based on network position.

🏆 Key Findings

Community-Level Prevalence: Soulslike communities exhibit significantly higher average toxicity than MMO communities, driven heavily by language tied to violent/illicit gameplay contexts and adversarial interaction styles. AAA/AA titles also show elevated toxicity compared to Indie titles.

Thread-Level Contagion: Toxicity is structurally amplified. Discussion threads exhibit robust local contagion—replies to toxic comments are consistently more toxic, and highly toxic threads tend to grow larger, deeper, and more "structurally viral."

User-Level Network Roles (Assortative Mixing): Toxicity is a heavily networked phenomenon. Users exhibit positive assortative mixing (toxic users preferentially interact with other toxic users). Furthermore, centrality differentiates exposure from expression: users with high in-degree centrality are exposed to more toxicity, while users with high out-degree centrality produce more toxicity.

🛠️ Data & Methodology

Dataset: 550,641 Reddit posts and comments extracted via Arctic Shift across 9 subreddits (e.g., r/WoW, r/DarkSouls, r/gaming).

Toxicity Scoring: Dual-system annotation using OpenAI's omni-moderation-latest (for policy-relevant harmfulness signals) and Detoxify (original-small, for abuse-pattern probability).

Network Graph Analysis: Construction of thread-level reply trees and user-reply directed graphs to compute mixed-effects models, Spearman rank correlations, and node-label permutation tests.

📂 Repository Structure

Note: The following structure outlines the primary execution flow of the project's codebase.

├── data/                       # Directory for raw and processed datasets (Ignored in .gitignore)
├── notebooks/                  
│   ├── 01_data_extraction.ipynb   # Arctic Shift ingestion and preprocessing
│   ├── 02_nlp_classification.ipynb# Execution of OpenAI API and Detoxify scoring
│   ├── 03_thread_analysis.ipynb   # Contagion modeling and cascade structure
│   └── 04_user_networks.ipynb     # Assortativity and centrality-toxicity mapping
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

🚀 How to Run
Clone the repository:
Bash
git clone https://github.com/glenjpdavid/toxic-gaming-train.git
cd toxic-gaming-train
Install dependencies:
It is recommended to use a virtual environment.
Bash
pip install -r requirements.txt
API Keys:
To run the NLP classification models, you will need to add your OpenAI API key to a .env file in the root directory:
Plaintext
OPENAI_API_KEY=your_key_here


4. **Execute Notebooks:**
Run the notebooks in the `notebooks/` directory in sequential order to replicate the data extraction, classification, and statistical analysis steps.

## ✍️ Authors & Acknowledgements

* **Serafina Smith** - First Author
* **Glen J. David** - Second Author | Data Scientist & ML Engineer | [LinkedIn](https://linkedin.com/in/glenjpdavid)
* **Lizzy Brunn** - Co-Author
* **Vy Nguyen** - Co-Author
* **Eun Cheol Choi** - Teaching Assistant
* **Luca Luceri** - Professor
