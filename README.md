# Networked Dynamics of Toxicity in Reddit Gaming Communities

This repository contains the data extraction, classification, and network analysis code for our paper, "Networked Dynamics of Toxicity in Reddit Gaming Communities," accepted for presentation at **NetSci '26** and **CySoc '26**.

## 📊 Project Overview

This research investigates how game genre and developer scale jointly shape the prevalence, diffusion, and structural concentration of toxicity across social networks. The study utilizes a novel $2\times2$ comparative framework crossing game genre (MMORPG vs. Soulslike) with developer scale (AAA/AA vs. indie). 

Using over 550,641 Reddit posts and comments from nine gaming communities, content was scored using two independent toxicity detection systems: OpenAI's omni-moderation and Detoxify. The analysis evaluates toxicity at three structural levels:
* **Community-level:** Assessing differences in toxicity prevalence, severity, and category composition across genres and developer scales.
* **Thread-level:** Examining local contagion and cascade structure, including depth, size, and structural virality.
* **User-level:** Analyzing assortative mixing and centrality-based patterns of toxic exposure and production within user-reply networks.

## 📁 Repository Structure

* **`data/`**: Directory for raw and processed datasets (Ignored in `.gitignore`)
* **`notebooks/`**:
  * `01_data_extraction.ipynb` - Arctic Shift ingestion and preprocessing
  * `02_nlp_classification.ipynb` - Execution of OpenAI API and Detoxify scoring
  * `03_thread_analysis.ipynb` - Contagion modeling and cascade structure
  * `04_user_networks.ipynb` - Assortativity and centrality-toxicity mapping
* **`requirements.txt`**: Python dependencies
* **`README.md`**: Project documentation

## 🚀 How to Run

1. **Clone the repository:**
```bash
git clone [https://github.com/glenjpdavid/toxic-gaming-train.git](https://github.com/glenjpdavid/toxic-gaming-train.git)
cd toxic-gaming-train
```

2. **Install dependencies:**
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

3. **API Keys:**
To run the NLP classification models, you will need to add your OpenAI API key to a `.env` file in the root directory:
```text
OPENAI_API_KEY=your_key_here
```

4. **Execute Notebooks:**
Run the notebooks in the `notebooks/` directory in sequential order to replicate the data extraction, classification, and statistical analysis steps.

## ✍️ Authors & Acknowledgements

This project was a collaborative research effort accepted for presentation at **NetSci '26** and **CySoc '26**.

* **Serafina Smith** - First Author
* **Glen J. David** - Second Author | Data Scientist & ML Engineer | [LinkedIn](https://linkedin.com/in/glenjpdavid)
* **Lizzy Brunn** - Co-Author
* **Vy Nguyen** - Co-Author
* **Eun Cheol Choi** - Teaching Assistant
* **Luca Luceri** - Professor
