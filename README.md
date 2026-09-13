# university-rag-recommender
A RAG-based university recommendation system that helps students find relevant universities and programs
# 🎓 RAG-Based University Recommendation System

An AI-powered university recommendation system that uses **Retrieval-Augmented
Generation (RAG)** to suggest the best-matching universities for a student
based on their interests, preferred city, and budget — grounded entirely in
real university data (not hallucinated).

Built with **Python**, **LangChain**, **FAISS**, and **HuggingFace**
transformers.

## How it works

1. **Data** — A structured dataset of universities (name, location, programs,
   tuition, ranking, description) is stored in `data/universities.json`.
2. **Indexing** — Each university record is embedded using a HuggingFace
   sentence-transformer model (`all-MiniLM-L6-v2`) and stored in a **FAISS**
   vector index for fast semantic search.
3. **Retrieval** — When a student describes what they're looking for (e.g.
   *"I want to study AI in Islamabad with a budget under 400,000 PKR"*), the
   system retrieves the most semantically relevant universities from the
   vector store.
4. **Generation** — The retrieved university data is passed as context to a
   language model (`google/flan-t5-base`), which generates a natural-language
   recommendation **grounded only in the retrieved context** — this is the
   core RAG pattern, reducing hallucination compared to a plain LLM call.

## Project structure

\`\`\`
university-rag-recommender/
├── data/
│   └── universities.json      # Sample university dataset
├── src/
│   ├── build_index.py         # Builds the FAISS vector index
│   └── recommend.py           # Core RAG pipeline (retrieve + generate)
├── app.py                     # Streamlit demo UI
├── requirements.txt
└── README.md
\`\`\`

## Setup

\`\`\`bash
# 1. Clone the repo
git clone https://github.com/Aroojkhan192/university-rag-recommender.git
cd university-rag-recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the vector index (run once)
python src/build_index.py

# 4. Get a recommendation from the command line
python src/recommend.py "I want to study Computer Science in Lahore, budget under 300000 PKR"

# 5. Or launch the interactive web app
streamlit run app.py
\`\`\`

## Example

**Input:**
> "I want to study Artificial Intelligence in Islamabad with a budget under 400,000 PKR per year"

**Output (example):**
> Based on your interest in Artificial Intelligence and budget, **Air
> University** (Islamabad) is a strong match — it offers a dedicated AI
> program at approximately PKR 320,000/year. **FAST-NUCES** is also worth
> considering for its strong AI and Data Science curriculum, though tuition
> is slightly higher...

## Tech Stack

- **Python**
- **LangChain** — RAG orchestration
- **FAISS** — vector similarity search
- **HuggingFace Transformers / Sentence-Transformers** — embeddings and
  text generation
- **Streamlit** — demo web interface

## Future Improvements

- [ ] Expand dataset to cover more universities and programs
- [ ] Add filtering by tuition range and city as structured metadata filters
- [ ] Swap in a larger instruction-tuned model for richer recommendations
- [ ] Add user feedback loop to improve retrieval ranking over time

