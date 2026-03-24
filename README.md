# Talentlens - Resume Intelligence Platform

AI-powered candidate discovery using Retrieval Augmented Generation (RAG). This platform helps recruiters quickly find and rank candidates based on skills, experience, and job requirements.

## 🎯 Features

### Core Functionality
- **Resume Search**: Natural language queries to find matching candidates
- **Upload & Rank**: Bulk resume upload with intelligent ranking
- **Skill-Based Matching**: Automatic skill extraction and matching (0-100% scores)
- **Real-time Scoring**: Dynamic match percentage calculation
- **Candidate Cards**: Clean, professional candidate display with match details

### Advanced Features
- **Smart Skill Extraction**: Automatically extracts 40+ tech skills from resumes and job descriptions
- **Intelligent Matching**: Normalized skill matching with fallback logic
- **Session Persistence**: Results persist across UI interactions
- **Debug Mode**: Transparent matching pipeline for troubleshooting
- **Export Functionality**: Download candidate results as CSV

<img width="1919" height="871" alt="image" src="https://github.com/user-attachments/assets/c55c2463-2a9b-4086-9875-6ff1d28e11ed" />
<img width="1918" height="902" alt="image" src="https://github.com/user-attachments/assets/edaf75d5-59d5-486f-b73f-186eec507a2c" />
<img width="1919" height="944" alt="image" src="https://github.com/user-attachments/assets/a0ae2e57-a1bf-4524-9934-d074c3c26db3" />


## 🚀 Quick Start

### 1. Environment Setup
Create `.env` in the project root:
```bash
OPENAI_API_KEY=YOUR_OPENAI_KEY
PINECONE_API_KEY=YOUR_PINECONE_KEY
PINECONE_INDEX=resumes-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Upload Resume Data (One-time)
```bash
python upsert_resumes.py --csv Resume_cleaned.csv
```

### 4. Run the Application
```bash
streamlit run app.py
```

Visit `http://localhost:8501` to access Talentlens.

## 🎨 UI Overview

### Resume Search Tab
1. **Query Input**: Describe the role you're looking for
2. **Refine Search**: Set experience range, location, and number of candidates
3. **Results**: View matched candidates with scores and skill highlights
4. **Actions**: View full resume or add to shortlist

### Upload & Rank Tab
1. **Job Description**: Paste the job requirements
2. **Upload Resumes**: Upload PDF, DOCX, or TXT files
3. **Rank Candidates**: Get AI-powered ranking with match scores
4. **Export Results**: Download ranked candidates as CSV

## 🧠 Skill Matching Pipeline

### Skill Extraction
The platform automatically extracts skills from:
- **Resumes**: 40+ tech skills including Python, AWS, Docker, React, etc.
- **Job Descriptions**: Natural language processing to identify requirements
- **Fallback Logic**: Default skills if none detected

### Matching Algorithm
```python
# Normalized skill intersection
matched = set(candidate_skills) & set(jd_skills)
score = (len(matched) / len(jd_skills)) * 100
```

### Supported Skills
- **Programming**: Python, Java, JavaScript, C++, TypeScript
- **Cloud**: AWS, Azure, GCP, Docker, Kubernetes
- **Databases**: SQL, PostgreSQL, MongoDB, Redis
- **Frameworks**: React, Node.js, Django, Flask, Spring
- **AI/ML**: Machine Learning, AI, TensorFlow, PyTorch
- **Tools**: Git, CI/CD, Jira, Power BI, Tableau

## 📊 Match Scoring

### Score Calculation
- **100%**: All required skills found
- **66.7%**: 2 out of 3 skills match
- **33.3%**: 1 out of 3 skills match
- **0%**: No skill overlap

### Visual Indicators
- 🎯 **Match Skills**: Shows overlapping skills
- ✅ **Skill Count**: Number of matches found
- ❌ **No Matches**: Clear indication when no overlap

## 🛠️ Technical Stack

### Core Technologies
- **Frontend**: Streamlit (Python web framework)
- **AI/ML**: OpenAI GPT-4o-mini, Embeddings
- **Vector DB**: Pinecone (serverless)
- **Processing**: Python, Pandas, Regex

### Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│  Query Pipeline  │───▶│   Pinecone DB   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  OpenAI APIs     │
                       │ (Embed + Chat)   │
                       └──────────────────┘
```

## 📁 Project Structure

```
TalentLens/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (not committed)
├── README.md             # This file
├── src/
│   ├── config.py         # Configuration and constants
│   ├── parser.py         # Resume parsing and skill extraction
│   ├── embed.py          # OpenAI embedding helpers
│   ├── query_pipeline.py # Search and retrieval logic
│   └── tools.py          # Utility functions
├── scripts/
│   └── upsert_resumes.py # Data upload script
└── Resume/
    └── Resume_cleaned.csv # Sample resume data
```

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=resumes-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

### Model Settings
- **Embedding Model**: `text-embedding-3-small` (1536 dimensions)
- **Chat Model**: `gpt-4o-mini`
- **Similarity Metric**: Cosine
- **Top Results**: 5-20 candidates

## 🐛 Troubleshooting

### Common Issues

#### 0% Match Scores
- **Cause**: Empty skill lists or extraction failure
- **Fix**: Check debug panel for skill extraction details
- **Solution**: Ensure resumes contain tech keywords

#### No Results Found
- **Cause**: Empty database or wrong index
- **Fix**: Verify Pinecone index and data upload
- **Solution**: Run upsert script with correct CSV

#### Button Clicks Reset Page
- **Cause**: Streamlit form issues
- **Fix**: Session state persistence implemented
- **Solution**: All actions now persist properly

### Debug Mode
Enable debug information by expanding the "🔍 Debug: Matching Details" section to see:
- JD skills extracted
- Candidate skills found
- Skill matching calculation
- Final scoring breakdown

## 📈 Performance

### Speed
- **Query Response**: 2-5 seconds
- **File Upload**: 1-3 seconds per resume
- **Skill Extraction**: <1 second per document

### Scaling
- **Database**: Pinecone serverless (auto-scaling)
- **Concurrent Users**: Streamlit supports multiple users
- **File Limits**: No hard limit on resume uploads

## 🤝 Contributing

### Development Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up `.env` file with API keys
4. Run: `streamlit run app.py`

### Adding New Skills
Update the `SKILL_KEYWORDS` list in `app.py`:
```python
SKILL_KEYWORDS = [
    "python", "sql", "aws",  # existing skills
    "new_skill", "another_skill"  # add new skills here
]
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Enable debug mode for detailed logs
3. Review the skill extraction pipeline
4. Verify environment configuration

---

**Talentlens** - Transforming recruitment with AI-powered candidate intelligence. 🚀
