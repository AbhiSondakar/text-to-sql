# 🤖 NLP-Powered Conversational Assistant to Database

A production-ready Text-to-SQL Agent that empowers users to query their databases using natural language. Built with Streamlit, PostgreSQL, and a robust Two-Agent AI System (Generator & Validator), this tool ensures accurate SQL generation and provides automatic data visualization.

---

## ✨ Key Features

* 🗣️ **Natural Language Interface**: Ask questions in plain English (e.g., "Show me top 5 students by GPA").
* 🛡️ **Two-Agent Architecture**:
   * **Agent 1 (Generator)**: Converts your question into a SQL query.
   * **Agent 2 (Validator)**: Reviews the query for safety, correctness, and efficiency before execution.
* 📊 **Intelligent Visualization**: Automatically selects the best chart type (Bar, Line, Pie, Scatter, Heatmap) based on the data returned.
* 🔒 **Security First**:
   * Read-only access to your data.
   * Strict validation against forbidden keywords (`DROP`, `DELETE`, etc.).
   * Enforced row limits.
* 🔌 **Multi-Provider Support**: Works with OpenRouter, Google Gemini, DeepSeek, Anthropic, OpenAI, and Groq.
* 💾 **Persistent History**: Saves chat sessions and query results to a dedicated application database.

---

## 🏗️ Architecture

The system is designed with a clear separation of concerns:

1. **Frontend**: Streamlit interface for chat and visualization.
2. **Application DB** (`text_to_sql_db`): Stores chat history, sessions, and user feedback. (Read/Write)
3. **Data DB** (`Student`): The target database containing your business data. (Read-Only)
4. **AI Engine**: Handles prompt engineering, model switching, and response parsing.

---

## 🚀 Quick Start

### Prerequisites

* Python 3.11+
* PostgreSQL installed and running

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/AbhiSondakar/text-to-sql.git
cd text-to-sql
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the Setup Script**

This automated script creates the databases, configures users, and generates your `.env` file.

```bash
python setup_databases.py
```

**4. Configure API Keys**

Edit the generated `.env` file and add your AI provider key (e.g., OpenRouter or Google Gemini).

```ini
OPENROUTER_API_KEY=sk-or-v1-your-key
```

**5. Start the Application**

```bash
streamlit run app.py
```

---

## 📖 Usage Guide

1. **Select a Chat**: Start a new chat or continue a previous session from the sidebar.
2. **Ask a Question**: Type your query in the input box.
   * Example: _"How many students are enrolled in Computer Science?"_
   * Example: _"Show the distribution of grades as a pie chart."_
3. **Review Results**:
   * See the AI's explanation of the query.
   * View the **Generated SQL** code.
   * Explore the **Data Table** and **Visualization**.

---

## 🛠️ Configuration

You can customize the AI models in your `.env` file to balance cost and performance:

| Variable | Description | Recommended Model |
|----------|-------------|-------------------|
| `SQL_AGENT_MODEL` | Generates the initial SQL query | `deepseek/deepseek-chat` |
| `VALIDATOR_AGENT_MODEL` | Validates and fixes the query | `deepseek/deepseek-chat` or `anthropic/claude-3.5-sonnet` |

See `test/config.md` for a detailed guide on available models and pricing.

---

## 🧪 Testing & Diagnostics

**Test Database Connections:**

```bash
python test/main.py
```

**Verify Config:**

```bash
python verify_config.py
```

---

## 📂 Project Structure

```
text-to-sql/
├── app.py                      # Main Streamlit application entry point
├── ai_service.py               # Handles interactions with AI providers
├── visualization_service.py    # Logic for generating charts from data
├── sql_validator.py            # Security checks and SQL parsing
├── setup_databases.py          # Script to initialize PostgreSQL databases
├── database.py                 # Database connection management
├── models.py                   # SQLAlchemy ORM models for chat history
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment configuration
├── SETUP.md                    # Detailed setup instructions
└── test/
    ├── main.py                 # Connection diagnostic tests
    └── config.md               # Model configuration guide
```

---

## 🔒 Security Features

* **Read-Only Database Access**: The `readonly_user` has no `INSERT`, `UPDATE`, or `DELETE` privileges on the data database.
* **SQL Injection Prevention**: All queries are validated against a blacklist of dangerous keywords.
* **Row Limit Enforcement**: Maximum result size is configurable to prevent performance issues.
* **Schema Whitelisting**: Optionally restrict which tables the AI can access.

---

## 🌟 Example Queries

* "What is the average GPA by major?"
* "List the top 10 students with the highest grades"
* "Show me enrollment trends over the last 5 years"
* "How many courses are offered in each department?"
* "Create a bar chart comparing student counts by year"

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

* Built with [Streamlit](https://streamlit.io/)
* Powered by [OpenRouter](https://openrouter.ai/), [Google Gemini](https://ai.google.dev/), and other AI providers
* Database management via [SQLAlchemy](https://www.sqlalchemy.org/)

---

## 📧 Support

For questions or support, please:

* Open an issue on GitHub
* Check the [SETUP.md](SETUP.md) for detailed configuration help
* Review `test/config.md` for model-specific troubleshooting

---
