# 🛠️ Setup & Configuration Guide

This guide provides step-by-step instructions to set up the Text-to-SQL Agent, a two-agent system (Generator & Validator) that converts natural language into safe, executable SQL queries.

---

## 📋 Prerequisites

Ensure you have the following installed on your system:

- **Python 3.11+**
- **PostgreSQL** (Must be running and accessible)
- **Git**

---

## 🚀 Quick Start (Automated Setup)

The project includes an automated script that handles database creation, user permissions, and environment configuration.

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/AbhiSondakar/text-to-sql
cd text-to-sql

# Create virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Auto-Setup Script

This script creates the necessary databases (`text_to_sql_db` and `Student`), configures secure users, and generates your `.env` file.

```bash
python setup_databases.py
```

**What this script does:**

- ✅ **Creates Databases**: `text_to_sql_db` (for chat history) and `Student` (for data analysis)
- ✅ **Creates Users**:
  - `app_user`: Read/Write access for chat history
  - `readonly_user`: Read-Only access for querying data (security best practice)
- ✅ **Generates Data**: Optionally populates the `Student` database with sample data
- ✅ **Configures Environment**: Auto-generates a valid `.env` file for you

> **Note:** You will need your PostgreSQL superuser password (usually for the `postgres` user) to run this script.

### 3. Add Your API Key

Open the newly created `.env` file and add your AI provider key.

### 4. Verify Configuration

Run the verification tool to check your API keys and model settings:

```bash
python verify_config.py
```

### 5. Run the App

```bash
streamlit run app.py
```

---

## ⚙️ Manual Configuration

If you prefer to set up the environment manually, follow these steps.

### 1. Database Setup

You need two PostgreSQL databases:

- **Application DB** (`text_to_sql_db`): Stores chat sessions. (User needs INSERT/UPDATE/SELECT permissions)
- **Data DB** (`Student`): Stores the business data to be queried. (User MUST be READ ONLY to prevent SQL injection)

### 2. Environment Variables (.env)

Create a `.env` file in the root directory.

#### Recommended Configuration (Groq + OpenRouter)

This configuration uses Groq for fast execution and OpenRouter for flexible model options.

```ini
# --- AI Provider Keys ---
# Get Key: https://console.groq.com/keys
GROQ_API_KEY=gsk_...

# Get Key: https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-v1-...

# --- Model Selection ---
# Agent 1 (SQL Generator): Uses OpenRouter default fallback
SQL_AGENT_MODEL=mistralai/mistral-7b-instruct-v0.3

# Agent 2 (Validator): Auto-detects 'groq' provider via model name
VALIDATOR_AGENT_MODEL=llama-3.3-70b-versatile

# --- Database Connection Strings ---
APP_DB_URL=postgresql://app_user:pass@localhost:5432/text_to_sql_db
DATA_DB_URL=postgresql://readonly_user:pass@localhost:5432/Student
```

---

## 🧠 Model Selection & Provider Logic

The system automatically detects the API provider based on the **Model Name** you provide in `.env`.

| Provider | Trigger Keywords in Model Name | Required API Key Env Var | Example Model String |
|----------|-------------------------------|--------------------------|---------------------|
| **Groq** | `llama-3.3`, `llama-3.1`, `mixtral`, `gemma2`, `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| **Google** | `gemini` | `GOOGLE_API_KEY` | `gemini-2.0-flash-exp` |
| **Anthropic** | `claude` | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-20240620` |
| **DeepSeek** | `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| **OpenAI** | `gpt-4`, `gpt-3.5`, `o1-`, `o3-` | `OPENAI_API_KEY` | `gpt-4o` |
| **OpenRouter** | `openrouter/` prefix OR any unknown model | `OPENROUTER_API_KEY` | `mistralai/mistral-7b-instruct` |

---

## 💡 Supported Configurations

### 1. The "Fast & Free" Stack (Google Gemini)

Uses Google's free tier (if available) for both agents.

```ini
GOOGLE_API_KEY=AIzaSy...
SQL_AGENT_MODEL=gemini-2.0-flash-exp
VALIDATOR_AGENT_MODEL=gemini-2.0-flash-exp
```

### 2. The "Performance" Stack (DeepSeek + Anthropic)

Best for complex queries and rigorous validation.

```ini
DEEPSEEK_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

SQL_AGENT_MODEL=deepseek-chat
VALIDATOR_AGENT_MODEL=claude-3-5-sonnet-20240620
```

### 3. The "Default" Stack (OpenRouter + Groq)

As seen in the example configuration.

```ini
OPENROUTER_API_KEY=sk-or-v1-...
GROQ_API_KEY=gsk_...

SQL_AGENT_MODEL=mistralai/mistral-7b-instruct-v0.3
VALIDATOR_AGENT_MODEL=llama-3.3-70b-versatile
```

---

## 🔍 Troubleshooting & Diagnostics

### Database Connection Issues

If the app fails to connect to the database, run the diagnostic tool:

```bash
python test/main.py
```

This script tests:
- `psycopg2` driver connections
- SQLAlchemy engine connections
- Simulated Streamlit connections
- Environment variable correctness

### "Output Not Available" / Redaction

If the AI cannot see your tables:

- Check `MAX_SCHEMA_TABLES` in `.env` (default is 50)
- Use `SCHEMA_TABLE_WHITELIST` in `.env` (comma-separated list) to specify exactly which tables the AI can access

### Rate Limiting

If you see rate limit errors (especially with Free tiers):

- Switch to a paid model or OpenRouter
- Wait 60 seconds before retrying (the app has built-in backoff)

---

## 🔒 Security Notes

- **Read-Only Access**: The `setup_databases.py` script specifically revokes `INSERT`, `UPDATE`, and `DELETE` permissions for the `readonly_user`. Do not use the `app_user` for the `DATA_DB_URL`.

- **Injection Protection**: The `sql_validator.py` module explicitly blocks keywords like `DROP`, `TRUNCATE`, and `GRANT` before execution.

- **Secrets**: Never commit your `.env` file. It is already added to `.gitignore`.

---

## 📚 Additional Resources

- **API Provider Documentation:**
  - [Groq Console](https://console.groq.com/)
  - [OpenRouter](https://openrouter.ai/docs)
  
- **PostgreSQL Setup:** [Official Installation Guide](https://www.postgresql.org/download/)

---

## 🤝 Support

If you encounter issues not covered in this guide:

1. Check the `test/config.md` file for advanced configuration options
2. Review the application logs in the Streamlit interface
3. Open an issue on the project repository with detailed error messages

---
