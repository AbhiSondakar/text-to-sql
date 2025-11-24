# 🛠️ NLP Powered Conversational Assistance to Database Setup Guide

This project is a sophisticated AI-powered SQL generator and visualizer. It uses a **Two-Agent System** (Generator & Validator) to convert natural language into safe, executable SQL queries against a PostgreSQL database.

---

## 📋 Prerequisites

Before starting, ensure you have the following installed:

- **Python 3.11+**
- **PostgreSQL** (Must be running locally or accessible via network)
- **Git**

---

## 🚀 Quick Start (Recommended)

The project includes an automated setup script that handles database creation, user permissions, and configuration file generation.

### 1. Clone & Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd text-to-sql

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Auto-Setup Script

This script is the heart of the setup. It will:

- ✅ Create the **Application DB** (`text_to_sql_db`) for chat history
- ✅ Create the **Data DB** (`Student`) for querying
- ✅ Create secure database users (`app_user` and `readonly_user`)
- ✅ Generate sample data
- ✅ Auto-generate your `.env` file

```bash
python setup_databases.py
```

> **Note:** You will be prompted for your PostgreSQL superuser password (usually the `postgres` user).

### 3. Configure API Keys

Open the newly created `.env` file. You must set your API provider details. The system supports **OpenRouter**, **Google Gemini**, **DeepSeek**, **Anthropic**, and **OpenAI**.

#### For a Free/Low-Cost Start:

```ini
# In your .env file
OPENROUTER_API_KEY=sk-or-v1-your-key-here
MODEL_NAME=deepseek/deepseek-chat
```

### 4. Verify Configuration

Run the verification script to ensure your API keys and models are detected correctly:

```bash
python verify_config.py
```

If you see **"✅ Your configuration is correct!"**, you are ready to go.

### 5. Launch the Application

```bash
streamlit run app.py
```

---

## ⚙️ Manual Configuration (Advanced)

If you prefer to configure manually or the script fails, follow these steps.

### 1. Database Architecture

This system requires two logical databases (or connections) for security:

- **Application Database** (`text_to_sql_db`): Stores chat sessions and logs. Needs **WRITE** access.
- **Data Database** (`Student`): The target data you want to query. The AI uses a **READ-ONLY** user here to prevent accidental data modification.

### 2. Environment Variables (.env)

Create a `.env` file in the root directory:

```ini
# --- AI Provider (Choose One) ---
# OPENROUTER (Recommended)
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_API_URL=https://openrouter.ai/api/v1

# OR GOOGLE GEMINI (Free Tier Available)
# GOOGLE_API_KEY=AIzaSy...

# --- Database Config ---
# User for Chat History (Needs INSERT/UPDATE)
APP_DB_URL=postgresql://app_user:pass@localhost:5432/text_to_sql_db

# User for Data Querying (MUST BE READ-ONLY)
DATA_DB_URL=postgresql://readonly_user:pass@localhost:5432/Student

# --- Model Selection ---
# Agent 1: Generates the SQL
SQL_AGENT_MODEL=deepseek/deepseek-chat

# Agent 2: Validates the SQL
VALIDATOR_AGENT_MODEL=deepseek/deepseek-chat
```

---

## 🧠 Model Selection Guide

You can mix and match models for the SQL Generator and Validator agents.

| Strategy | SQL Model (SQL_AGENT_MODEL) | Validator Model (VALIDATOR_AGENT_MODEL) | Cost | Performance |
|----------|----------------------------|----------------------------------------|------|-------------|
| **Best Value** | `deepseek/deepseek-chat` | `deepseek/deepseek-chat` | Low | ⭐⭐⭐⭐⭐ (Excellent SQL) |
| **Free Tier** | `google/gemini-2.0-flash-exp:free` | `google/gemini-2.0-flash-exp:free` | Free | ⭐⭐⭐⭐ (Good) |
| **Premium** | `deepseek/deepseek-chat` | `anthropic/claude-3.5-sonnet` | High | ⭐⭐⭐⭐⭐ (Best Reasoning) |

> Reference `test/config.md` for more model details.

---

## 🛠️ Troubleshooting

### Connection Issues?

Run the connection diagnostic tool:

```bash
python test/main.py
```

This will test `psycopg2`, `sqlalchemy`, and simulated streamlit connections to pinpoint exactly where the connection is failing (e.g., wrong password, missing driver, firewall).

### "Output Not Available" / Redaction?

If the AI refuses to answer, check the **Schema Whitelist** settings in `.env`:

```ini
# Only allow AI to see these tables
SCHEMA_TABLE_WHITELIST=students,courses
MAX_SCHEMA_TABLES=50
```

### Resetting the Environment

If you need to clear the database cache in Streamlit:

- Click the **"Clear Cache"** button in the app sidebar
- Or delete the `__pycache__` folders manually

---

## 🔒 Security Notes

- **Read-Only Access:** The `setup_databases.py` script specifically revokes `INSERT`, `UPDATE`, and `DELETE` permissions for the `readonly_user`. Do not use the `app_user` for the `DATA_DB_URL`.

- **Injection Protection:** The `sql_validator.py` module explicitly blocks keywords like `DROP`, `TRUNCATE`, and `GRANT` before execution.

- **Secrets:** Never commit your `.env` file. It is already added to `.gitignore`.

---

## 📚 Additional Resources

- **API Provider Documentation:**
  - [OpenRouter](https://openrouter.ai/docs)
  - [Google Gemini](https://ai.google.dev/)
  - [Anthropic Claude](https://docs.anthropic.com/)
  
- **PostgreSQL Setup:** [Official Installation Guide](https://www.postgresql.org/download/)

---

## 🤝 Support

If you encounter issues not covered in this guide:

1. Check the `test/config.md` file for advanced configuration options
2. Review the application logs in the Streamlit interface
3. Open an issue on the project repository with detailed error messages

---

**Happy querying! 🎉**
