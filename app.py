import streamlit as st
import pandas as pd
from uuid import UUID
import time
import os

# Local imports
import database
from models import ChatSession, ChatMessage
import ai_service
import sql_validator
from visualization_service import VisualizationService


st.set_page_config(page_title="NLP powered conversational Assistance", layout="wide")

# Load environment variables
from dotenv import load_dotenv

# Load local .env first (local development)
load_dotenv()

# --- Inject Streamlit secrets into os.environ (do not overwrite existing env vars) ---
def _cast_to_str(v):
    # Keep None as None
    if v is None:
        return None
    # For booleans and numbers, convert to string so os.environ has string values
    return str(v)

def inject_secrets_to_env(secrets_dict: dict):
    """
    Flatten top-level keys and nested sections into environment variables.
    Examples:
      GROQ_API_KEY -> GROQ_API_KEY
      [openrouter] api_url -> OPENROUTER_API_URL
    This will NOT overwrite existing environment variables.
    """
    if not isinstance(secrets_dict, dict):
        return

    for key, val in secrets_dict.items():
        if isinstance(val, dict):
            # nested section -> SECTION_KEY style
            for subk, subv in val.items():
                env_key = f"{key.upper()}_{subk.upper()}"
                if env_key not in os.environ and subv is not None:
                    os.environ[env_key] = _cast_to_str(subv)
        else:
            env_key = key.upper()
            if env_key not in os.environ and val is not None:
                os.environ[env_key] = _cast_to_str(val)

# Try to get st.secrets (works in Streamlit runtime). If missing, ignore.
try:
    # st.secrets is a mapping-like object; convert to a plain dict for processing
    secrets_map = dict(st.secrets) if hasattr(st, "secrets") else {}
except Exception:
    secrets_map = {}

inject_secrets_to_env(secrets_map)
# --- End injection ---

# Helper to read secret (prefers explicit env var, falls back to st.secrets)
def get_secret(key: str, section: str | None = None):
    """
    Priority:
      1. Environment variable (KEY or SECTION_KEY)
      2. st.secrets[section][key] (if section provided)
      3. st.secrets[key] (top-level)
    Returns raw values (strings or original types from st.secrets).
    """
    # 1) env var: if section provided, check SECTION_KEY first
    if section:
        env_key = f"{section.upper()}_{key.upper()}"
        if env_key in os.environ:
            return os.environ[env_key]

    # 2) env var without section
    env_key_plain = key.upper()
    if env_key_plain in os.environ:
        return os.environ[env_key_plain]

    # 3) st.secrets nested
    try:
        if section:
            sect = st.secrets.get(section, {})
            # try direct key then case-insensitive fallback
            if isinstance(sect, dict):
                if key in sect:
                    return sect.get(key)
                # case-insensitive search
                for k, v in sect.items():
                    if k.lower() == key.lower():
                        return v
        # top-level
        if key in st.secrets:
            return st.secrets.get(key)
        # case-insensitive top-level fallback
        for k, v in dict(st.secrets).items():
            if k.lower() == key.lower():
                return v
    except Exception:
        # Not running under Streamlit or st.secrets not available
        pass

    return None


# Initialize database (create tables if they don't exist)
database.init_db()

# Get database connections
app_db_conn = database.get_app_db_connection()
data_db_conn = database.get_data_db_connection()

if not app_db_conn or not data_db_conn:
    st.error("Failed to initialize database connections. Please check your .env file and database status.")
    st.stop()

# Validate API configuration on startup
# --- FIX: Correctly unpack the return values from get_model_config ---
sql_agent, validator_agent, is_configured, provider_info = ai_service.get_model_config()
if not is_configured:
    # Use the more detailed error message
    st.error("""
    ❌ No AI API configured!

    Please set at least one API key in your .env file:
    - GROQ_API_KEY (https://console.groq.com/keys) [FREE & FAST]
    - GOOGLE_API_KEY (https://aistudio.google.com/app/apikey) [FREE]
    - OPENROUTER_API_KEY (https://openrouter.ai/keys)
    - ANTHROPIC_API_KEY (https://console.anthropic.com/settings/keys)
    - OPENAI_API_KEY (https://platform.openai.com/api-keys)
    - DEEPSEEK_API_KEY (https://platform.deepseek.com/api_keys)

    Then set your models in .env:
    - SQL_AGENT_MODEL=your-sql-model
    - VALIDATOR_AGENT_MODEL=your-validator-model
    """)
    st.stop()
# --- END FIX ---

# Initialize visualization service
viz_service = VisualizationService()

# --- SESSION STATE INITIALIZATION ---

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id: UUID | None = None

if "request_count" not in st.session_state:
    st.session_state.request_count = 0

if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0

# Rate limiting configuration
MAX_REQUESTS_PER_MINUTE = 10


# --- HELPER FUNCTIONS ---

def get_filtered_schema():
    """Get schema with optional table filtering."""
    table_whitelist = None
    # Use get_secret so we check both env and st.secrets
    whitelist_env = get_secret("SCHEMA_TABLE_WHITELIST")
    if whitelist_env:
        # allow comma-separated values either from env string or list
        if isinstance(whitelist_env, (list, tuple)):
            table_whitelist = [t.strip() for t in whitelist_env]
        else:
            table_whitelist = [t.strip() for t in str(whitelist_env).split(",")]

    return ai_service.get_database_schema(data_db_conn.engine, table_whitelist)


def check_rate_limit() -> bool:
    """Check if user has exceeded rate limit."""
    current_time = time.time()

    # Reset counter if a minute has passed
    if current_time - st.session_state.last_request_time > 60:
        st.session_state.request_count = 0
        st.session_state.last_request_time = current_time

    # Check limit
    if st.session_state.request_count >= MAX_REQUESTS_PER_MINUTE:
        return False

    st.session_state.request_count += 1
    return True


def save_message(session_id: UUID, role: str, content: dict) -> UUID:
    """Saves a message and returns the message ID for reliable updates."""
    try:
        with app_db_conn.session as s:
            msg = ChatMessage(
                session_id=session_id,
                role=role,
                content=content
            )
            s.add(msg)
            s.commit()
            return msg.id
    except Exception as e:
        st.error(f"Error saving message: {e}")
        raise


def display_chat_messages(session_id: UUID):
    """Queries and displays all messages for the active chat session."""
    try:
        with app_db_conn.session as s:
            messages = s.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.created_at).all()

            for msg in messages:
                with st.chat_message(msg.role):
                    if msg.role == "user":
                        st.markdown(msg.content.get("text", "No content"))

                    elif msg.role == "assistant":
                        st.markdown(msg.content.get("explanation", "Here is the result:"))

                        sql_query = msg.content.get("sqlQuery", "# No SQL generated")
                        st.code(sql_query, language="sql")

                        # Show query stats if available
                        stats = sql_validator.get_validation_stats(sql_query)
                        if stats:
                            with st.expander("📊 Query Statistics"):
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Joins", stats.get('num_joins', 0))
                                col2.metric("Subqueries", stats.get('num_subqueries', 0))
                                col3.metric("Has Limit", "✅" if stats.get('has_limit') else "❌")

                        results_data = msg.content.get("results")
                        if results_data:
                            df = pd.DataFrame(results_data)
                            st.dataframe(df, use_container_width=True)
                            st.caption(f"Returned {len(df)} rows")

                            # Display saved visualization if exists
                            viz_data = msg.content.get("visualization")
                            if viz_data:
                                with st.expander("📈 Saved Visualization", expanded=False):
                                    st.caption(f"Chart Type: {viz_data['chart_type'].title()}")
                                    st.image(f"data:image/png;base64,{viz_data['image']}")
                        else:
                            st.info("Query executed successfully but returned no data.")

                    elif msg.role == "system":
                        st.error(msg.content.get("error", "An unknown system error occurred."))

    except Exception as e:
        st.error(f"Error loading chat history: {e}")


# --- CALLBACK FUNCTIONS ---

def set_active_chat(session_id: UUID):
    """Callback to set the active chat session."""
    st.session_state.active_chat_id = session_id


def new_chat_callback():
    """Callback to create a new chat session and set it as active."""
    try:
        with app_db_conn.session as s:
            session_count = s.query(ChatSession).count()
            new_session = ChatSession(title=f"Chat {session_count + 1}")
            s.add(new_session)
            s.commit()
            new_session_id = new_session.id

        st.session_state.active_chat_id = new_session_id

    except Exception as e:
        st.error(f"Error creating new chat: {e}")


def clear_schema_cache():
    """Callback to clear the schema cache."""
    ai_service.get_database_schema.clear()
    st.success("Schema cache cleared!")


# --- SIDEBAR UI ---

with st.sidebar:
    st.title("📊 NLP powered conversational Assistance")

    # Display current model configuration
    if sql_agent and validator_agent:
        st.success(f"✅ Two-Agent System Active")

        with st.expander("🤖 Agent Configuration", expanded=False):
            st.markdown(f"""
            **Agent 1: SQL Generator**
            - Model: `{sql_agent}`
            - Role: Convert language to SQL

            **Agent 2: Validator**
            - Model: `{validator_agent}`
            - Role: Validate and format output
            """)

            st.markdown("""
            **Change Agents:**
            Edit your `.env` file:
            ```env
            SQL_AGENT_MODEL=your-model
            VALIDATOR_AGENT_MODEL=your-model
            ```

            **Recommended Combos:**
            - Best: `deepseek/deepseek-chat` + `claude-3.5-sonnet`
            - Budget: `gemini-2.0-flash-exp` + `llama-3.3-70b-versatile` (Groq)
            - Fastest: `gemini-2.0-flash-exp` + `gemini-2.0-flash-exp`

            See `TWO_AGENT_SYSTEM.md` for details.
            """)

    col1, col2 = st.columns(2)
    with col1:
        st.button("New Chat", on_click=new_chat_callback, use_container_width=True)
    with col2:
        st.button("Clear Cache", on_click=clear_schema_cache, use_container_width=True)

    st.divider()
    st.markdown("### 💬 Chat History")

    try:
        with app_db_conn.session as s:
            sessions = s.query(ChatSession).order_by(ChatSession.created_at.desc()).all()

        if not sessions:
            st.caption("No chat history yet.")

        for session in sessions:
            st.button(
                session.title,
                key=f"session_btn_{session.id}",
                on_click=set_active_chat,
                args=(session.id,),
                use_container_width=True,
                type="primary" if st.session_state.active_chat_id == session.id else "secondary"
            )
    except Exception as e:
        st.error(f"Could not load chat sessions: {e}")

    # Footer with stats
    st.divider()
    st.caption(f"⚡ Rate Limit: {st.session_state.request_count}/{MAX_REQUESTS_PER_MINUTE} requests/min")
    st.caption("📈 Visualization: Enabled")

# --- MAIN CHAT WINDOW UI ---

if st.session_state.active_chat_id is None:
    st.info("👈 Select a chat from the sidebar or start a new one to begin.")

    st.markdown(f"""
    ### Welcome to Text-to-SQL Agent! 🚀

    This app uses a **two-agent system** to convert your questions into SQL queries:

    **🤖 Agent 1 (SQL Generator):** Converts natural language to SQL
    **🔍 Agent 2 (Validator):** Validates and formats the output

    **Example questions:**
    - "Show me all tables in the database"
    - "How many students are there?"
    - "What are the top 10 students by grade?"
    - "Show me students who enrolled this year"

    **Features:**
    - ✅ Two-agent validation system
    - ✅ Automatic error detection and correction
    - ✅ Query optimization and security checks
    - ✅ Chat history persistence
    - ✅ Intelligent data visualization

    **Current Configuration:**
    - SQL Agent: `{sql_agent}`
    - Validator Agent: `{validator_agent}`
    """)

else:
    # Display all historical messages
    display_chat_messages(st.session_state.active_chat_id)

    # Get new user input
    if prompt := st.chat_input("Ask a question about your data..."):

        # Input validation
        if len(prompt) > 1000:
            st.error("❌ Query too long. Please keep it under 1000 characters.")
            st.stop()

        if not prompt.strip():
            st.error("❌ Please enter a valid question.")
            st.stop()

        # Rate limiting
        if not check_rate_limit():
            st.error(
                f"❌ Rate limit exceeded. Please wait before sending more requests. ({MAX_REQUESTS_PER_MINUTE} requests per minute)")
            st.stop()

        # Save and display user message
        try:
            save_message(st.session_state.active_chat_id, "user", {"text": prompt})
        except Exception as e:
            st.error(f"Failed to save message: {e}")
            st.stop()

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- CORE AI/SQL PIPELINE ---
        with st.chat_message("assistant"):
            try:
                with st.spinner("Analyzing request and generating SQL..."):

                    # 1. Get Schema with filtering
                    schema = get_filtered_schema()
                    if "Error" in schema:
                        error_msg = f"Failed to get schema: {schema}"
                        st.error(error_msg)
                        save_message(st.session_state.active_chat_id, "system", {"error": error_msg})
                        st.stop()

                    # 2. Get AI Response (Explanation + SQL)
                    ai_response, error = ai_service.get_ai_response(prompt, schema)
                    if error:
                        st.error(f"AI Generation Error: {error}")
                        save_message(st.session_state.active_chat_id, "system", {"error": error})
                        st.stop()

                    # 3. Validate SQL
                    validated_sql, error = sql_validator.validate_sql_query(ai_response['sqlQuery'])
                    if error:
                        st.error(f"SQL Validation Error: {error}")
                        save_message(st.session_state.active_chat_id, "system", {"error": error})
                        st.stop()

                    # 4. Execute SQL
                    with st.spinner("Executing query..."):
                        results, error = sql_executor.execute_sql_query(validated_sql, data_db_conn.engine)
                        if error:
                            st.error(f"Execution Error: {error}")
                            save_message(st.session_state.active_chat_id, "system", {"error": error})
                            st.stop()

                    # 5. Prepare assistant content
                    assistant_content = {
                        "explanation": ai_response['explanation'],
                        "sqlQuery": validated_sql,
                        "results": results
                    }

                    # 6. Generate visualization if data exists
                    if results:
                        df = pd.DataFrame(results)
                        viz_analysis = viz_service.analyze_data_for_viz(df)

                        if viz_analysis['can_visualize']:
                            try:
                                chart_image = viz_service.create_chart(
                                    df,
                                    viz_analysis['recommended_chart'],
                                    x=viz_analysis.get('x_column'),
                                    y=viz_analysis.get('y_column'),
                                    title=f"{prompt[:50]}... - {viz_analysis['recommended_chart'].title()} Chart"
                                )
                                assistant_content['visualization'] = {
                                    'chart_type': viz_analysis['recommended_chart'],
                                    'image': chart_image
                                }
                            except Exception as viz_error:
                                st.warning(f"Visualization skipped: {str(viz_error)}")

                    # 7. Display results
                    st.markdown(assistant_content["explanation"])
                    st.code(validated_sql, language="sql")

                    # Show query stats
                    stats = sql_validator.get_validation_stats(validated_sql)
                    if stats:
                        with st.expander("📊 Query Statistics"):
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Joins", stats.get('num_joins', 0))
                            col2.metric("Subqueries", stats.get('num_subqueries', 0))
                            col3.metric("Has Limit", "✅" if stats.get('has_limit') else "❌")

                    if results:
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                        st.caption(f"Returned {len(df)} rows")

                        # --- VISUALIZATION SECTION ---
                        if 'visualization' in assistant_content:
                            with st.expander("📈 Data Visualization", expanded=True):
                                col1, col2 = st.columns([3, 1])

                                with col1:
                                    chart_type = assistant_content['visualization']['chart_type']
                                    st.info(f"💡 Recommended: {chart_type.title()} Chart")
                                    st.caption(viz_analysis['reason'])

                                with col2:
                                    chart_options = ['auto', 'line', 'bar', 'pie', 'scatter', 'histogram', 'heatmap',
                                                     'boxplot']
                                    new_chart_type = st.selectbox(
                                        "Change Chart Type",
                                        chart_options,
                                        index=0,
                                        key=f"chart_select_{st.session_state.active_chat_id}"
                                    )

                                st.image(f"data:image/png;base64,{assistant_content['visualization']['image']}")

                                if new_chart_type != 'auto' and new_chart_type != chart_type:
                                    if st.button("🔄 Regenerate with Selected Type",
                                                 key=f"regen_{st.session_state.active_chat_id}"):
                                        try:
                                            with st.spinner("Creating new visualization..."):
                                                new_chart_image = viz_service.create_chart(
                                                    df,
                                                    new_chart_type,
                                                    x=viz_analysis.get('x_column'),
                                                    y=viz_analysis.get('y_column'),
                                                    title=f"{prompt[:50]}... - {new_chart_type.title()} Chart"
                                                )
                                                assistant_content['visualization'] = {
                                                    'chart_type': new_chart_type,
                                                    'image': new_chart_image
                                                }
                                                st.rerun()
                                        except Exception as viz_error:
                                            st.error(f"Visualization Error: {str(viz_error)}")
                        else:
                            st.info(f"ℹ️ Visualization not available: {viz_analysis['reason']}")
                    else:
                        st.info("Query executed successfully but returned no data.")

                    # 8. Save complete message to database
                    save_message(st.session_state.active_chat_id, "assistant", assistant_content)

            except Exception as e:
                error_msg = f"Unexpected error in pipeline: {str(e)}"
                st.error(error_msg)
                try:
                    save_message(st.session_state.active_chat_id, "system", {"error": error_msg})
                except:
                    pass
