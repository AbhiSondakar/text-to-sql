import streamlit as st
import pandas as pd
from uuid import UUID
import time
import os
import datetime  # Ensure this is imported
import decimal  # Added for Decimal serialization
import json

# Local imports
import database
from models import ChatSession, ChatMessage
import ai_service
import sql_validator
from test import sql_executor
from visualization_service import VisualizationService

st.set_page_config(page_title="NLP powered conversational Assistance", layout="wide")

# Load environment variables
from dotenv import load_dotenv

# Load local .env first (local development)
load_dotenv()


# --- Inject Streamlit secrets into os.environ (do not overwrite existing env vars) ---
def _cast_to_str(v):
    if v is None:
        return None
    return str(v)


def inject_secrets_to_env(secrets_dict: dict):
    if not isinstance(secrets_dict, dict):
        return

    for key, val in secrets_dict.items():
        if isinstance(val, dict):
            for subk, subv in val.items():
                env_key = f"{key.upper()}_{subk.upper()}"
                if env_key not in os.environ and subv is not None:
                    os.environ[env_key] = _cast_to_str(subv)
        else:
            env_key = key.upper()
            if env_key not in os.environ and val is not None:
                os.environ[env_key] = _cast_to_str(val)


try:
    secrets_map = dict(st.secrets) if hasattr(st, "secrets") else {}
except Exception:
    secrets_map = {}

inject_secrets_to_env(secrets_map)


# --- End injection ---

# Helper to read secret
def get_secret(key: str, section: str | None = None):
    if section:
        env_key = f"{section.upper()}_{key.upper()}"
        if env_key in os.environ:
            return os.environ[env_key]

    env_key_plain = key.upper()
    if env_key_plain in os.environ:
        return os.environ[env_key_plain]

    try:
        if section:
            sect = st.secrets.get(section, {})
            if isinstance(sect, dict):
                if key in sect:
                    return sect.get(key)
                for k, v in sect.items():
                    if k.lower() == key.lower():
                        return v
        if key in st.secrets:
            return st.secrets.get(key)
        for k, v in dict(st.secrets).items():
            if k.lower() == key.lower():
                return v
    except Exception:
        pass
    return None


# Initialize database
database.init_db()

# Get database connections
app_db_conn = database.get_app_db_connection()
data_db_conn = database.get_data_db_connection()

if not app_db_conn or not data_db_conn:
    st.error("Failed to initialize database connections. Please check your .env file and database status.")
    st.stop()

# Validate API configuration
sql_agent, validator_agent, is_configured, provider_info = ai_service.get_model_config()
if not is_configured:
    st.error("""
    ❌ No AI API configured!
    Please set at least one API key in your .env file.
    """)
    st.stop()

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
    table_whitelist = None
    whitelist_env = get_secret("SCHEMA_TABLE_WHITELIST")
    if whitelist_env:
        if isinstance(whitelist_env, (list, tuple)):
            table_whitelist = [t.strip() for t in whitelist_env]
        else:
            table_whitelist = [t.strip() for t in str(whitelist_env).split(",")]

    return ai_service.get_database_schema(data_db_conn.engine, table_whitelist)


def check_rate_limit() -> bool:
    current_time = time.time()
    if current_time - st.session_state.last_request_time > 60:
        st.session_state.request_count = 0
        st.session_state.last_request_time = current_time

    if st.session_state.request_count >= MAX_REQUESTS_PER_MINUTE:
        return False

    st.session_state.request_count += 1
    return True


# --- NEW: Serialization Helper ---
def serialize_data(obj):
    """
    Recursively converts objects not supported by JSON (Date, Decimal, UUID)
    into strings/floats.
    """
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, list):
        return [serialize_data(item) for item in obj]
    if isinstance(obj, dict):
        return {k: serialize_data(v) for k, v in obj.items()}
    return obj


# ---------------------------------

def save_message(session_id: UUID, role: str, content: dict) -> UUID:
    """Saves a message and returns the message ID."""
    try:
        # Ensure content is JSON serializable before saving
        safe_content = serialize_data(content)

        with app_db_conn.session as s:
            msg = ChatMessage(
                session_id=session_id,
                role=role,
                content=safe_content
            )
            s.add(msg)
            s.commit()
            return msg.id
    except Exception as e:
        st.error(f"Error saving message: {e}")
        raise


def display_chat_messages(session_id: UUID):
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
    st.session_state.active_chat_id = session_id


def new_chat_callback():
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
    ai_service.get_database_schema.clear()
    st.success("Schema cache cleared!")


# --- SIDEBAR UI ---

with st.sidebar:
    st.title("📊 NLP powered conversational Assistance")

    if sql_agent and validator_agent:
        st.success(f"✅ Two-Agent System Active")

        with st.expander("🤖 Agent Configuration", expanded=False):
            st.markdown(f"""
            **Agent 1: SQL Generator**
            - Model: `{sql_agent}`
            **Agent 2: Validator**
            - Model: `{validator_agent}`
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

    st.divider()
    st.caption(f"⚡ Rate Limit: {st.session_state.request_count}/{MAX_REQUESTS_PER_MINUTE} requests/min")
    st.caption("📈 Visualization: Enabled")

# --- MAIN CHAT WINDOW UI ---

if st.session_state.active_chat_id is None:
    st.info("👈 Select a chat from the sidebar or start a new one to begin.")

else:
    display_chat_messages(st.session_state.active_chat_id)

    if prompt := st.chat_input("Ask a question about your data..."):

        if len(prompt) > 1000:
            st.error("❌ Query too long.")
            st.stop()

        if not prompt.strip():
            st.error("❌ Please enter a valid question.")
            st.stop()

        if not check_rate_limit():
            st.error("❌ Rate limit exceeded.")
            st.stop()

        try:
            save_message(st.session_state.active_chat_id, "user", {"text": prompt})
        except Exception as e:
            st.error(f"Failed to save message: {e}")
            st.stop()

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Analyzing request and generating SQL..."):

                    schema = get_filtered_schema()
                    if "Error" in schema:
                        error_msg = f"Failed to get schema: {schema}"
                        st.error(error_msg)
                        save_message(st.session_state.active_chat_id, "system", {"error": error_msg})
                        st.stop()

                    ai_response, error = ai_service.get_ai_response(prompt, schema)
                    if error:
                        st.error(f"AI Generation Error: {error}")
                        save_message(st.session_state.active_chat_id, "system", {"error": error})
                        st.stop()

                    validated_sql, error = sql_validator.validate_sql_query(ai_response['sqlQuery'])
                    if error:
                        st.error(f"SQL Validation Error: {error}")
                        save_message(st.session_state.active_chat_id, "system", {"error": error})
                        st.stop()

                    with st.spinner("Executing query..."):
                        results, error = sql_executor.execute_sql_query(validated_sql, data_db_conn.engine)
                        if error:
                            st.error(f"Execution Error: {error}")
                            save_message(st.session_state.active_chat_id, "system", {"error": error})
                            st.stop()

                    # Prepare content
                    # We will serialize 'results' inside save_message, so we can keep raw data for viz if needed here
                    assistant_content = {
                        "explanation": ai_response['explanation'],
                        "sqlQuery": validated_sql,
                        "results": results
                    }

                    # Generate visualization
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

                    # Display results
                    st.markdown(assistant_content["explanation"])
                    st.code(validated_sql, language="sql")

                    if results:
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                        st.caption(f"Returned {len(df)} rows")

                        if 'visualization' in assistant_content:
                            with st.expander("📈 Data Visualization", expanded=True):
                                st.info(
                                    f"💡 Recommended: {assistant_content['visualization']['chart_type'].title()} Chart")
                                st.image(f"data:image/png;base64,{assistant_content['visualization']['image']}")
                        else:
                            st.info(f"ℹ️ Visualization not available: {viz_analysis.get('reason', 'N/A')}")
                    else:
                        st.info("Query executed successfully but returned no data.")

                    # Save complete message to database (save_message handles serialization)
                    save_message(st.session_state.active_chat_id, "assistant", assistant_content)

            except Exception as e:
                error_msg = f"Unexpected error in pipeline: {str(e)}"
                st.error(error_msg)
                try:
                    save_message(st.session_state.active_chat_id, "system", {"error": error_msg})
                except:
                    pass
