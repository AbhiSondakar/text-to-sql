"""
AI Service Module - Production Ready
Handles multi-provider AI API calls with robust error handling, retry logic, and monitoring.
"""

import os
import time
import logging
from functools import wraps
from typing import TypedDict, Tuple, Optional, Dict, Any, List
import streamlit as st
import requests
import json
import re
from sqlalchemy import inspect, Engine
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AISqlResponse(TypedDict):
    explanation: str
    sqlQuery: str


class ProviderConfig(TypedDict):
    provider: str
    api_key: str
    api_url: str
    model_name: str


class APIError(Exception):
    """Custom exception for API errors with detailed context"""

    def __init__(self, message: str, provider: str, status_code: int = None, response_text: str = None):
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(self.message)


# --- FIX: Add validation function (Issue 15) ---
def validate_model_name(model: str) -> bool:
    """Validate model name format for security."""
    if not model or not model.strip():
        logger.error("Model name is empty or None")
        return False

    # Check for injection attempts or dangerous characters
    dangerous_chars = [';', '--', '/*', '*/', 'DROP', 'DELETE', 'INSERT', 'UPDATE']
    model_upper = model.upper()
    if any(char in model_upper for char in dangerous_chars):
        logger.error(f"Potentially malicious model name detected: {model}")
        return False

    # Check for reasonable length
    if len(model) > 100:
        logger.error(f"Model name is excessively long: {model}")
        return False

    return True


# --- END FIX ---


# Request session with connection pooling and retry logic
def get_requests_session() -> requests.Session:
    """
    Create requests session with retry logic and connection pooling.
    Production-ready configuration for reliability.
    """
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        backoff_factor=1  # 1s, 2s, 4s delays
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=20
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


# Global session instance (reused for connection pooling)
_session = None


def get_session() -> requests.Session:
    """Get or create global session instance"""
    global _session
    if _session is None:
        _session = get_requests_session()
    return _session


def rate_limit_handler(func):
    """Decorator to handle rate limiting"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except APIError as e:
                if e.status_code == 429 and attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise
        return None, "Rate limit exceeded after retries"

    return wrapper


def validate_api_key(api_key: str, provider: str) -> bool:
    """
    Validate API key format for basic security.
    Returns True if key looks valid, False otherwise.
    """
    if not api_key or api_key.strip() == "":
        return False

    # Check for placeholder values
    placeholders = ['your-key', 'your-api-key', 'xxx', 'placeholder', 'here']
    if any(placeholder in api_key.lower() for placeholder in placeholders):
        logger.error(f"{provider}: API key appears to be a placeholder")
        return False

    # Provider-specific validation
    key_patterns = {
        'openrouter': 'sk-or-v1-',
        'anthropic': 'sk-ant-',
        'openai': 'sk-',
        'google': 'AIzaSy',
        'deepseek': 'sk-',
        'groq': 'gsk_'
    }

    if provider in key_patterns:
        expected_prefix = key_patterns[provider]
        if not api_key.startswith(expected_prefix):
            logger.warning(f"{provider}: API key doesn't start with expected prefix '{expected_prefix}'")
            # Don't fail - some providers may have different formats

    return True


def detect_provider(model: str) -> str:
    """
    Detect provider from model name with improved pattern matching.
    """
    if not model:
        return 'openrouter'

    model_lower = model.lower()

    # Explicit prefix detection
    if model_lower.startswith('openrouter/'):
        return 'openrouter'

    # Provider-specific patterns (order matters - most specific first)
    provider_patterns = {
        'groq': ['llama-3.3', 'llama-3.2', 'llama-3.1', 'mixtral', 'gemma2', 'groq'],
        'anthropic': ['claude'],
        'google': ['gemini'],
        'deepseek': ['deepseek'],
        'openai': ['gpt-4', 'gpt-3.5', 'o1-', 'o3-']
    }

    for provider, patterns in provider_patterns.items():
        if any(pattern in model_lower for pattern in patterns):
            return provider

    # Default fallback
    logger.warning(f"Could not detect provider for model: {model}. Using OpenRouter as default.")
    return 'openrouter'


def get_provider_config(model: str) -> Optional[ProviderConfig]:
    """
    Get provider configuration with enhanced validation and error messages.
    Production-ready with detailed diagnostics.
    """
    # --- FIX: Use validation function (Issue 15) ---
    if not validate_model_name(model):
        st.error(f"❌ Invalid or potentially malicious model name specified: {model}")
        return None
    # --- END FIX ---

    provider = detect_provider(model)
    clean_model = model.replace('openrouter/', '').strip()

    logger.info(f"Configuring provider: {provider} for model: {clean_model}")

    # Provider configuration map
    provider_configs = {
        'openrouter': {
            'key_env': 'OPENROUTER_API_KEY',
            'url_env': 'OPENROUTER_API_URL',
            'default_url': 'https://openrouter.ai/api/v1',
            'docs_url': 'https://openrouter.ai/keys'
        },
        'anthropic': {
            'key_env': 'ANTHROPIC_API_KEY',
            'url_env': 'ANTHROPIC_API_URL',
            'default_url': 'https://api.anthropic.com/v1',
            'docs_url': 'https://console.anthropic.com/settings/keys'
        },
        'openai': {
            'key_env': 'OPENAI_API_KEY',
            'url_env': 'OPENAI_API_URL',
            'default_url': 'https://api.openai.com/v1',
            'docs_url': 'https://platform.openai.com/api-keys'
        },
        'google': {
            'key_env': 'GOOGLE_API_KEY',
            'url_env': 'GOOGLE_API_URL',
            'default_url': 'https://generativelanguage.googleapis.com/v1beta',
            'docs_url': 'https://aistudio.google.com/app/apikey'
        },
        'deepseek': {
            'key_env': 'DEEPSEEK_API_KEY',
            'url_env': 'DEEPSEEK_API_URL',
            'default_url': 'https://api.deepseek.com/v1',
            'docs_url': 'https://platform.deepseek.com/api_keys'
        },
        'groq': {
            'key_env': 'GROQ_API_KEY',
            'url_env': 'GROQ_API_URL',
            'default_url': 'https://api.groq.com/openai/v1',
            'docs_url': 'https://console.groq.com/keys'
        }
    }

    if provider not in provider_configs:
        logger.error(f"Unsupported provider: {provider}")
        st.error(f"❌ Unsupported provider: {provider}")
        return None

    config = provider_configs[provider]
    api_key = os.getenv(config['key_env'])
    api_url = os.getenv(config['url_env'], config['default_url'])

    # Validate API key
    if not validate_api_key(api_key, provider):
        logger.error(f"{provider}: Invalid or missing API key")
        st.error(f"""
        ❌ {provider.title()} API key required for model: {model}

        **Setup Instructions:**
        1. Get your API key from: {config['docs_url']}
        2. Add to your .env file:
        ```
        {config['key_env']}=your-actual-key-here
        ```
        3. Restart the application

        **Security Note:** Never commit API keys to version control!
        """)
        return None

    logger.info(f"✅ {provider.title()} configured successfully")

    return {
        'provider': provider,
        'api_key': api_key,
        'api_url': api_url,
        'model_name': clean_model
    }


def get_model_config() -> Tuple[Optional[str], Optional[str], bool, Dict[str, str]]:
    """
    Get model configuration with fallback defaults.
    Production-ready with comprehensive provider detection.
    """
    sql_agent_model = os.getenv("SQL_AGENT_MODEL", "gemini-2.0-flash-exp")
    validator_agent_model = os.getenv("VALIDATOR_AGENT_MODEL", "llama-3.3-70b-versatile")

    # Check configured providers
    providers_status = {
        'openrouter': bool(validate_api_key(os.getenv("OPENROUTER_API_KEY"), 'openrouter')),
        'anthropic': bool(validate_api_key(os.getenv("ANTHROPIC_API_KEY"), 'anthropic')),
        'openai': bool(validate_api_key(os.getenv("OPENAI_API_KEY"), 'openai')),
        'google': bool(validate_api_key(os.getenv("GOOGLE_API_KEY"), 'google')),
        'deepseek': bool(validate_api_key(os.getenv("DEEPSEEK_API_KEY"), 'deepseek')),
        'groq': bool(validate_api_key(os.getenv("GROQ_API_KEY"), 'groq'))
    }

    available_providers = [k for k, v in providers_status.items() if v]
    is_configured = len(available_providers) > 0

    provider_info = {
        'sql_agent_provider': detect_provider(sql_agent_model),
        'validator_agent_provider': detect_provider(validator_agent_model),
        'available_providers': available_providers,
        'providers_status': providers_status
    }

    logger.info(f"Model config: SQL={sql_agent_model}, Validator={validator_agent_model}")
    logger.info(f"Available providers: {', '.join(available_providers)}")

    return sql_agent_model, validator_agent_model, is_configured, provider_info


@st.cache_data(ttl=600)
def get_database_schema(_engine: Engine, table_whitelist: Optional[List[str]] = None) -> str:
    """
    Get database schema with improved error handling and performance.
    Cached for 10 minutes to reduce database load.
    """
    try:
        logger.info("Fetching database schema...")
        inspector = inspect(_engine)
        table_names = inspector.get_table_names()

        if not table_names:
            logger.warning("No tables found in database")
            return "Error: No tables found in the database."

        # Apply whitelist filter
        if table_whitelist:
            original_count = len(table_names)
            table_names = [t for t in table_names if t in table_whitelist]
            logger.info(f"Whitelist filter: {len(table_names)}/{original_count} tables")

            if not table_names:
                return f"Error: No tables match whitelist: {table_whitelist}"

        # Limit schema size for performance
        max_tables = int(os.getenv("MAX_SCHEMA_TABLES", "50"))
        if len(table_names) > max_tables:
            logger.warning(f"Large schema: limiting to {max_tables} tables")
            st.warning(f"⚠️ Large schema detected: showing {max_tables} of {len(table_names)} tables.")
            table_names = table_names[:max_tables]

        # Build schema string
        schema_str = ""
        for table_name in table_names:
            try:
                columns = inspector.get_columns(table_name)
                schema_str += f"CREATE TABLE {table_name} (\n"

                col_definitions = []
                for column in columns:
                    col_name = column['name']
                    col_type = str(column['type'])
                    col_definitions.append(f"  {col_name} {col_type}")

                schema_str += ",\n".join(col_definitions)
                schema_str += "\n);\n\n"

            except Exception as e:
                logger.error(f"Error reading table {table_name}: {e}")
                continue

        logger.info(f"✅ Schema fetched successfully: {len(table_names)} tables")
        return schema_str

    except Exception as e:
        logger.error(f"Schema inspection failed: {e}", exc_info=True)
        return f"Error inspecting schema: {str(e)}"


@rate_limit_handler
def call_api(
        provider_config: ProviderConfig,
        messages: list,
        temperature: float = 0.3,
        force_json: bool = True,
        timeout: int = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Universal API caller with robust error handling and monitoring.
    Production-ready with retries, timeouts, and detailed logging.
    """
    provider = provider_config['provider']
    api_key = provider_config['api_key']
    api_url = provider_config['api_url']
    model = provider_config['model_name']

    # Use environment variable or default
    if timeout is None:
        timeout = int(os.getenv('API_TIMEOUT', '60'))

    logger.info(f"API call: provider={provider}, model={model}, temp={temperature}")

    try:
        session = get_session()

        # OpenAI-compatible providers
        if provider in ['openrouter', 'openai', 'deepseek', 'groq']:
            endpoint = api_url.rstrip('/')
            if not endpoint.endswith('/chat/completions'):
                if not endpoint.endswith('/v1'):
                    endpoint += '/v1'
                endpoint += '/chat/completions'

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            if provider == 'openrouter':
                headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERER", "https://github.com/text-to-sql")
                headers["X-Title"] = os.getenv("OPENROUTER_TITLE", "Text-to-SQL Agent")

            payload = {
                "model": model,
                "temperature": temperature,
                "messages": messages,
            }

            # JSON mode support (Groq doesn't support it)
            if force_json and provider != 'groq':
                payload["response_format"] = {"type": "json_object"}

            start_time = time.time()
            resp = session.post(endpoint, headers=headers, json=payload, timeout=timeout)
            elapsed = time.time() - start_time

            logger.info(f"API response: {provider} - {resp.status_code} - {elapsed:.2f}s")

            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content")

            # Log token usage if available
            usage = data.get("usage", {})
            if usage:
                logger.info(f"Token usage: {usage}")

            return content, None

        # Anthropic API
        elif provider == 'anthropic':
            endpoint = f"{api_url.rstrip('/')}/messages"

            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }

            # Convert message format
            system_message = ""
            anthropic_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message += msg["content"] + "\n"
                else:
                    anthropic_messages.append(msg)

            payload = {
                "model": model,
                "max_tokens": 4096,
                "temperature": temperature,
                "messages": anthropic_messages
            }

            if system_message:
                payload["system"] = system_message.strip()

            start_time = time.time()
            resp = session.post(endpoint, headers=headers, json=payload, timeout=timeout)
            elapsed = time.time() - start_time

            logger.info(f"API response: {provider} - {resp.status_code} - {elapsed:.2f}s")

            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", [{}])[0].get("text", "")

            # Log usage
            usage = data.get("usage", {})
            if usage:
                logger.info(f"Token usage: {usage}")

            return content, None

        # Google Gemini API
        elif provider == 'google':
            endpoint = f"{api_url.rstrip('/')}/models/{model}:generateContent?key={api_key}"

            headers = {
                "Content-Type": "application/json"
            }

            # Convert message format
            parts = []
            for msg in messages:
                parts.append({"text": f"{msg['role']}: {msg['content']}"})

            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "temperature": temperature,
                }
            }

            # --- FIX for Google Gemini JSON ---
            if force_json:
                payload["generationConfig"]["responseMimeType"] = "application/json"
            # --- END FIX ---

            start_time = time.time()
            resp = session.post(endpoint, headers=headers, json=payload, timeout=timeout)
            elapsed = time.time() - start_time

            logger.info(f"API response: {provider} - {resp.status_code} - {elapsed:.2f}s")

            resp.raise_for_status()
            data = resp.json()

            if not data.get("candidates"):
                error_msg = f"API returned no candidates. Response: {data}"
                logger.error(error_msg)
                raise APIError(error_msg, provider, response_text=str(data))

            content = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

            return content, None

        else:
            raise APIError(f"Unsupported provider: {provider}", provider)

    except requests.exceptions.Timeout:
        error_msg = f"Request timeout after {timeout}s"
        logger.error(f"{provider}: {error_msg}")
        raise APIError(error_msg, provider)

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else None
        error_detail = e.response.text if e.response else str(e)

        logger.error(f"{provider} HTTP error {status_code}: {error_detail}")

        # Parse error details
        try:
            error_json = e.response.json()
            if provider == 'google':
                error_msg = error_json.get('error', {}).get('message', error_detail)
            elif provider == 'anthropic':
                error_msg = error_json.get('error', {}).get('message', error_detail)
            else:  # OpenAI-compatible
                error_msg = error_json.get('error', {}).get('message', error_detail)
        except:
            error_msg = error_detail

        raise APIError(error_msg, provider, status_code, error_detail)

    except requests.exceptions.RequestException as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(f"{provider}: {error_msg}")
        raise APIError(error_msg, provider)

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"{provider}: {error_msg}", exc_info=True)
        raise APIError(error_msg, provider)


def parse_json_response(content: str) -> Optional[dict]:
    """
    Robust JSON parsing with multiple fallback strategies.
    Production-ready with comprehensive error handling.
    """
    if not content:
        logger.warning("Empty content received")
        return None

    if isinstance(content, dict):
        return content

    content = content.strip()

    # Remove markdown code blocks
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*', '', content)
    content = content.strip()

    # Strategy 1: Direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parse failed: {e}")

    # Strategy 2: Extract from code blocks
    json_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{[^{}]*"explanation"[^{}]*"sqlQuery"[^{}]*\}',
        r'\{.*?\}',
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                cleaned = match.strip()
                cleaned = re.sub(r'```json|```', '', cleaned).strip()
                parsed = json.loads(cleaned)
                logger.info("JSON parsed successfully using pattern matching")
                return parsed
            except json.JSONDecodeError:
                continue

    # Strategy 3: Find JSON object boundaries
    start = content.find('{')
    end = content.rfind('}')

    if start != -1 and end != -1 and end > start:
        try:
            json_str = content[start:end + 1]
            parsed = json.loads(json_str)
            logger.info("JSON parsed successfully using boundary detection")
            return parsed
        except json.JSONDecodeError:
            pass

    logger.error("All JSON parsing strategies failed")
    return None


def sql_generation_agent(
        user_prompt: str,
        schema: str,
        model: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Agent 1: SQL Generation Agent
    Production-ready with comprehensive error handling.
    """
    logger.info(f"SQL Generation Agent started: model={model}")

    provider_config = get_provider_config(model)
    if not provider_config:
        return None, f"Failed to configure provider for model: {model}"

    system_prompt = f"""You are a SQL Generation Specialist. Your ONLY task is to convert natural language questions into PostgreSQL queries.

RULES:
1. Generate ONLY SELECT queries (read-only)
2. Use proper PostgreSQL syntax
3. Return a single, executable SQL query
4. If data doesn't exist, query information_schema to show what's available
5. Always use explicit column names (avoid SELECT *)
6. Add LIMIT clauses for safety (default: 100)
7. Use proper JOINs when needed
8. Handle edge cases gracefully

DATABASE SCHEMA:
{schema}

OUTPUT FORMAT:
Return ONLY a valid SQL query string. No explanation, no markdown, just the SQL query.

Example outputs:
SELECT name, age FROM students LIMIT 10
SELECT COUNT(*) as total_students FROM students
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' LIMIT 20
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate SQL for: {user_prompt}"}
    ]

    st.info(f"🤖 **Agent 1 (SQL Generator):** {provider_config['provider']} - {provider_config['model_name']}")

    try:
        content, error = call_api(
            provider_config=provider_config,
            messages=messages,
            temperature=0.1,
            force_json=False
        )

        if error:
            logger.error(f"SQL Generation failed: {error}")
            return None, f"SQL Generation Agent error: {error}"

        if not content:
            return None, "SQL Generation Agent returned empty response"

        # Clean SQL query
        sql_query = content.strip()
        sql_query = re.sub(r'```sql\s*', '', sql_query)
        sql_query = re.sub(r'```\s*', '', sql_query)
        sql_query = sql_query.strip()

        # Remove comments
        lines = sql_query.split('\n')
        sql_lines = [line for line in lines if not line.strip().startswith('--') and line.strip()]
        sql_query = '\n'.join(sql_lines).strip()

        logger.info(f"✅ SQL query generated successfully")
        st.success(f"✅ **Agent 1:** Generated SQL query")
        st.code(sql_query, language="sql")

        return sql_query, None

    except APIError as e:
        error_msg = f"API Error: {e.message}"
        logger.error(error_msg)
        return None, error_msg

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg


def validation_and_formatting_agent(
        user_prompt: str,
        sql_query: str,
        schema: str,
        model: str
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Agent 2: Validation and Formatting Agent
    Production-ready with comprehensive validation.
    """
    logger.info(f"Validation Agent started: model={model}")

    provider_config = get_provider_config(model)
    if not provider_config:
        return None, f"Failed to configure provider for model: {model}"

    system_prompt = f"""You are a SQL Validation and Quality Assurance Specialist.

Your tasks:
1. VALIDATE the SQL query for correctness and safety
2. CHECK if it matches the user's intent
3. VERIFY it uses only tables/columns from the schema
4. EXPLAIN what the query does in simple terms
5. SUGGEST improvements if needed (but keep original if valid)

DATABASE SCHEMA:
{schema}

VALIDATION CHECKLIST:
✓ Query uses SELECT only (no INSERT/UPDATE/DELETE/DROP)
✓ All tables exist in schema
✓ All columns referenced exist
✓ JOINs are properly structured
✓ Query has LIMIT clause for safety
✓ Query actually answers the user's question
✓ Syntax is valid PostgreSQL

OUTPUT FORMAT (STRICT JSON):
{{
  "explanation": "One-sentence explanation of what the query does",
  "sqlQuery": "The final validated SQL query (may be improved version)",
  "validation_status": "valid|needs_fix|invalid",
  "issues_found": ["list of any issues"],
  "improvements_made": ["list of improvements if any"]
}}

CRITICAL: Return ONLY the JSON object, nothing else. No markdown, no extra text.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""
User asked: "{user_prompt}"

Generated SQL:
{sql_query}

Please validate this SQL query and format the response.
"""}
    ]

    st.info(f"🔍 **Agent 2 (Validator):** {provider_config['provider']} - {provider_config['model_name']}")

    try:
        # Groq doesn't support JSON mode, so we don't force it
        force_json_mode = provider_config['provider'] != 'groq'

        content, error = call_api(
            provider_config=provider_config,
            messages=messages,
            temperature=0.2,
            force_json=force_json_mode
        )

        if error:
            logger.error(f"Validation failed: {error}")
            return None, f"Validation Agent error: {error}"

        if not content:
            return None, "Validation Agent returned empty response"

        parsed = parse_json_response(content)

        if not parsed:
            logger.error(f"Failed to parse JSON response: {content[:300]}")
            return None, f"Validation Agent returned invalid JSON: {content[:300]}..."

        if "explanation" not in parsed or "sqlQuery" not in parsed:
            logger.error(f"Missing required fields in response: {list(parsed.keys())}")
            return None, f"Validation Agent missing required fields. Got: {list(parsed.keys())}"

        validation_status = parsed.get("validation_status", "unknown")
        issues = parsed.get("issues_found", [])
        improvements = parsed.get("improvements_made", [])

        if validation_status == "valid":
            st.success("✅ **Agent 2:** SQL query validated successfully")
        elif validation_status == "needs_fix":
            st.warning("⚠️ **Agent 2:** SQL query needs minor improvements (applied)")
        else:
            st.error("❌ **Agent 2:** SQL query has issues")

        if issues:
            with st.expander("⚠️ Issues Found"):
                for issue in issues:
                    st.write(f"- {issue}")

        if improvements:
            with st.expander("✨ Improvements Made"):
                for improvement in improvements:
                    st.write(f"- {improvement}")

        logger.info(f"✅ Validation completed: status={validation_status}")

        return {
            "explanation": parsed["explanation"],
            "sqlQuery": parsed["sqlQuery"]
        }, None

    except APIError as e:
        error_msg = f"API Error: {e.message}"
        logger.error(error_msg)
        return None, error_msg

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg


def get_ai_response(
        user_prompt: str,
        schema: str,
        image_url: Optional[str] = None
) -> Tuple[Optional[AISqlResponse], Optional[str]]:
    """
    Two-Agent System for SQL Generation and Validation
    Production-ready with comprehensive monitoring and error handling.
    """
    logger.info(f"AI Response pipeline started for prompt: {user_prompt[:100]}")

    try:
        sql_agent_model, validator_agent_model, is_configured, provider_info = get_model_config()

        if not is_configured:
            error_msg = (
                "❌ No AI API configured!\n\n"
                "Please set at least one API key in your .env file:\n"
                "- GROQ_API_KEY (https://console.groq.com/keys) [FREE & FAST]\n"
                "- GOOGLE_API_KEY (https://aistudio.google.com/app/apikey) [FREE]\n"
                "- OPENROUTER_API_KEY (https://openrouter.ai/keys)\n"
                "- ANTHROPIC_API_KEY (https://console.anthropic.com/settings/keys)\n"
                "- OPENAI_API_KEY (https://platform.openai.com/api-keys)\n"
                "- DEEPSEEK_API_KEY (https://platform.deepseek.com/api_keys)\n\n"
                "Then set your models:\n"
                "- SQL_AGENT_MODEL\n"
                "- VALIDATOR_AGENT_MODEL"
            )
            logger.error("No AI providers configured")
            return None, error_msg

        # Display agent configuration
        st.info(f"""
**🤖 Two-Agent System Active**
- **Agent 1 (SQL Generator):** {sql_agent_model} ({provider_info['sql_agent_provider']})
- **Agent 2 (Validator):** {validator_agent_model} ({provider_info['validator_agent_provider']})
- **Available Providers:** {', '.join(provider_info['available_providers'])}
        """)

        # STEP 1: SQL Generation Agent
        st.markdown("---")
        sql_query, error = sql_generation_agent(
            user_prompt=user_prompt,
            schema=schema,
            model=sql_agent_model
        )

        if error:
            logger.error(f"SQL generation failed: {error}")
            return None, error

        if not sql_query:
            logger.error("SQL generation returned empty query")
            return None, "SQL Generation Agent failed to produce a query"

        # STEP 2: Validation and Formatting Agent
        st.markdown("---")
        formatted_response, error = validation_and_formatting_agent(
            user_prompt=user_prompt,
            sql_query=sql_query,
            schema=schema,
            model=validator_agent_model
        )

        if error:
            logger.error(f"Validation failed: {error}")
            st.error(f"❌ Validation agent failed: {error}")
            # Fallback: Use the original SQL query if validation fails
            st.warning("⚠️ Validation agent failed. Falling back to original SQL query.")
            return {
                "explanation": "Validation agent failed. Using the raw SQL query.",
                "sqlQuery": sql_query
            }, None

        st.markdown("---")
        st.success("✅ **Both agents completed successfully!**")
        logger.info("✅ AI response pipeline completed successfully")

        return formatted_response, None

    except Exception as e:
        error_msg = f"Unexpected error in two-agent system: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg


# Backwards compatibility functions
def get_available_models() -> List[str]:
    """Returns list of configured models for display purposes."""
    sql_agent, validator_agent, _, _ = get_model_config()
    return [sql_agent, validator_agent] if sql_agent and validator_agent else []


def select_best_model() -> str:
    """Returns the SQL generation agent model."""
    sql_agent, _, _, _ = get_model_config()
    return sql_agent or "gemini-2.0-flash-exp"