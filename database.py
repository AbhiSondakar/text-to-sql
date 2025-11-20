import os
import logging
import streamlit as st
from sqlalchemy import text, create_engine
from sqlalchemy.exc import OperationalError, ProgrammingError
from models import Base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define connection names
APP_DB_CONNECTION_NAME = "app_db"
DATA_DB_CONNECTION_NAME = "data_db"


def test_connection_string(db_url: str, db_name: str) -> tuple[bool, str]:
    """Test if a database connection string works."""
    try:
        engine = create_engine(db_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"✅ {db_name} connection test passed")
        return True, "Connection successful"
    except OperationalError as e:
        error_msg = str(e.orig) if hasattr(e, 'orig') else str(e)
        logger.error(f"❌ {db_name} connection failed: {error_msg}")
        return False, error_msg
    except Exception as e:
        logger.error(f"❌ {db_name} unexpected error: {str(e)}")
        return False, str(e)


@st.cache_resource
def get_app_db_connection():
    """
    Returns the Streamlit SQLConnection for the application's R/W database.
    Uses st.cache_resource to ensure the connection object is reused across reruns.
    """
    try:
        app_db_url = os.getenv("APP_DB_URL")

        if not app_db_url:
            st.error("""
            ❌ APP_DB_URL not configured!

            Please set APP_DB_URL in your .env file:
            ```
            APP_DB_URL=postgresql://app_user:password@localhost:5432/text_to_sql_db
            ```

            Note: This user needs WRITE permissions to store chat history.
            """)
            return None

        # Test connection first
        success, error_msg = test_connection_string(app_db_url, "APP_DB")
        if not success:
            st.error(f"""
            ❌ Failed to connect to Application Database

            **Error:** {error_msg}

            **Common fixes:**
            1. Check if PostgreSQL is running: `pg_isready`
            2. Verify database exists: `psql -U postgres -c "\\l"`
            3. Test credentials: `psql "{app_db_url}"`
            4. Check pg_hba.conf for authentication settings

            **Expected format:**
            ```
            postgresql://username:password@host:port/database
            ```
            """)
            return None

        # Create Streamlit connection
        conn = st.connection(APP_DB_CONNECTION_NAME, type="sql", url=app_db_url)

        # Verify write permissions
        try:
            with conn.session as s:
                s.execute(text("CREATE TEMP TABLE _write_test (id INT)"))
                s.execute(text("DROP TABLE _write_test"))
            logger.info("✅ APP_DB has write permissions")
        except Exception as e:
            st.warning(f"""
            ⚠️ APP_DB may not have write permissions!

            Error: {str(e)}

            The application database needs write access to store chat history.
            Grant full permissions:
            ```sql
            GRANT ALL PRIVILEGES ON DATABASE text_to_sql_db TO app_user;
            GRANT ALL PRIVILEGES ON SCHEMA public TO app_user;
            ```
            """)

        return conn

    except Exception as e:
        logger.error(f"Failed to initialize APP_DB connection: {e}")
        st.error(f"""
        ❌ Error initializing Application Database connection

        **Error:** {str(e)}

        Check the Streamlit terminal for detailed error logs.
        """)
        return None


@st.cache_resource
def get_data_db_connection():
    """
    Returns the Streamlit SQLConnection for the data warehouse (Read-Only).
    Validates that the connection is actually read-only.
    """
    try:
        data_db_url = os.getenv("DATA_DB_URL")

        if not data_db_url:
            st.error("""
            ❌ DATA_DB_URL not configured!

            Please set DATA_DB_URL in your .env file:
            ```
            DATA_DB_URL=postgresql://readonly_user:password@localhost:5432/Student
            ```

            Note: This user should have READ-ONLY permissions for security.
            """)
            return None

        # Test connection first
        success, error_msg = test_connection_string(data_db_url, "DATA_DB")
        if not success:
            st.error(f"""
            ❌ Failed to connect to Data Database

            **Error:** {error_msg}

            **Common fixes:**
            1. Check if database exists: `psql -U postgres -c "\\l" | grep Student`
            2. Test credentials: `psql "{data_db_url}"`
            3. Verify SELECT permissions:
            ```sql
            GRANT USAGE ON SCHEMA public TO readonly_user;
            GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
            ```
            """)
            return None

        # Create Streamlit connection
        conn = st.connection(DATA_DB_CONNECTION_NAME, type="sql", url=data_db_url)

        # Validate read-only permissions
        is_readonly = validate_readonly_connection(conn)
        if not is_readonly:
            st.warning("""
            ⚠️ WARNING: DATA_DB connection has WRITE permissions!

            **Security Risk:** The AI could generate harmful SQL queries.

            **Fix:** Create a read-only user:
            ```sql
            -- Revoke write permissions
            REVOKE INSERT, UPDATE, DELETE, TRUNCATE 
            ON ALL TABLES IN SCHEMA public FROM your_user;

            -- Or create new read-only user
            CREATE USER readonly_user WITH PASSWORD 'secure_pass';
            GRANT CONNECT ON DATABASE Student TO readonly_user;
            GRANT USAGE ON SCHEMA public TO readonly_user;
            GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
            ```
            """)
        else:
            logger.info("✅ DATA_DB is read-only (verified)")

        return conn

    except Exception as e:
        logger.error(f"Failed to initialize DATA_DB connection: {e}")
        st.error(f"""
        ❌ Error initializing Data Database connection

        **Error:** {str(e)}

        Check the Streamlit terminal for detailed error logs.
        """)
        return None


def validate_readonly_connection(conn) -> bool:
    """
    Validates that the database connection is truly read-only.

    Tests:
    1. Check if transaction is read-only
    2. Attempt to create a temporary table (should fail)

    Returns True if read-only, False if has write permissions.
    """
    try:
        with conn.session as s:
            # Test 1: Check transaction_read_only setting
            try:
                result = s.execute(text("SHOW transaction_read_only"))
                row = result.fetchone()
                if row and row[0] == 'on':
                    return True
            except:
                pass  # Setting might not be available

            # Test 2: Try to create a temp table (should fail if read-only)
            try:
                s.execute(text("CREATE TEMP TABLE _readonly_test (id INT)"))
                s.rollback()  # Has write access
                logger.warning("DATA_DB has write permissions")
                return False
            except (ProgrammingError, OperationalError) as e:
                # Permission denied - this is good!
                s.rollback()
                logger.info("DATA_DB write test failed (read-only confirmed)")
                return True

        # If we can't definitively prove read-only, assume it's not
        return False

    except Exception as e:
        logger.error(f"Error validating read-only status: {e}")
        return False


def init_db():
    """
    Initializes the application database by creating tables defined in models.py.
    Also validates database connections.
    """
    logger.info("Initializing databases...")

    # Get connections
    app_db = get_app_db_connection()
    data_db = get_data_db_connection()

    if not app_db:
        st.error("❌ Application database connection failed. Cannot continue.")
        st.stop()

    if not data_db:
        st.error("❌ Data database connection failed. Cannot continue.")
        st.stop()

    try:
        # Create tables in application database
        logger.info("Creating application database tables...")
        engine = app_db.engine
        Base.metadata.create_all(engine)
        logger.info("✅ Application database tables created/verified")

        # Verify we can query the data database
        with data_db.session as s:
            result = s.execute(text("""
                                    SELECT COUNT(*)
                                    FROM information_schema.tables
                                    WHERE table_schema = 'public'
                                    """))
            table_count = result.scalar()

            if table_count == 0:
                st.warning("""
                ⚠️ No tables found in the data database!

                Make sure:
                1. Your database has tables to query
                2. The user has SELECT permissions on those tables
                3. Tables are in the 'public' schema

                Create sample data:
                ```sql
                CREATE TABLE students (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    age INT,
                    grade VARCHAR(10)
                );

                INSERT INTO students (name, age, grade) VALUES
                ('Alice', 20, 'A'),
                ('Bob', 21, 'B');

                GRANT SELECT ON students TO readonly_user;
                ```
                """)
            else:
                logger.info(f"✅ Found {table_count} tables in data database")
                st.success(f"✅ Connected to data database with {table_count} tables")

    except OperationalError as e:
        logger.error(f"Error initializing application database: {e}")
        st.error(f"""
        ❌ Error initializing application database

        **Error:** {str(e)}

        This usually means:
        1. User lacks CREATE TABLE permissions
        2. Database doesn't exist
        3. Schema permissions issue

        **Fix:**
        ```sql
        GRANT ALL PRIVILEGES ON DATABASE text_to_sql_db TO app_user;
        GRANT ALL PRIVILEGES ON SCHEMA public TO app_user;
        ```
        """)
        st.stop()

    except Exception as e:
        logger.error(f"Unexpected error during database initialization: {e}")
        st.error(f"""
        ❌ Unexpected error during database initialization

        **Error:** {str(e)}

        Check server logs for details.
        """)
        st.stop()


def clear_cache():
    """Clear all database connection caches. Useful for reconnecting."""
    st.cache_resource.clear()
    logger.info("Cache cleared")