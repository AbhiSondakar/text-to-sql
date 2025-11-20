import psycopg2
from sqlalchemy import create_engine, text
import sys
import os


def test_psycopg2_connection():
    """Test connection using psycopg2"""
    print("=" * 60)
    print("TESTING DATABASE CONNECTIONS WITH PSYCOPG2")
    print("=" * 60)

    # Test Application DB
    print("\n1. Testing Application DB (text_to_sql_db)...")
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="text_to_sql_db",
            user="readonly_user",
            password="secure_password_123!"
        )
        cursor = conn.cursor()

        # Test basic query
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        print("✅ Basic query test: PASSED")

        # Check if tables exist
        cursor.execute("""
                       SELECT table_name
                       FROM information_schema.tables
                       WHERE table_schema = 'public'
                       """)
        tables = cursor.fetchall()
        print(f"✅ Found {len(tables)} tables in database")

        # Check current user and permissions
        cursor.execute("SELECT current_user, current_database()")
        user_info = cursor.fetchone()
        print(f"✅ Connected as: {user_info[0]} to database: {user_info[1]}")

        conn.close()
        print("🎉 Application DB connection: SUCCESS")

    except psycopg2.OperationalError as e:
        print(f"❌ Connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    # Test Data DB
    print("\n2. Testing Data DB (Student)...")
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="Student",
            user="readonly_user",
            password="secure_password_123!"
        )
        cursor = conn.cursor()

        # Test basic query
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        print("✅ Basic query test: PASSED")

        # Check if tables exist
        cursor.execute("""
                       SELECT table_name
                       FROM information_schema.tables
                       WHERE table_schema = 'public'
                       """)
        tables = cursor.fetchall()
        print(f"✅ Found {len(tables)} tables in database")

        # List table names
        if tables:
            table_names = [table[0] for table in tables]
            print(f"📊 Tables: {', '.join(table_names)}")

        conn.close()
        print("🎉 Data DB connection: SUCCESS")
        return True

    except psycopg2.OperationalError as e:
        print(f"❌ Connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_sqlalchemy_connection():
    """Test connection using SQLAlchemy (what Streamlit uses)"""
    print("\n" + "=" * 60)
    print("TESTING DATABASE CONNECTIONS WITH SQLALCHEMY")
    print("=" * 60)

    # Test Application DB
    print("\n1. Testing Application DB with SQLAlchemy...")
    try:
        app_db_url = "postgresql://readonly_user:secure_password_123!@localhost:5432/text_to_sql_db"
        engine = create_engine(app_db_url)

        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1, version()"))
            row = result.fetchone()
            print(f"✅ Basic query test: PASSED")
            print(f"✅ PostgreSQL version: {row[1]}")

        # Check tables
        with engine.connect() as conn:
            result = conn.execute(text("""
                                       SELECT table_name
                                       FROM information_schema.tables
                                       WHERE table_schema = 'public'
                                       """))
            tables = result.fetchall()
            print(f"✅ Found {len(tables)} tables")

        print("🎉 SQLAlchemy Application DB connection: SUCCESS")

    except Exception as e:
        print(f"❌ SQLAlchemy connection failed: {e}")
        return False

    # Test Data DB
    print("\n2. Testing Data DB with SQLAlchemy...")
    try:
        data_db_url = "postgresql://readonly_user:secure_password_123!@localhost:5432/Student"
        engine = create_engine(data_db_url)

        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Basic query test: PASSED")

        # Check tables and show sample data
        with engine.connect() as conn:
            result = conn.execute(text("""
                                       SELECT table_name
                                       FROM information_schema.tables
                                       WHERE table_schema = 'public'
                                       """))
            tables = result.fetchall()
            print(f"✅ Found {len(tables)} tables")

            # Show sample data from first table if exists
            if tables:
                first_table = tables[0][0]
                print(f"📋 Sampling data from table: {first_table}")

                try:
                    sample_result = conn.execute(text(f"SELECT * FROM {first_table} LIMIT 3"))
                    sample_rows = sample_result.fetchall()
                    print(f"📊 Sample data ({len(sample_rows)} rows):")
                    for row in sample_rows:
                        print(f"   {row}")
                except Exception as sample_error:
                    print(f"   Note: Could not sample data: {sample_error}")

        print("🎉 SQLAlchemy Data DB connection: SUCCESS")
        return True

    except Exception as e:
        print(f"❌ SQLAlchemy connection failed: {e}")
        return False


def test_streamlit_connection_simulation():
    """Simulate how Streamlit would connect"""
    print("\n" + "=" * 60)
    print("SIMULATING STREAMLIT CONNECTION")
    print("=" * 60)

    # This simulates what Streamlit does internally
    try:
        import streamlit as st
        print("✅ Streamlit is available")

        # Test app_db connection
        print("\n1. Testing Streamlit app_db connection...")
        try:
            app_db_url = "postgresql://readonly_user:secure_password_123!@localhost:5432/text_to_sql_db"
            conn = st.connection("app_db", type="sql", url=app_db_url)

            with conn.session as s:
                result = s.execute(text("SELECT 1"))
                print("✅ Streamlit app_db connection: SUCCESS")

        except Exception as e:
            print(f"❌ Streamlit app_db connection failed: {e}")

        # Test data_db connection
        print("\n2. Testing Streamlit data_db connection...")
        try:
            data_db_url = "postgresql://readonly_user:secure_password_123!@localhost:5432/Student"
            conn = st.connection("data_db", type="sql", url=data_db_url)

            with conn.session as s:
                result = s.execute(text("SELECT 1"))
                print("✅ Streamlit data_db connection: SUCCESS")

        except Exception as e:
            print(f"❌ Streamlit data_db connection failed: {e}")

    except ImportError:
        print("ℹ️ Streamlit not available in current environment")


def check_environment_variables():
    """Check if environment variables are set correctly"""
    print("\n" + "=" * 60)
    print("CHECKING ENVIRONMENT VARIABLES")
    print("=" * 60)

    env_vars = {
        'APP_DB_URL': 'postgresql://readonly_user:secure_password_123!@localhost:5432/text_to_sql_db',
        'DATA_DB_URL': 'postgresql://readonly_user:secure_password_123!@localhost:5432/Student'
    }

    for var_name, expected_value in env_vars.items():
        actual_value = os.getenv(var_name)
        if actual_value:
            print(f"✅ {var_name} is set")
            if actual_value == expected_value:
                print(f"   Value matches expected: {actual_value}")
            else:
                print(f"   ⚠️  Value differs:")
                print(f"   Expected: {expected_value}")
                print(f"   Actual: {actual_value}")
        else:
            print(f"❌ {var_name} is not set")


def check_dependencies():
    """Check if required packages are installed"""
    print("\n" + "=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)

    dependencies = [
        'psycopg2',
        'sqlalchemy',
        'streamlit',
        'pandas',
        'python-dotenv'
    ]

    for package in dependencies:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")


def main():
    """Run all tests"""
    print("🔍 DATABASE CONNECTION DIAGNOSTIC TOOL")
    print("This will test your PostgreSQL connections and identify issues.\n")

    # Check dependencies first
    check_dependencies()

    # Test connections
    psycopg2_success = test_psycopg2_connection()
    sqlalchemy_success = test_sqlalchemy_connection()

    # Check environment
    check_environment_variables()

    # Test Streamlit simulation
    test_streamlit_connection_simulation()

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    if psycopg2_success and sqlalchemy_success:
        print("🎉 ALL CONNECTION TESTS PASSED!")
        print("\nNext steps:")
        print("1. Ensure your .streamlit/secrets.toml has the correct format:")
        print("""
[connections.app_db]
url = "postgresql://readonly_user:secure_password_123!@localhost:5432/text_to_sql_db"

[connections.data_db]
url = "postgresql://readonly_user:secure_password_123!@localhost:5432/Student"
""")
        print("2. Restart your Streamlit app")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nCommon issues and solutions:")
        print("1. Database doesn't exist - Run: CREATE DATABASE text_to_sql_db;")
        print("2. User doesn't exist - Run: CREATE USER readonly_user WITH PASSWORD 'secure_password_123!';")
        print("3. Permission issues - Grant CONNECT and SELECT permissions")
        print("4. Wrong password - Double-check the password")
        print("5. PostgreSQL not running - Start PostgreSQL service")


if __name__ == "__main__":
    main()