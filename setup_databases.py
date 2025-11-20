"""
Database Setup and Diagnostic Script
Run this to set up your databases correctly and diagnose connection issues.
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import getpass
import sys


def get_postgres_connection():
    """Get connection to PostgreSQL as superuser."""
    print("\n🔐 PostgreSQL Superuser Credentials")
    print("=" * 60)

    host = input("Host [localhost]: ").strip() or "localhost"
    port = input("Port [5432]: ").strip() or "5432"
    user = input("Superuser username [postgres]: ").strip() or "postgres"
    password = getpass.getpass("Superuser password: ")

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        print("✅ Connected to PostgreSQL\n")
        return conn, host, port
    except psycopg2.OperationalError as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)


def create_databases(cursor):
    """Create necessary databases."""
    print("📦 Creating Databases")
    print("=" * 60)

    databases = [
        ("text_to_sql_db", "Application database (chat history)"),
        ("Student", "Data database (your data)")
    ]

    for db_name, description in databases:
        try:
            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
            if cursor.fetchone():
                print(f"ℹ️  {db_name} already exists - {description}")
            else:
                cursor.execute(f"CREATE DATABASE {db_name}")
                print(f"✅ Created {db_name} - {description}")
        except Exception as e:
            print(f"❌ Error with {db_name}: {e}")


def create_users(cursor):
    """Create database users with appropriate permissions."""
    print("\n👤 Creating Database Users")
    print("=" * 60)

    # App user (read/write)
    print("\n1. Application User (needs WRITE access)")
    app_user = input("   Username [app_user]: ").strip() or "app_user"
    app_pass = getpass.getpass("   Password: ") or "secure_app_pass_123!"

    try:
        cursor.execute(f"SELECT 1 FROM pg_user WHERE usename = '{app_user}'")
        if cursor.fetchone():
            print(f"   ℹ️  User {app_user} already exists")
            cursor.execute(f"ALTER USER {app_user} WITH PASSWORD '{app_pass}'")
            print(f"   ✅ Updated password for {app_user}")
        else:
            cursor.execute(f"CREATE USER {app_user} WITH PASSWORD '{app_pass}'")
            print(f"   ✅ Created user {app_user}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Read-only user
    print("\n2. Read-Only User (for data security)")
    readonly_user = input("   Username [readonly_user]: ").strip() or "readonly_user"
    readonly_pass = getpass.getpass("   Password: ") or "secure_readonly_pass_123!"

    try:
        cursor.execute(f"SELECT 1 FROM pg_user WHERE usename = '{readonly_user}'")
        if cursor.fetchone():
            print(f"   ℹ️  User {readonly_user} already exists")
            cursor.execute(f"ALTER USER {readonly_user} WITH PASSWORD '{readonly_pass}'")
            print(f"   ✅ Updated password for {readonly_user}")
        else:
            cursor.execute(f"CREATE USER {readonly_user} WITH PASSWORD '{readonly_pass}'")
            print(f"   ✅ Created user {readonly_user}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    return app_user, app_pass, readonly_user, readonly_pass


def configure_permissions(host, port, app_user, readonly_user):
    """Configure database permissions."""
    print("\n🔐 Configuring Permissions")
    print("=" * 60)

    superuser = input("Superuser username [postgres]: ").strip() or "postgres"
    superpass = getpass.getpass("Superuser password: ")

    # Configure app database (full access)
    print("\n1. Configuring text_to_sql_db (full access for app_user)...")
    try:
        conn = psycopg2.connect(
            host=host, port=port, user=superuser, password=superpass,
            database="text_to_sql_db"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE text_to_sql_db TO {app_user}")
        cursor.execute(f"GRANT ALL PRIVILEGES ON SCHEMA public TO {app_user}")
        cursor.execute(f"GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {app_user}")
        cursor.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON TABLES TO {app_user}")

        print(f"   ✅ Granted full permissions to {app_user}")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Configure data database (read-only)
    print("\n2. Configuring Student (read-only for readonly_user)...")
    try:
        conn = psycopg2.connect(
            host=host, port=port, user=superuser, password=superpass,
            database="Student"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        cursor.execute(f"GRANT CONNECT ON DATABASE Student TO {readonly_user}")
        cursor.execute(f"GRANT USAGE ON SCHEMA public TO {readonly_user}")
        cursor.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {readonly_user}")
        cursor.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {readonly_user}")

        # Revoke write permissions if any
        cursor.execute(f"""
            REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER 
            ON ALL TABLES IN SCHEMA public FROM {readonly_user}
        """)

        print(f"   ✅ Granted read-only permissions to {readonly_user}")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"   ❌ Error: {e}")


def create_sample_data(host, port):
    """Create sample data in Student database."""
    print("\n📊 Sample Data")
    print("=" * 60)

    create_sample = input("Create sample student data? (y/n) [n]: ").strip().lower()
    if create_sample != 'y':
        return

    superuser = input("Superuser username [postgres]: ").strip() or "postgres"
    superpass = getpass.getpass("Superuser password: ")

    try:
        conn = psycopg2.connect(
            host=host, port=port, user=superuser, password=superpass,
            database="Student"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute("""
                       SELECT EXISTS (SELECT
                                      FROM information_schema.tables
                                      WHERE table_schema = 'public'
                                        AND table_name = 'students')
                       """)

        if cursor.fetchone()[0]:
            print("   ℹ️  students table already exists")
        else:
            cursor.execute("""
                           CREATE TABLE students
                           (
                               id            SERIAL PRIMARY KEY,
                               name          VARCHAR(100) NOT NULL,
                               age           INT,
                               grade         VARCHAR(10),
                               department    VARCHAR(50),
                               gpa           DECIMAL(3, 2),
                               enrolled_date DATE DEFAULT CURRENT_DATE
                           )
                           """)

            cursor.execute("""
                           INSERT INTO students (name, age, grade, department, gpa)
                           VALUES ('Alice Johnson', 20, 'A', 'Computer Science', 3.85),
                                  ('Bob Smith', 21, 'B', 'Engineering', 3.45),
                                  ('Charlie Brown', 19, 'A', 'Computer Science', 3.92),
                                  ('Diana Prince', 22, 'B+', 'Mathematics', 3.67),
                                  ('Eve Davis', 20, 'A-', 'Physics', 3.78),
                                  ('Frank Miller', 21, 'C+', 'Engineering', 3.12),
                                  ('Grace Lee', 19, 'A', 'Computer Science', 3.95),
                                  ('Henry Wilson', 23, 'B', 'Mathematics', 3.56),
                                  ('Iris Chen', 20, 'A-', 'Physics', 3.73),
                                  ('Jack Thomas', 22, 'B-', 'Engineering', 3.34)
                           """)

            print("   ✅ Created students table with sample data")
            print("   📊 10 sample records inserted")

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"   ❌ Error: {e}")


def generate_env_file(host, port, app_user, app_pass, readonly_user, readonly_pass):
    """Generate .env file with correct configuration."""
    print("\n📝 Generating .env File")
    print("=" * 60)

    env_content = f"""# =============================================================================
# AI PROVIDER CONFIGURATION
# =============================================================================
# Configure ONE of the following providers

# OpenRouter (Recommended)
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here
OPENROUTER_API_URL=https://openrouter.ai/api/v1
OPENROUTER_REFERER=https://github.com/yourusername/text-to-sql
OPENROUTER_TITLE=Text-to-SQL Agent

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Application Database (WRITE access) - Stores chat history
APP_DB_URL=postgresql://{app_user}:{app_pass}@{host}:{port}/text_to_sql_db

# Data Database (READ-ONLY access) - Your data to query
DATA_DB_URL=postgresql://{readonly_user}:{readonly_pass}@{host}:{port}/Student
"""

    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file")
        print("\n⚠️  IMPORTANT: Update OPENROUTER_API_KEY with your actual API key!")
        print("   Get one at: https://openrouter.ai/keys")
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        print("\nManually create .env with:")
        print(env_content)


def test_connections(host, port, app_user, app_pass, readonly_user, readonly_pass):
    """Test all database connections."""
    print("\n🧪 Testing Connections")
    print("=" * 60)

    # Test app database
    print("\n1. Testing Application Database (write access)...")
    try:
        conn = psycopg2.connect(
            host=host, port=port, user=app_user, password=app_pass,
            database="text_to_sql_db"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.execute("CREATE TEMP TABLE _test (id INT)")
        cursor.execute("DROP TABLE _test")
        print("   ✅ Connection successful (write access confirmed)")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")

    # Test data database
    print("\n2. Testing Data Database (read-only access)...")
    try:
        conn = psycopg2.connect(
            host=host, port=port, user=readonly_user, password=readonly_pass,
            database="Student"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT 1")

        # Test read access
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
        table_count = cursor.fetchone()[0]
        print(f"   ✅ Connection successful ({table_count} tables found)")

        # Verify read-only
        try:
            cursor.execute("CREATE TEMP TABLE _test (id INT)")
            print("   ⚠️  WARNING: User has write access (should be read-only!)")
        except:
            print("   ✅ Read-only access confirmed")

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")


def main():
    """Main setup flow."""
    print("""
╔════════════════════════════════════════════════════════════╗
║         Text-to-SQL Database Setup Script                  ║
║                                                            ║
║  This script will:                                         ║
║  1. Create databases (text_to_sql_db, Student)            ║
║  2. Create users with correct permissions                 ║
║  3. Configure security settings                           ║
║  4. Generate .env file                                    ║
║  5. Test all connections                                  ║
╚════════════════════════════════════════════════════════════╝
""")

    try:
        # Connect to PostgreSQL
        conn, host, port = get_postgres_connection()
        cursor = conn.cursor()

        # Create databases
        create_databases(cursor)

        # Create users
        app_user, app_pass, readonly_user, readonly_pass = create_users(cursor)

        cursor.close()
        conn.close()

        # Configure permissions
        configure_permissions(host, port, app_user, readonly_user)

        # Create sample data
        create_sample_data(host, port)

        # Generate .env
        generate_env_file(host, port, app_user, app_pass, readonly_user, readonly_pass)

        # Test connections
        test_connections(host, port, app_user, app_pass, readonly_user, readonly_pass)

        print("\n" + "=" * 60)
        print("🎉 SETUP COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Update your API key in .env file")
        print("2. Run: streamlit run app.py")
        print("3. Click 'Clear Cache' in sidebar if you see connection errors")

    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)


# if __name__ == "__main__":