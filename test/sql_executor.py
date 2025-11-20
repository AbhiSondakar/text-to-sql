import pandas as pd
from sqlalchemy import text, Engine
from sqlalchemy.exc import ProgrammingError, OperationalError
from typing import List, Dict, Any, Tuple, Optional

def execute_sql_query(sql_query: str, engine: Engine) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Executes a validated SELECT query against the read-only data database.
    Catches SQL-level errors (syntax, missing columns) and returns them as strings.

    Returns (data, None) on success, or (None, error_message) on failure.
    """
    try:
        with engine.connect() as connection:
            # Execute the raw, validated SQL text
            result = connection.execute(text(sql_query))

            # Convert results to a list of dicts for JSON serialization and DataFrame
            #.mappings() provides a dict-like interface
            data = [dict(row) for row in result.mappings().all()]
            return data, None

    except (ProgrammingError, OperationalError) as e:
        # Catch SQL syntax errors, table/column not found, etc.
        # This is expected if the AI makes a mistake.
        # Return the database's original error message
        error_msg = f"SQL Execution Error: {str(e.orig)}"
        return None, error_msg
    except Exception as e:
        return None, f"An unexpected execution error occurred: {str(e)}"