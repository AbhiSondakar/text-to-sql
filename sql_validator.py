import sqlparse
from typing import Tuple, Optional
import re
import logging

logger = logging.getLogger(__name__)

# A set of keywords that are explicitly forbidden
FORBIDDEN_KEYWORDS = {
    'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'TRUNCATE', 'CREATE',
    'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK', 'SET', 'EXECUTE', 'CALL', 'ATTACH',
    'DETACH', 'IMPORT', 'REINDEX', 'RELEASE', 'SAVEPOINT', 'VACUUM', 'REPLACE',
    'MERGE', 'COPY', 'LOCK', 'UNLOCK'
}

# Maximum allowed query complexity
MAX_JOINS = 5
MAX_SUBQUERIES = 3
DEFAULT_LIMIT = 1000


def count_joins(sql: str) -> int:
    """Count the number of JOIN operations in the query."""
    sql_upper = sql.upper()
    join_types = ['JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'OUTER JOIN', 'CROSS JOIN']
    return sum(sql_upper.count(join_type) for join_type in join_types)


def count_subqueries(sql: str) -> int:
    """Count the number of subqueries (nested SELECT statements)."""
    # Remove string literals to avoid false positives
    sql_no_strings = re.sub(r"'[^']*'", '', sql)
    sql_no_strings = re.sub(r'"[^"]*"', '', sql_no_strings)
    return sql_no_strings.upper().count('SELECT') - 1  # -1 for the main query


def has_limit_clause(sql: str) -> bool:
    """Check if the query has a LIMIT clause."""
    return bool(re.search(r'\bLIMIT\s+\d+', sql, re.IGNORECASE))


def add_limit_if_missing(sql: str) -> str:
    """Add LIMIT clause if not present."""
    if not has_limit_clause(sql):
        # Remove trailing semicolon if present
        sql = sql.rstrip().rstrip(';')
        return f"{sql} LIMIT {DEFAULT_LIMIT}"
    return sql


def validate_sql_query(sql_query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Validates a SQL query to ensure it is safe to execute.

    Checks performed:
    1. Strips comments
    2. Ensures single SELECT-only statement
    3. Blocks forbidden keywords
    4. Limits query complexity (JOINs, subqueries)
    5. Enforces LIMIT clause

    Returns (validated_sql, None) on success, or (None, error_message) on failure.
    """
    try:
        # 1. Strip comments to prevent injection attacks
        stripped_sql = sqlparse.format(sql_query, strip_comments=True).strip()
        if not stripped_sql:
            return None, "Validation Error: Query is empty after stripping comments."

        # 2. Parse the SQL and check for multiple statements
        parsed = sqlparse.parse(stripped_sql)
        if not parsed:
            return None, "Validation Error: Invalid SQL syntax."

        if len(parsed) > 1:
            return None, "Validation Error: Multiple SQL statements are not allowed."

        statement = parsed[0]

        # 3. Check if it's a SELECT statement
        stmt_type = statement.get_type()
        if stmt_type != 'SELECT':
            return None, f"Validation Error: Only SELECT statements are allowed. Found: {stmt_type}"

        # 4. Check for forbidden keywords (defense-in-depth)
        for token in statement.flatten():
            if token.is_keyword and token.value.upper() in FORBIDDEN_KEYWORDS:
                return None, f"Validation Error: Forbidden keyword '{token.value.upper()}' found."

        # 5. Check query complexity - JOINs
        num_joins = count_joins(stripped_sql)
        if num_joins > MAX_JOINS:
            return None, f"Validation Error: Too many JOINs ({num_joins}). Maximum allowed: {MAX_JOINS}"

        # 6. Check query complexity - Subqueries
        num_subqueries = count_subqueries(stripped_sql)
        if num_subqueries > MAX_SUBQUERIES:
            return None, f"Validation Error: Too many subqueries ({num_subqueries}). Maximum allowed: {MAX_SUBQUERIES}"

        # 7. Add LIMIT if not present
        validated_sql = add_limit_if_missing(stripped_sql)

        # 8. Final sanity check - no semicolons in the middle
        if ';' in validated_sql.rstrip(';'):
            return None, "Validation Error: Multiple statements detected (semicolon found)."

        return validated_sql, None

    except Exception as e:
        return None, f"An unexpected validation error occurred: {str(e)}"


def get_validation_stats(sql_query: str) -> dict:
    """
    Get statistics about a query for informational purposes.
    Returns a dict with query complexity metrics.
    """
    try:
        stripped_sql = sqlparse.format(sql_query, strip_comments=True).strip()
        return {
            'num_joins': count_joins(stripped_sql),
            'num_subqueries': count_subqueries(stripped_sql),
            'has_limit': has_limit_clause(stripped_sql),
            'length': len(stripped_sql)
        }
    # --- FIX: (Issue 5) Handle exceptions and return a consistent dict ---
    except Exception as e:
        logger.error(f"Error getting validation stats: {e}")
        return {
            'num_joins': 0,
            'num_subqueries': 0,
            'has_limit': False,
            'length': 0,
            'error': str(e)
        }
    # --- END FIX ---