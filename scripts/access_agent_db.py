import os
import sqlite3
from pathlib import Path


def find_project_root(marker_file=".env"):
    """Find the project root by looking for a marker file (e.g., .env)"""
    current_path = Path(__file__).resolve().parent
    while current_path != current_path.parent:  # Stop at the filesystem root
        if (current_path / marker_file).is_file():
            return current_path
        current_path = current_path.parent
    # If .env is not found by going up from script's dir, try current working directory
    # This helps if the script is run from the project root directly
    if (Path.cwd() / marker_file).is_file():
        return Path.cwd()
    return None


def load_env_vars(env_file_path):
    """Load environment variables from a .env file."""
    variables = {}
    if not env_file_path or not env_file_path.is_file():
        print(f"Warning: .env file not found at {env_file_path}")
        return variables

    with open(env_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                # Remove potential surrounding quotes and export prefix
                value = value.strip().strip("'\"")
                key = key.replace("export ", "")  # Handle if "export" is used
                variables[key.strip()] = value
    return variables


def main():
    project_root = find_project_root()
    if not project_root:
        print(
            "Error: Could not determine project root. Make sure a .env file exists in your project root."
        )
        return

    env_file = project_root / ".env"

    env_vars = load_env_vars(env_file)

    store_path_str = env_vars.get("STORE_PATH")
    db_filename = "memeooorr.db"  # Expected database filename

    if not store_path_str:
        print(f"Error: STORE_PATH not found or empty in {env_file}")
        return

    # Ensure store_path_str is an absolute path or resolve it relative to project root
    store_path = Path(store_path_str)
    if not store_path.is_absolute():
        store_path = (project_root / store_path).resolve()
        print(f"Interpreted STORE_PATH as relative: {store_path}")

    db_full_path = store_path / db_filename

    print(f"Attempting to access database at: {db_full_path}")

    if not db_full_path.is_file():
        print(f"Error: Database file not found at {db_full_path}")
        print(
            f"Please ensure the agent service has run and created the database, or check STORE_PATH in {env_file}."
        )
        return

    try:
        conn = sqlite3.connect(db_full_path)
        cursor = conn.cursor()
        print(f"Successfully connected to {db_full_path}")

        # Example: List tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("\nTables in the database:")
        if tables:
            for table in tables:
                print(f"- {table[0]}")
        else:
            print("No tables found.")

        print("\nEnter SQL queries to execute, or type 'exit' or '.quit' to close.")
        while True:
            query = input("sqlite> ")
            if query.lower() in ["exit", ".quit"]:
                break
            if not query.strip():
                continue
            try:
                cursor.execute(query)
                if query.strip().upper().startswith("SELECT"):
                    results = cursor.fetchall()
                    if results:
                        for row in results:
                            print(row)
                    else:
                        print("Query returned no results.")
                else:
                    conn.commit()  # Commit changes for non-SELECT queries (INSERT, UPDATE, DELETE)
                    print("Query executed.")
                    if cursor.rowcount > 0:
                        print(f"{cursor.rowcount} rows affected.")
            except sqlite3.Error as e:
                print(f"SQLite error: {e}")

    except sqlite3.Error as e:
        print(f"Error connecting to or interacting with the database: {e}")
    finally:
        if "conn" in locals() and conn:
            conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    main()
