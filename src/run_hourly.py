"""One-shot scheduled job for hosted platforms such as Render Cron Jobs."""

from db_schema import initialize_database_schema
from drift import monitor_accuracy_drift
from poll import generate_live_inference


def main():
    initialize_database_schema()
    generate_live_inference()
    monitor_accuracy_drift(window_hours=100)


if __name__ == "__main__":
    main()
