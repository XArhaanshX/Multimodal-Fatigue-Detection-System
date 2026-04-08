import csv
import os
from datetime import datetime

CSV_FILE = "sessions.csv"
HEADERS = [
    "driver_name", "driver_phone", "emergency_contact_name",
    "emergency_contact_phone", "start_time", "end_time",
    "max_fatigue_score", "critical_event_triggered"
]

def initialize_csv():
    """Ensure sessions.csv exists with correct headers."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(HEADERS)

def append_session(data: dict) -> int:
    """
    Appends a new session start entry.
    Returns the 0-based index of the data row.
    """
    initialize_csv()
    
    row = [
        data.get("driver_name", ""),
        data.get("driver_phone", ""),
        data.get("emergency_contact_name", ""),
        data.get("emergency_contact_phone", ""),
        data.get("start_time", ""),
        "", # end_time placeholder
        0.0, # max_fatigue_score placeholder
        False # critical_event_triggered placeholder
    ]
    
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    
    # Count rows to find index of the just-added row
    with open(CSV_FILE, mode='r') as file:
        count = sum(1 for _ in file)
        return count - 2  # Subtract 1 for header and 1 for 0-index

def update_session(row_index: int, end_data: dict):
    """
    Updates a specific session row with end-of-session metrics.
    """
    if not os.path.exists(CSV_FILE):
        return

    with open(CSV_FILE, mode='r', newline='') as file:
        rows = list(csv.reader(file))
    
    # row_index + 1 to account for header
    if 0 <= (row_index + 1) < len(rows):
        row = rows[row_index + 1]
        row[5] = end_data.get("end_time", "")
        row[6] = end_data.get("max_fatigue_score", 0.0)
        row[7] = end_data.get("critical_event_triggered", False)
        
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
