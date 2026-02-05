# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import logging
import sqlite3


class HotelBooker:
  """
  Core business logic for hotel booking, independent of any web framework.
  """

  def __init__(self, db_name="data.db"):
    self.db_name = db_name
    self._initialize_db()

  def _get_db_connection(self):
    """Helper to get a new, independent database connection."""
    conn = sqlite3.connect(self.db_name)
    conn.row_factory = sqlite3.Row
    return conn

  def _initialize_db(self):
    """
    Drops, creates, and populates the database tables with sample data.
    """
    conn = None
    try:
      conn = self._get_db_connection()
      cursor = conn.cursor()

      cursor.execute("DROP TABLE IF EXISTS bookings")
      cursor.execute("DROP TABLE IF EXISTS hotels")
      conn.commit()

      cursor.execute("""
                CREATE TABLE IF NOT EXISTS hotels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    location TEXT NOT NULL,
                    total_rooms INTEGER NOT NULL,
                    available_rooms INTEGER NOT NULL,
                    price_per_night REAL NOT NULL
                )
            """)
      cursor.execute("""
                CREATE TABLE IF NOT EXISTS bookings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    custom_booking_id TEXT UNIQUE,
                    hotel_id INTEGER NOT NULL,
                    guest_name TEXT NOT NULL,
                    check_in_date TEXT NOT NULL,
                    check_out_date TEXT NOT NULL,
                    num_rooms INTEGER NOT NULL,
                    total_price REAL NOT NULL,
                    FOREIGN KEY (hotel_id) REFERENCES hotels(id)
                )
            """)

      conn.commit()

      sample_hotels = [
          ("Grand Hyatt", "New York", 200, 150, 250.00),
          ("The Plaza Hotel", "New York", 150, 100, 350.00),
          ("Hilton Chicago", "Chicago", 300, 250, 180.00),
          ("Marriott Marquis", "San Francisco", 250, 200, 220.00),
      ]
      cursor.executemany(
          """
                INSERT INTO hotels (name, location, total_rooms, available_rooms, price_per_night)
                VALUES (?, ?, ?, ?, ?)
            """,
          sample_hotels,
      )
      conn.commit()

      initial_bookings_data = [
          (1, "Alice Smith", "2025-08-10", "2025-08-15", 1, 1250.00),
          (3, "Bob Johnson", "2025-09-01", "2025-09-03", 2, 720.00),
      ]
      for booking_data in initial_bookings_data:
        cursor.execute(
            """
                    INSERT INTO bookings (hotel_id, guest_name, check_in_date, check_out_date, num_rooms, total_price)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
            booking_data,
        )
        booking_id_int = cursor.lastrowid
        custom_id = f"HB-{booking_id_int}"
        cursor.execute(
            "UPDATE bookings SET custom_booking_id = ? WHERE id = ?",
            (custom_id, booking_id_int),
        )
      conn.commit()
    except sqlite3.Error as e:
      if conn:
        conn.rollback()
    finally:
      if conn:
        conn.close()

  def is_token_valid(self, conn, token):
    """Checks if a given token is valid and not expired."""
    logging.info("not implemented")
    return True

  def get_available_hotels(self, cursor, location=None):
    """Retrieves a list of available hotels, optionally filtered by location."""
    query = (
        "SELECT id, name, location, available_rooms, price_per_night FROM"
        " hotels WHERE available_rooms > 0"
    )
    params = []
    if location:
      query += " AND location LIKE ?"
      params.append(f"%{location}%")
    try:
      cursor.execute(query, params)
      rows = cursor.fetchall()
      return [dict(row) for row in rows], None
    except sqlite3.Error as e:
      return None, f"Error getting available hotels: {e}"

  def book_a_room(
      self, conn, hotel_id, guest_name, check_in_date, check_out_date, num_rooms
  ):
    """Books a room in a specified hotel."""
    cursor = conn.cursor()
    try:
      cursor.execute(
          "SELECT available_rooms, price_per_night FROM hotels WHERE id = ?",
          (hotel_id,),
      )
      hotel_info = cursor.fetchone()

      if not hotel_info:
        return None, f"Hotel with ID {hotel_id} not found."

      available_rooms, price_per_night = (
          hotel_info["available_rooms"],
          hotel_info["price_per_night"],
      )
      if available_rooms < num_rooms:
        return (
            None,
            (
                f"Not enough rooms available at hotel ID {hotel_id}. Available:"
                f" {available_rooms}, Requested: {num_rooms}"
            ),
        )

      try:
        check_in_dt = datetime.datetime.strptime(check_in_date, "%Y-%m-%d")
        check_out_dt = datetime.datetime.strptime(check_out_date, "%Y-%m-%d")
      except ValueError:
        return None, "Invalid date format. Please use YYYY-MM-DD."

      num_nights = (check_out_dt - check_in_dt).days
      if num_nights <= 0:
        return None, "Check-out date must be after check-in date."

      total_price = num_rooms * price_per_night * num_nights

      cursor.execute(
          "UPDATE hotels SET available_rooms = ? WHERE id = ?",
          (available_rooms - num_rooms, hotel_id),
      )

      cursor.execute(
          """
                INSERT INTO bookings (hotel_id, guest_name, check_in_date, check_out_date, num_rooms, total_price)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
          (
              hotel_id,
              guest_name,
              check_in_date,
              check_out_date,
              num_rooms,
              total_price,
          ),
      )

      booking_id_int = cursor.lastrowid
      custom_booking_id = f"HB-{booking_id_int}"

      cursor.execute(
          "UPDATE bookings SET custom_booking_id = ? WHERE id = ?",
          (custom_booking_id, booking_id_int),
      )

      conn.commit()
      return custom_booking_id, None
    except sqlite3.Error as e:
      conn.rollback()
      return None, f"Error booking room: {e}"

  def get_booking_details(self, cursor, booking_id=None, guest_name=None):
    """Retrieves details for a specific booking."""
    query = """
            SELECT
                b.custom_booking_id,
                h.name AS hotel_name,
                h.location AS hotel_location,
                b.guest_name,
                b.check_in_date,
                b.check_out_date,
                b.num_rooms,
                b.total_price
            FROM
                bookings b
            JOIN
                hotels h ON b.hotel_id = h.id
        """
    params = []
    result_type = "single"

    if booking_id:
      query += " WHERE b.custom_booking_id = ?"
      params.append(booking_id)
    elif guest_name:
      query += " WHERE LOWER(b.guest_name) LIKE LOWER(?)"
      params.append(f"%{guest_name}%")
      result_type = "list"
    else:
      return (
          None,
          (
              "Please provide either a booking ID or a guest name to retrieve"
              " booking details."
          ),
      )

    try:
      cursor.execute(query, params)
      rows = cursor.fetchall()

      if not rows:
        return (
            None,
            (
                f"No booking found for the given criteria (ID: {booking_id},"
                f" Name: {guest_name})."
            ),
        )

      bookings = [dict(row) for row in rows]
      return bookings if result_type == "list" else bookings[0], None
    except sqlite3.Error as e:
      return None, f"Error getting booking details: {e}"
