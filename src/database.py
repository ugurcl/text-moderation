import os
import sqlite3
from src.config import DB_PATH


class PredictionDB:
    def __init__(self, db_path=None):
        path = str(db_path or DB_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                label TEXT,
                confidence REAL,
                allowed INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self.conn.commit()

    def save(self, text, label, confidence, allowed):
        self.conn.execute(
            "INSERT INTO predictions (text, label, confidence, allowed) VALUES (?, ?, ?, ?)",
            (text[:500], label, confidence, int(allowed)),
        )
        self.conn.commit()

    def get_stats(self):
        cur = self.conn.execute("SELECT COUNT(*) FROM predictions")
        total = cur.fetchone()[0]

        cur = self.conn.execute(
            "SELECT label, COUNT(*) FROM predictions GROUP BY label ORDER BY COUNT(*) DESC"
        )
        by_label = {row[0]: row[1] for row in cur.fetchall()}

        cur = self.conn.execute("SELECT COUNT(*) FROM predictions WHERE allowed = 0")
        blocked = cur.fetchone()[0]

        return {
            "total": total,
            "blocked": blocked,
            "allowed": total - blocked,
            "by_label": by_label,
        }

    def get_recent(self, limit=20):
        cur = self.conn.execute(
            "SELECT text, label, confidence, allowed, created_at "
            "FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [
            {
                "text": row[0],
                "label": row[1],
                "confidence": row[2],
                "allowed": bool(row[3]),
                "created_at": row[4],
            }
            for row in cur.fetchall()
        ]
