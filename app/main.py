"""
Employee Attendance & Analytics API

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (a local .env is loaded for convenience)
"""
import math
import os
import re
import calendar
from datetime import datetime, timedelta, timezone, date as date_type
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, field_validator
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = client[os.getenv("MONGO_DB", "attendance_db")]

IST = timezone(timedelta(hours=5, minutes=30))

app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0")

# ── Constants ──────────────────────────────────────────────────────────────── #
PRESENCE_STATUSES = {"PRESENT", "WFH", "ON_DUTY"}
ALL_STATUSES = {"PRESENT", "WFH", "ON_DUTY", "ABSENT", "LEAVE"}
EPOCH_MIN = 100_000_000_000
EPOCH_MAX = 4_102_444_800_000


# ── Startup: create indexes ────────────────────────────────────────────────── #
_indexes_created = False


def _ensure_indexes():
    """Create all needed indexes idempotently. Safe to call multiple times."""
    global _indexes_created
    if _indexes_created:
        return
    try:
        # Employees: unique on emp_code (enforces concurrency on create)
        db.employees.create_index("emp_code", unique=True, name="ux_emp_code")

        # Attendance: unique compound on (emp_code, date) – enforces one record per day
        db.attendance_logs.create_index(
            [("emp_code", ASCENDING), ("date", ASCENDING)],
            unique=True,
            name="ux_empcode_date",
        )

        # Attendance: for list sorted by date desc, emp_code asc + range filters
        db.attendance_logs.create_index(
            [("date", DESCENDING), ("emp_code", ASCENDING)],
            name="ix_date_desc_emp_asc",
        )

        # Attendance: for analytics that filter on date range
        db.attendance_logs.create_index(
            [("date", ASCENDING)],
            name="ix_date_asc",
        )

        # Employees: for department queries
        db.employees.create_index("department", name="ix_department")

        _indexes_created = True
    except Exception:
        pass  # Will retry on next call


@app.on_event("startup")
def on_startup():
    """Try to create indexes at startup; non-fatal if MongoDB is not ready yet."""
    _ensure_indexes()


# ── Helpers ────────────────────────────────────────────────────────────────── #
def _round_half_up(value: float, decimals: int = 2) -> float:
    """Round using ROUND_HALF_UP (not banker's rounding)."""
    d = Decimal(str(value))
    quant = Decimal(10) ** -decimals
    return float(d.quantize(quant, rounding=ROUND_HALF_UP))


def _truncate_to_seconds(dt: datetime) -> datetime:
    """Truncate a datetime to whole seconds (drop microseconds)."""
    return dt.replace(microsecond=0)


def _epoch_ms_to_dt(ms: int) -> datetime:
    """Convert epoch milliseconds to a UTC-aware datetime, truncated to seconds."""
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return _truncate_to_seconds(dt)


def _dt_to_epoch_ms(dt: datetime) -> int | None:
    """Convert a datetime to epoch milliseconds. Returns None for None."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _validate_epoch_ms(value, field_name: str = "punched_at") -> int:
    """Validate an epoch-ms value from a request."""
    if value is None:
        return None
    if isinstance(value, float):
        raise HTTPException(422, detail=f"{field_name} must be an integer, not a float")
    if not isinstance(value, int):
        raise HTTPException(422, detail=f"{field_name} must be an integer")
    if value < EPOCH_MIN or value > EPOCH_MAX:
        raise HTTPException(422, detail=f"{field_name} out of range")
    return value


def _get_ist_date(dt_utc: datetime) -> str:
    """Get the IST calendar date string YYYY-MM-DD from a UTC datetime."""
    ist_dt = dt_utc.astimezone(IST)
    return ist_dt.strftime("%Y-%m-%d")


def _compute_attendance_date(punch_in_utc: datetime, shift_start: str, shift_end: str) -> str:
    """
    Compute the attendance date per R1.
    For overnight shifts (shift_end <= shift_start): if the IST time of punch_in
    is earlier than shift_end, the record belongs to the previous calendar day.
    """
    ist_dt = punch_in_utc.astimezone(IST)
    ist_date_str = ist_dt.strftime("%Y-%m-%d")

    # Check if overnight shift
    if shift_end <= shift_start:
        punch_time_str = ist_dt.strftime("%H:%M")
        # If punch-in IST time is earlier than shift_end, it belongs to previous day's shift
        if punch_time_str < shift_end:
            prev_day = ist_dt.date() - timedelta(days=1)
            return prev_day.isoformat()

    return ist_date_str


def _compute_late_minutes(punch_in_utc: datetime, shift_start: str) -> int:
    """
    R2: Grace period 10 min. Late only if strictly more than 10:00 after shift_start.
    late_minutes = floor(minutes since shift_start), not since grace end.
    """
    ist_dt = punch_in_utc.astimezone(IST)
    h, m = map(int, shift_start.split(":"))
    start_ist = ist_dt.replace(hour=h, minute=m, second=0, microsecond=0)

    # Handle overnight: if punch_in is before the shift_start on the clock,
    # it might have wrapped. But for late calculation, we compare same-day times.
    diff_seconds = (ist_dt - start_ist).total_seconds()

    # If diff_seconds is negative (punch before shift), not late
    if diff_seconds <= 0:
        return 0

    # Grace: strictly more than 10 minutes (600 seconds)
    if diff_seconds <= 600:
        return 0

    return int(diff_seconds // 60)  # floor of minutes since shift_start


def _compute_overtime(punch_out_utc: datetime, shift_end: str, att_date_str: str, shift_start: str) -> int:
    """
    R3: overtime_minutes = floor(minutes between shift_end and punch_out), only if >= 30.
    For overnight shifts, shift_end falls on the next calendar day.
    """
    h, m = map(int, shift_end.split(":"))
    att_date = date_type.fromisoformat(att_date_str)

    # For overnight shifts, shift_end is on the next day
    if shift_end <= shift_start:
        end_date = att_date + timedelta(days=1)
    else:
        end_date = att_date

    end_dt = datetime(end_date.year, end_date.month, end_date.day, h, m, 0, tzinfo=IST)
    end_dt_utc = end_dt.astimezone(timezone.utc)

    diff_seconds = (punch_out_utc - end_dt_utc).total_seconds()
    if diff_seconds < 0:
        return 0

    minutes = int(diff_seconds // 60)
    return minutes if minutes >= 30 else 0


def _compute_work_hours(punch_in_utc: datetime, punch_out_utc: datetime) -> float:
    """R4: (punch_out - punch_in) in seconds / 3600, rounded 2 decimals half-up."""
    seconds = (punch_out_utc - punch_in_utc).total_seconds()
    return _round_half_up(seconds / 3600, 2)


def _compute_half_day(work_hours: float) -> bool:
    """R5: half_day = true when rounded work_hours < 4.50."""
    return work_hours < 4.50


def _serialize_attendance(doc: dict) -> dict:
    """Convert a MongoDB attendance doc to the API response format."""
    result = {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": _dt_to_epoch_ms(doc.get("punch_in")),
        "punch_out": _dt_to_epoch_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes", 0),
        "overtime_minutes": doc.get("overtime_minutes", 0),
        "half_day": doc.get("half_day", False),
        "history": _serialize_history(doc.get("history", [])),
    }
    return result


def _serialize_history(history: list) -> list:
    """Convert history entries: BSON dates -> epoch ms."""
    result = []
    for entry in history:
        h = {
            "at": _dt_to_epoch_ms(entry["at"]),
            "by": entry["by"],
            "reason": entry["reason"],
            "changes": {},
        }
        for field, change in entry.get("changes", {}).items():
            if field in ("punch_in", "punch_out"):
                h["changes"][field] = {
                    "from": _dt_to_epoch_ms(change.get("from")),
                    "to": _dt_to_epoch_ms(change.get("to")),
                }
            else:
                h["changes"][field] = {
                    "from": change.get("from"),
                    "to": change.get("to"),
                }
        result.append(h)
    return result


def _serialize_employee(doc: dict) -> dict:
    """Convert a MongoDB employee doc to the API response format."""
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc.get("shift_start", "09:30"),
        "shift_end": doc.get("shift_end", "18:30"),
        "joined_on": doc["joined_on"],
        "created_at": _dt_to_epoch_ms(doc["created_at"]),
    }


def _validate_page_params(page: int, page_size: int):
    """Validate pagination parameters per R10."""
    if page < 1:
        raise HTTPException(422, detail="page must be >= 1")
    if page_size < 1 or page_size > 100:
        raise HTTPException(422, detail="page_size must be between 1 and 100")


def _validate_month(month: str) -> tuple[str, str]:
    """Validate YYYY-MM and return (first_day, last_day) as YYYY-MM-DD strings."""
    if not re.match(r"^\d{4}-(0[1-9]|1[0-2])$", month):
        raise HTTPException(422, detail="month must be YYYY-MM")
    year, mon = map(int, month.split("-"))
    _, last = calendar.monthrange(year, mon)
    return f"{month}-01", f"{month}-{last:02d}"


def _validate_date_str(d: str) -> str:
    """Validate a YYYY-MM-DD date string."""
    try:
        date_type.fromisoformat(d)
    except (ValueError, TypeError):
        raise HTTPException(422, detail=f"Invalid date: {d}")
    return d


def _count_working_days(start_str: str, end_str: str) -> int:
    """Count Mon-Fri days between start and end inclusive."""
    start = date_type.fromisoformat(start_str)
    end = date_type.fromisoformat(end_str)
    count = 0
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon=0 .. Fri=4
            count += 1
        current += timedelta(days=1)
    return count


# ── Models ─────────────────────────────────────────────────────────────────── #
class EmployeeIn(BaseModel):
    emp_code: str
    name: str
    email: str
    department: str
    shift_start: str = "09:30"
    shift_end: str = "18:30"
    joined_on: str

    @field_validator("emp_code")
    @classmethod
    def validate_emp_code(cls, v):
        if not re.match(r"^EMP\d{4,6}$", v):
            raise ValueError("emp_code must match EMP followed by 4-6 digits")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v or len(v) > 100:
            raise ValueError("name must be 1-100 characters")
        return v

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v) or len(v) > 120:
            raise ValueError("invalid email")
        return v

    @field_validator("department")
    @classmethod
    def validate_department(cls, v):
        if not v or len(v) > 50:
            raise ValueError("department must be 1-50 characters")
        return v

    @field_validator("shift_start", "shift_end")
    @classmethod
    def validate_shift_time(cls, v):
        if not re.match(r"^([01]\d|2[0-3]):[0-5]\d$", v):
            raise ValueError("shift time must be HH:MM (24-hour)")
        return v

    @field_validator("joined_on")
    @classmethod
    def validate_joined_on(cls, v):
        try:
            date_type.fromisoformat(v)
        except ValueError:
            raise ValueError("joined_on must be YYYY-MM-DD")
        return v


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[int] = None
    status: str = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[int] = None


class RegularizeIn(BaseModel):
    reason: str
    regularized_by: str
    status: Optional[str] = None
    punch_in: Optional[int] = None
    punch_out: Optional[int] = None

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, v):
        if len(v) < 5 or len(v) > 200:
            raise ValueError("reason must be 5-200 characters")
        return v

    @field_validator("regularized_by")
    @classmethod
    def validate_regularized_by(cls, v):
        if not v or len(v) > 50:
            raise ValueError("regularized_by must be 1-50 characters")
        return v


# ── Endpoints ──────────────────────────────────────────────────────────────── #

# ── Health ──
@app.get("/health")
def health():
    try:
        db.command("ping")
        _ensure_indexes()
        return {"status": "ok"}
    except Exception:
        raise HTTPException(503, detail="MongoDB unavailable")


# ── Employees ──

@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    if body.shift_start == body.shift_end:
        raise HTTPException(422, detail="shift_start must differ from shift_end")

    doc = body.model_dump()
    doc["created_at"] = datetime.now(timezone.utc)

    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, detail="emp_code already exists")

    doc.pop("_id", None)
    return _serialize_employee(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    _validate_page_params(page, page_size)
    q = {}
    if department:
        q["department"] = department

    total = db.employees.count_documents(q)
    skip = (page - 1) * page_size
    items = list(
        db.employees.find(q, {"_id": 0})
        .sort("emp_code", ASCENDING)
        .skip(skip)
        .limit(page_size)
    )
    return {
        "items": [_serialize_employee(e) for e in items],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ── Attendance: Punch-in ──

@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    # Validate status
    if body.status not in PRESENCE_STATUSES:
        raise HTTPException(422, detail=f"status must be one of {sorted(PRESENCE_STATUSES)}")

    # Validate punched_at
    if body.punched_at is not None:
        _validate_epoch_ms(body.punched_at, "punched_at")

    # Look up employee
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, detail="Employee not found")

    # Determine timestamp
    if body.punched_at is not None:
        ts = _epoch_ms_to_dt(body.punched_at)
    else:
        ts = _truncate_to_seconds(datetime.now(timezone.utc))

    # Compute attendance date per R1
    att_date = _compute_attendance_date(ts, emp["shift_start"], emp["shift_end"])

    # Compute late minutes per R2
    late_minutes = _compute_late_minutes(ts, emp["shift_start"])

    doc = {
        "emp_code": body.emp_code,
        "date": att_date,
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": late_minutes,
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }

    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, detail="Already punched in for this date")

    doc.pop("_id", None)
    return _serialize_attendance(doc)


# ── Attendance: Punch-out ──

@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    # Validate punched_at
    if body.punched_at is not None:
        _validate_epoch_ms(body.punched_at, "punched_at")

    # Look up employee
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, detail="Employee not found")

    # Determine timestamp
    if body.punched_at is not None:
        ts = _epoch_ms_to_dt(body.punched_at)
    else:
        ts = _truncate_to_seconds(datetime.now(timezone.utc))

    # Find the most recent record with punch_in <= punched_at and no punch_out
    record = db.attendance_logs.find_one(
        {
            "emp_code": body.emp_code,
            "punch_in": {"$lte": ts},
            "punch_out": None,
        },
        sort=[("punch_in", DESCENDING)],
    )

    if not record:
        # Check if there's a record already punched out (409) vs no record at all (404)
        any_record = db.attendance_logs.find_one(
            {
                "emp_code": body.emp_code,
                "punch_in": {"$lte": ts},
                "punch_out": {"$ne": None},
            },
            sort=[("punch_in", DESCENDING)],
        )
        if any_record:
            raise HTTPException(409, detail="Already punched out")
        raise HTTPException(404, detail="No open punch-in found")

    # Validate: punch_out must be after punch_in
    if ts <= record["punch_in"]:
        raise HTTPException(422, detail="punched_at must be after punch_in")

    # Validate: punch_out must be within 24 hours of punch_in
    if (ts - record["punch_in"]).total_seconds() > 86400:
        raise HTTPException(422, detail="punched_at must be within 24 hours of punch_in")

    # Compute derived fields
    work_hours = _compute_work_hours(record["punch_in"], ts)
    overtime = _compute_overtime(ts, emp["shift_end"], record["date"], emp["shift_start"])
    half_day = _compute_half_day(work_hours)

    # Atomic update: use punch_out: None as condition to prevent double punch-out
    result = db.attendance_logs.update_one(
        {"_id": record["_id"], "punch_out": None},
        {
            "$set": {
                "punch_out": ts,
                "work_hours": work_hours,
                "overtime_minutes": overtime,
                "half_day": half_day,
            }
        },
    )

    if result.modified_count == 0:
        raise HTTPException(409, detail="Already punched out (concurrent request)")

    # Fetch updated record
    updated = db.attendance_logs.find_one({"_id": record["_id"]})
    return _serialize_attendance(updated)


# ── Attendance: List ──

@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    _validate_page_params(page, page_size)

    if date_from:
        _validate_date_str(date_from)
    if date_to:
        _validate_date_str(date_to)
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, detail="date_from must be <= date_to")
    if status and status not in ALL_STATUSES:
        raise HTTPException(422, detail=f"Invalid status: {status}")

    q = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status

    total = db.attendance_logs.count_documents(q)
    skip = (page - 1) * page_size
    docs = list(
        db.attendance_logs.find(q, {"_id": 0})
        .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
        .skip(skip)
        .limit(page_size)
    )
    return {
        "items": [_serialize_attendance(d) for d in docs],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ── Attendance: Regularize (PATCH) ──

@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(emp_code: str, date: str, body: RegularizeIn):
    _validate_date_str(date)

    # Validate employee exists
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(404, detail="Employee not found")

    # Find the record
    record = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not record:
        raise HTTPException(404, detail="No attendance record for this date")

    new_status = body.status if body.status is not None else record["status"]

    # Validate status value
    if body.status is not None and body.status not in ALL_STATUSES:
        raise HTTPException(422, detail=f"Invalid status: {body.status}")

    # Validate epoch ms values
    new_punch_in_ms = None
    new_punch_out_ms = None
    if body.punch_in is not None:
        _validate_epoch_ms(body.punch_in, "punch_in")
        new_punch_in_ms = body.punch_in
    if body.punch_out is not None:
        _validate_epoch_ms(body.punch_out, "punch_out")
        new_punch_out_ms = body.punch_out

    # Rule: ABSENT/LEAVE clears punch times, so supplying punch_in/punch_out with it is 422
    if new_status in ("ABSENT", "LEAVE"):
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(422, detail="Cannot supply punch times with ABSENT/LEAVE status")

    # Build the new values
    if new_status in ("ABSENT", "LEAVE"):
        final_punch_in = None
        final_punch_out = None
    else:
        # Presence status requires punch_in (supplied or already stored)
        if body.punch_in is not None:
            final_punch_in = _epoch_ms_to_dt(body.punch_in)
        else:
            final_punch_in = record.get("punch_in")

        if final_punch_in is None:
            raise HTTPException(422, detail="Presence status requires a punch_in")

        if body.punch_out is not None:
            final_punch_out = _epoch_ms_to_dt(body.punch_out)
        else:
            final_punch_out = record.get("punch_out")

        # Validate: punch_in must stay on the record's attendance date (R1)
        computed_date = _compute_attendance_date(final_punch_in, emp["shift_start"], emp["shift_end"])
        if computed_date != date:
            raise HTTPException(422, detail="punch_in must stay on the record's attendance date")

        # Validate punch_out constraints
        if final_punch_out is not None:
            if final_punch_out <= final_punch_in:
                raise HTTPException(422, detail="punch_out must be after punch_in")
            if (final_punch_out - final_punch_in).total_seconds() > 86400:
                raise HTTPException(422, detail="punch_out must be within 24 hours of punch_in")

    # Compute derived fields
    if new_status in ("ABSENT", "LEAVE"):
        new_late_minutes = 0
        new_overtime_minutes = 0
        new_work_hours = None
        new_half_day = False
    elif final_punch_out is not None:
        new_late_minutes = _compute_late_minutes(final_punch_in, emp["shift_start"])
        new_work_hours = _compute_work_hours(final_punch_in, final_punch_out)
        new_overtime_minutes = _compute_overtime(final_punch_out, emp["shift_end"], date, emp["shift_start"])
        new_half_day = _compute_half_day(new_work_hours)
    else:
        new_late_minutes = _compute_late_minutes(final_punch_in, emp["shift_start"])
        new_work_hours = None
        new_overtime_minutes = 0
        new_half_day = False

    # Build changes dict: only fields that actually changed
    old_vals = {
        "status": record["status"],
        "punch_in": record.get("punch_in"),
        "punch_out": record.get("punch_out"),
        "work_hours": record.get("work_hours"),
        "late_minutes": record.get("late_minutes", 0),
        "overtime_minutes": record.get("overtime_minutes", 0),
        "half_day": record.get("half_day", False),
    }
    new_vals = {
        "status": new_status,
        "punch_in": final_punch_in,
        "punch_out": final_punch_out,
        "work_hours": new_work_hours,
        "late_minutes": new_late_minutes,
        "overtime_minutes": new_overtime_minutes,
        "half_day": new_half_day,
    }

    changes = {}
    for field in ("status", "punch_in", "punch_out", "work_hours", "late_minutes", "overtime_minutes", "half_day"):
        old_v = old_vals[field]
        new_v = new_vals[field]
        if field in ("punch_in", "punch_out"):
            # Compare as epoch ms to handle naive vs aware datetime differences
            old_ms = _dt_to_epoch_ms(old_v) if old_v else None
            new_ms = _dt_to_epoch_ms(new_v) if new_v else None
            if old_ms != new_ms:
                changes[field] = {"from": old_v, "to": new_v}
        elif field == "work_hours":
            # Compare with tolerance for float
            if old_v is None and new_v is None:
                pass
            elif old_v is None or new_v is None:
                changes[field] = {"from": old_v, "to": new_v}
            elif abs(old_v - new_v) > 0.001:
                changes[field] = {"from": old_v, "to": new_v}
        else:
            if old_v != new_v:
                changes[field] = {"from": old_v, "to": new_v}

    if not changes:
        raise HTTPException(422, detail="No changes to apply")

    # Create history entry
    now_utc = datetime.now(timezone.utc)
    history_entry = {
        "at": now_utc,
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }

    # Atomic update: use the current history length to detect concurrent modifications
    current_history = record.get("history", [])
    update_filter = {
        "_id": record["_id"],
        f"history.{len(current_history)}": {"$exists": False},
    }

    update_set = {
        "status": new_status,
        "punch_in": final_punch_in,
        "punch_out": final_punch_out,
        "work_hours": new_work_hours,
        "late_minutes": new_late_minutes,
        "overtime_minutes": new_overtime_minutes,
        "half_day": new_half_day,
    }

    result = db.attendance_logs.update_one(
        update_filter,
        {
            "$set": update_set,
            "$push": {"history": history_entry},
        },
    )

    if result.modified_count == 0:
        raise HTTPException(409, detail="Concurrent modification detected")

    updated = db.attendance_logs.find_one({"_id": record["_id"]})
    return _serialize_attendance(updated)


# ── Analytics: Employee Monthly ──

@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str = Query(...)):
    first_day, last_day = _validate_month(month)

    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(404, detail="Employee not found")

    # Working days: Mon-Fri in the month, starting from joined_on if later
    joined = emp["joined_on"]
    period_start = max(first_day, joined)
    if period_start > last_day:
        working_days = 0
    else:
        working_days = _count_working_days(period_start, last_day)

    # Aggregate attendance for the month
    pipeline = [
        {"$match": {"emp_code": emp_code, "date": {"$gte": first_day, "$lte": last_day}}},
        {
            "$group": {
                "_id": None,
                "present_full": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {
                                        "$let": {
                                            "vars": {
                                                "d": {"$dateFromString": {"dateString": "$date", "format": "%Y-%m-%d"}},
                                            },
                                            "in": {
                                                "$and": [
                                                    {"$gte": [{"$dayOfWeek": "$$d"}, 2]},
                                                    {"$lte": [{"$dayOfWeek": "$$d"}, 6]},
                                                ]
                                            },
                                        }
                                    },
                                    {"$eq": [{"$ifNull": ["$half_day", False]}, False]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "present_half": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {
                                        "$let": {
                                            "vars": {
                                                "d": {"$dateFromString": {"dateString": "$date", "format": "%Y-%m-%d"}},
                                            },
                                            "in": {
                                                "$and": [
                                                    {"$gte": [{"$dayOfWeek": "$$d"}, 2]},
                                                    {"$lte": [{"$dayOfWeek": "$$d"}, 6]},
                                                ]
                                            },
                                        }
                                    },
                                    {"$eq": [{"$ifNull": ["$half_day", False]}, True]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "leave_days": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "late_count": {
                    "$sum": {
                        "$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]
                    }
                },
                "total_late_minutes": {
                    "$sum": {"$ifNull": ["$late_minutes", 0]}
                },
                "total_overtime_minutes": {
                    "$sum": {"$ifNull": ["$overtime_minutes", 0]}
                },
            }
        },
    ]

    results = list(db.attendance_logs.aggregate(pipeline))

    if results:
        r = results[0]
        present_days = r["present_full"] + r["present_half"] * 0.5
        leave_days = r["leave_days"]
        late_count = r["late_count"]
        total_late_minutes = r["total_late_minutes"]
        total_overtime_minutes = r["total_overtime_minutes"]
    else:
        present_days = 0
        leave_days = 0
        late_count = 0
        total_late_minutes = 0
        total_overtime_minutes = 0

    if working_days > 0:
        attendance_pct = _round_half_up(present_days / working_days * 100, 2)
    else:
        attendance_pct = None

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": leave_days,
        "late_count": late_count,
        "total_late_minutes": total_late_minutes,
        "total_overtime_minutes": total_overtime_minutes,
        "attendance_pct": attendance_pct,
    }


# ── Analytics: Department Summary ──

@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(...),
    department: Optional[str] = None,
):
    first_day, last_day = _validate_month(month)

    # Build employee filter
    emp_filter = {"joined_on": {"$lte": last_day}}
    if department:
        emp_filter["department"] = department

    # Get all departments with headcount (employees joined on or before month end)
    emp_pipeline = [
        {"$match": emp_filter},
        {
            "$group": {
                "_id": "$department",
                "headcount": {"$sum": 1},
                "emp_codes": {"$push": "$emp_code"},
            }
        },
        {"$sort": {"_id": 1}},
    ]
    dept_data = {d["_id"]: d for d in db.employees.aggregate(emp_pipeline)}

    if not dept_data:
        return {"month": month, "items": []}

    # Get attendance stats per department
    all_emp_codes = []
    for d in dept_data.values():
        all_emp_codes.extend(d["emp_codes"])

    att_pipeline = [
        {
            "$match": {
                "emp_code": {"$in": all_emp_codes},
                "date": {"$gte": first_day, "$lte": last_day},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "as": "emp_info",
            }
        },
        {"$unwind": "$emp_info"},
        {
            "$group": {
                "_id": "$emp_info.department",
                "present_full": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {"$eq": [{"$ifNull": ["$half_day", False]}, False]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "present_half": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {"$eq": [{"$ifNull": ["$half_day", False]}, True]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "work_hours_sum": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {"$ne": [{"$ifNull": ["$work_hours", None]}, None]},
                                ]
                            },
                            "$work_hours",
                            0,
                        ]
                    }
                },
                "work_hours_count": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {"$ne": [{"$ifNull": ["$work_hours", None]}, None]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {
                        "$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]
                    }
                },
                "total_late_minutes": {
                    "$sum": {"$ifNull": ["$late_minutes", 0]}
                },
                "leave_count": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "on_duty_count": {
                    "$sum": {"$cond": [{"$eq": ["$status", "ON_DUTY"]}, 1, 0]}
                },
            }
        },
    ]
    att_data = {d["_id"]: d for d in db.attendance_logs.aggregate(att_pipeline)}

    items = []
    for dept_name in sorted(dept_data.keys()):
        dept = dept_data[dept_name]
        if dept["headcount"] == 0:
            continue

        att = att_data.get(dept_name, {})
        present_full = att.get("present_full", 0)
        present_half = att.get("present_half", 0)
        present_days = present_full + present_half * 0.5

        wh_count = att.get("work_hours_count", 0)
        if wh_count > 0:
            avg_wh = _round_half_up(att.get("work_hours_sum", 0) / wh_count, 2)
        else:
            avg_wh = None

        items.append({
            "department": dept_name,
            "headcount": dept["headcount"],
            "present_days": present_days,
            "avg_work_hours": avg_wh,
            "late_count": att.get("late_count", 0),
            "total_late_minutes": att.get("total_late_minutes", 0),
            "leave_count": att.get("leave_count", 0),
            "on_duty_count": att.get("on_duty_count", 0),
        })

    return {"month": month, "items": items}


# ── Analytics: Late Leaderboard ──

@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(...),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    first_day, last_day = _validate_month(month)

    # Build match stage
    match_stage = {"date": {"$gte": first_day, "$lte": last_day}}

    pipeline = [
        {"$match": match_stage},
        # Only records with late_minutes > 0
        {"$match": {"$expr": {"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}}},
        # Lookup employee
        {
            "$lookup": {
                "from": "employees",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "as": "emp_info",
            }
        },
        {"$unwind": "$emp_info"},
    ]

    # Filter by department if specified
    if department:
        pipeline.append({"$match": {"emp_info.department": department}})

    pipeline.extend([
        # Group by employee
        {
            "$group": {
                "_id": "$emp_code",
                "name": {"$first": "$emp_info.name"},
                "department": {"$first": "$emp_info.department"},
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1},
            }
        },
        # Only employees with total > 0
        {"$match": {"total_late_minutes": {"$gt": 0}}},
        # Sort by total_late_minutes desc, emp_code asc
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        # Use $setWindowFields for competition ranking
        {
            "$setWindowFields": {
                "sortBy": {"total_late_minutes": -1},
                "output": {
                    "rank": {
                        "$rank": {},
                    }
                },
            }
        },
        # Filter by rank <= limit
        {"$match": {"rank": {"$lte": limit}}},
        # Final sort
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
    ])

    results = list(db.attendance_logs.aggregate(pipeline))

    items = []
    for r in results:
        items.append({
            "rank": r["rank"],
            "emp_code": r["_id"],
            "name": r["name"],
            "department": r["department"],
            "total_late_minutes": r["total_late_minutes"],
            "late_count": r["late_count"],
        })

    return {"month": month, "items": items}


# ── Analytics: Department Trend ──

@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    from_date: str = Query(..., alias="from"),
    to_date: str = Query(..., alias="to"),
):
    _validate_date_str(from_date)
    _validate_date_str(to_date)

    if to_date < from_date:
        raise HTTPException(422, detail="'to' must be >= 'from'")

    # Check range doesn't exceed 92 days
    d_from = date_type.fromisoformat(from_date)
    d_to = date_type.fromisoformat(to_date)
    if (d_to - d_from).days > 91:
        raise HTTPException(422, detail="Range must not exceed 92 days")

    # Check department exists
    dept_exists = db.employees.find_one({"department": department})
    if not dept_exists:
        raise HTTPException(404, detail="Unknown department")

    # Get all employees in the department with their joined_on dates
    dept_employees = list(db.employees.find(
        {"department": department},
        {"emp_code": 1, "joined_on": 1, "_id": 0},
    ))

    # Get attendance records for these employees in the date range
    emp_codes = [e["emp_code"] for e in dept_employees]
    att_records = list(db.attendance_logs.find(
        {
            "emp_code": {"$in": emp_codes},
            "date": {"$gte": from_date, "$lte": to_date},
        },
        {"_id": 0},
    ))

    # Build a lookup: date -> list of records
    records_by_date = {}
    for r in att_records:
        records_by_date.setdefault(r["date"], []).append(r)

    # Generate one row per calendar day
    items = []
    rates_window = []  # Store (date, rate_or_none) for moving average

    current = d_from
    while current <= d_to:
        date_str = current.isoformat()
        is_working_day = current.weekday() < 5

        # Headcount: employees with joined_on <= this day (R9)
        headcount = sum(1 for e in dept_employees if e["joined_on"] <= date_str)

        # Present count from records
        day_records = records_by_date.get(date_str, [])
        present_count = 0.0
        late_count = 0
        for r in day_records:
            if r["status"] in PRESENCE_STATUSES:
                if r.get("half_day", False):
                    present_count += 0.5
                else:
                    present_count += 1.0
            if (r.get("late_minutes", 0) or 0) > 0:
                late_count += 1

        # Attendance rate: only on working days with headcount > 0
        if is_working_day and headcount > 0:
            attendance_rate = _round_half_up(present_count / headcount, 4)
        else:
            attendance_rate = None

        rates_window.append(attendance_rate)

        # Moving average: last 7 entries (current + up to 6 preceding) within the requested range
        window = rates_window[-7:]
        non_null = [r for r in window if r is not None]
        if non_null:
            moving_avg_7d = _round_half_up(sum(non_null) / len(non_null), 4)
        else:
            moving_avg_7d = None

        # present_count: format as int if whole number, else float
        if present_count == int(present_count):
            present_count_out = int(present_count)
        else:
            present_count_out = present_count

        items.append({
            "date": date_str,
            "is_working_day": is_working_day,
            "headcount": headcount,
            "present_count": present_count_out,
            "late_count": late_count,
            "attendance_rate": attendance_rate,
            "moving_avg_7d": moving_avg_7d,
        })

        current += timedelta(days=1)

    return {"department": department, "items": items}


# ── Admin: Explain ──

@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str,
    emp_code: Optional[str] = None,
    month: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    valid_endpoints = {
        "attendance_list",
        "employee_monthly",
        "department_summary",
        "late_leaderboard",
        "department_trend",
    }
    if endpoint not in valid_endpoints:
        raise HTTPException(422, detail=f"Unknown endpoint: {endpoint}")

    if endpoint == "attendance_list":
        q = {}
        if emp_code:
            q["emp_code"] = emp_code
        if date_from or date_to:
            q["date"] = {}
            if date_from:
                q["date"]["$gte"] = date_from
            if date_to:
                q["date"]["$lte"] = date_to
        if status:
            q["status"] = status

        explain = db.attendance_logs.find(q).sort(
            [("date", DESCENDING), ("emp_code", ASCENDING)]
        ).skip((page - 1) * page_size).limit(page_size).explain()

        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain}

    elif endpoint == "employee_monthly":
        if not emp_code or not month:
            raise HTTPException(422, detail="emp_code and month are required")
        first_day, last_day = _validate_month(month)
        pipeline = [
            {"$match": {"emp_code": emp_code, "date": {"$gte": first_day, "$lte": last_day}}},
            {
                "$group": {
                    "_id": None,
                    "count": {"$sum": 1},
                }
            },
        ]
        explain = db.command(
            "aggregate",
            "attendance_logs",
            pipeline=pipeline,
            explain=True,
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain}

    elif endpoint == "department_summary":
        if not month:
            raise HTTPException(422, detail="month is required")
        first_day, last_day = _validate_month(month)
        match_stage = {"date": {"$gte": first_day, "$lte": last_day}}
        pipeline = [
            {"$match": match_stage},
            {
                "$lookup": {
                    "from": "employees",
                    "localField": "emp_code",
                    "foreignField": "emp_code",
                    "as": "emp_info",
                }
            },
            {"$unwind": "$emp_info"},
        ]
        if department:
            pipeline.append({"$match": {"emp_info.department": department}})
        pipeline.append({
            "$group": {
                "_id": "$emp_info.department",
                "count": {"$sum": 1},
            }
        })
        explain = db.command(
            "aggregate",
            "attendance_logs",
            pipeline=pipeline,
            explain=True,
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain}

    elif endpoint == "late_leaderboard":
        if not month:
            raise HTTPException(422, detail="month is required")
        first_day, last_day = _validate_month(month)
        match_stage = {"date": {"$gte": first_day, "$lte": last_day}}
        pipeline = [
            {"$match": match_stage},
            {"$match": {"$expr": {"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}}},
            {
                "$lookup": {
                    "from": "employees",
                    "localField": "emp_code",
                    "foreignField": "emp_code",
                    "as": "emp_info",
                }
            },
            {"$unwind": "$emp_info"},
        ]
        if department:
            pipeline.append({"$match": {"emp_info.department": department}})
        pipeline.append({
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1},
            }
        })
        explain = db.command(
            "aggregate",
            "attendance_logs",
            pipeline=pipeline,
            explain=True,
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain}

    elif endpoint == "department_trend":
        if not department or not from_date or not to_date:
            raise HTTPException(422, detail="department, from, and to are required")
        emp_codes = [
            e["emp_code"]
            for e in db.employees.find({"department": department}, {"emp_code": 1, "_id": 0})
        ]
        pipeline = [
            {
                "$match": {
                    "emp_code": {"$in": emp_codes},
                    "date": {"$gte": from_date, "$lte": to_date},
                }
            },
            {"$group": {"_id": "$date", "count": {"$sum": 1}}},
        ]
        explain = db.command(
            "aggregate",
            "attendance_logs",
            pipeline=pipeline,
            explain=True,
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain}
