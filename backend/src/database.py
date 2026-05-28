"""
MongoDB Database Service for Elderly Care AI System

Provides:
- Real-time storage of activities, emotions, and alerts
- Concurrent data persistence for dashboard
- Historical data retrieval
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading
import queue
import time
import os
from enum import Enum

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from pymongo import MongoClient, DESCENDING, ASCENDING
    from pymongo.collection import Collection
    from pymongo.database import Database
    from bson import ObjectId
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print(" pymongo not installed. MongoDB features disabled.")


class CollectionNames(Enum):
    """MongoDB collection names."""
    ACTIVITIES = "activities"
    EMOTIONS = "emotions"
    ALERTS = "alerts"
    PERSONS = "persons"
    SESSIONS = "sessions"
    MOVEMENTS = "movements"
    VIDEOS = "videos"


@dataclass
class ActivityRecord:
    """Activity record for MongoDB storage."""
    person_id: int
    activity_type: str
    confidence: float
    movement_score: float
    is_moving: bool
    duration_seconds: float
    timestamp: datetime
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "person_id": self.person_id,
            "activity_type": self.activity_type,
            "confidence": self.confidence,
            "movement_score": self.movement_score,
            "is_moving": self.is_moving,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "created_at": datetime.utcnow()
        }


@dataclass
class EmotionRecord:
    """Emotion record for MongoDB storage."""
    person_id: int
    emotion_type: str
    confidence: float
    mood_score: float
    is_distressed: bool
    all_scores: Dict[str, float]
    timestamp: datetime
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "person_id": self.person_id,
            "emotion_type": self.emotion_type,
            "confidence": self.confidence,
            "mood_score": self.mood_score,
            "is_distressed": self.is_distressed,
            "all_scores": self.all_scores,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "created_at": datetime.utcnow()
        }


@dataclass
class AlertRecord:
    """Alert record for MongoDB storage."""
    person_id: int
    alert_type: str
    severity: str
    message: str
    data: Dict
    timestamp: datetime
    status: str = "active"  # active, acknowledged, resolved
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "person_id": self.person_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
            "status": self.status,
            "session_id": self.session_id,
            "created_at": datetime.utcnow()
        }


@dataclass
class PatientRecord:
    """Patient/Elderly person record for MongoDB storage."""
    person_id: int
    name: str
    age: int
    gender: str
    medical_history: List[str]
    emergency_contact: str
    room_number: str
    status: str = "active"  # active, inactive
    
    def to_dict(self) -> Dict:
        return {
            "person_id": self.person_id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "medical_history": self.medical_history,
            "emergency_contact": self.emergency_contact,
            "room_number": self.room_number,
            "status": self.status,
            "created_at": datetime.utcnow()
        }


@dataclass
class VideoRecord:
    """Video upload record for testing and analysis."""
    filename: str
    original_filename: str
    file_size: int
    duration: Optional[float]
    upload_timestamp: datetime
    status: str = "uploaded"  # uploaded, processing, completed, failed
    analysis_results: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_size": self.file_size,
            "duration": self.duration,
            "upload_timestamp": self.upload_timestamp,
            "status": self.status,
            "analysis_results": self.analysis_results,
            "created_at": datetime.utcnow()
        }


@dataclass
class MovementRecord:
    """Movement/change record for real-time tracking."""
    person_id: int
    position_x: float
    position_y: float
    bbox: List[float]
    keypoints: Dict
    activity: str
    emotion: str
    fall_status: str
    movement_score: float
    timestamp: datetime
    frame_number: int
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "person_id": self.person_id,
            "position": {"x": self.position_x, "y": self.position_y},
            "bbox": self.bbox,
            "keypoints": self.keypoints,
            "activity": self.activity,
            "emotion": self.emotion,
            "fall_status": self.fall_status,
            "movement_score": self.movement_score,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "session_id": self.session_id,
            "created_at": datetime.utcnow()
        }


class DatabaseService:
    """
    MongoDB database service for elderly care monitoring.
    
    Features:
    - Async batch writes for performance
    - Real-time data persistence
    - Historical data retrieval
    - Session management
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize database service.
        
        Args:
            config: Database configuration dictionary.
        """
        self._current_session_id = None
        self.client = None
        self.db = None
        
        if not MONGODB_AVAILABLE:
            self.enabled = False
            print(" MongoDB is not available. Data will not be persisted.")
            return
            
        config = config or {}
        db_config = config.get('database', {})
        
        # Connection settings from environment or config
        self.connection_string = os.environ.get(
            'MONGO_URI',
            db_config.get('connection_string', 'mongodb://localhost:27017/')
        )
        self.database_name = os.environ.get(
            'MONGO_DB_NAME',
            db_config.get('database_name', 'elderly_care_ai')
        )
        
        # Batch write settings
        self.batch_size = db_config.get('batch_size', 50)
        self.flush_interval = db_config.get('flush_interval', 2.0)  # seconds
        
        # Initialize connection
        self._init_connection()
        
        # Write queues for async batch writes
        self._activity_queue = queue.Queue()
        self._emotion_queue = queue.Queue()
        self._alert_queue = queue.Queue()
        self._movement_queue = queue.Queue()
        
        # Background writer thread
        self._running = True
        self._writer_thread = threading.Thread(target=self._background_writer, daemon=True)
        self._writer_thread.start()
        
        # Session tracking
        self._current_session_id = None
        
        self.enabled = True
        print(f" MongoDB connected: {self.connection_string}{self.database_name}")
    
    def _init_connection(self) -> None:
        """Initialize MongoDB connection and create indexes."""
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
            # Test connection
            self.client.admin.command('ping')
            
            self.db: Database = self.client[self.database_name]
            
            # Create collections and indexes
            self._create_indexes()
            
        except Exception as e:
            print(f" MongoDB connection failed: {e}")
            self.enabled = False
            raise
    
    def _create_indexes(self) -> None:
        """Create indexes for efficient queries."""
        # Activities collection
        self.db[CollectionNames.ACTIVITIES.value].create_index([
            ("person_id", ASCENDING),
            ("timestamp", DESCENDING)
        ])
        self.db[CollectionNames.ACTIVITIES.value].create_index([
            ("session_id", ASCENDING)
        ])
        
        # Emotions collection
        self.db[CollectionNames.EMOTIONS.value].create_index([
            ("person_id", ASCENDING),
            ("timestamp", DESCENDING)
        ])
        self.db[CollectionNames.EMOTIONS.value].create_index([
            ("emotion_type", ASCENDING),
            ("timestamp", DESCENDING)
        ])
        
        # Alerts collection
        self.db[CollectionNames.ALERTS.value].create_index([
            ("severity", ASCENDING),
            ("timestamp", DESCENDING)
        ])
        self.db[CollectionNames.ALERTS.value].create_index([
            ("status", ASCENDING)
        ])
        
        # Movements collection (TTL index to auto-delete old records after 24 hours)
        self.db[CollectionNames.MOVEMENTS.value].create_index([
            ("timestamp", DESCENDING)
        ])
        self.db[CollectionNames.MOVEMENTS.value].create_index(
            [("created_at", ASCENDING)],
            expireAfterSeconds=86400  # 24 hours TTL
        )
    
    def start_session(self) -> str:
        """Start a new monitoring session."""
        session_doc = {
            "started_at": datetime.utcnow(),
            "status": "active"
        }
        result = self.db[CollectionNames.SESSIONS.value].insert_one(session_doc)
        self._current_session_id = str(result.inserted_id)
        print(f" Started monitoring session: {self._current_session_id}")
        return self._current_session_id
    
    def end_session(self) -> None:
        """End current monitoring session."""
        if self._current_session_id:
            self.db[CollectionNames.SESSIONS.value].update_one(
                {"_id": ObjectId(self._current_session_id)},
                {
                    "$set": {
                        "ended_at": datetime.utcnow(),
                        "status": "completed"
                    }
                }
            )
            print(f" Ended monitoring session: {self._current_session_id}")
            self._current_session_id = None
    
    def _background_writer(self) -> None:
        """Background thread for batch writing to MongoDB."""
        while self._running:
            try:
                time.sleep(self.flush_interval)
                self._flush_queues()
            except Exception as e:
                print(f" Background writer error: {e}")
    
    def _flush_queues(self) -> None:
        """Flush all write queues to MongoDB."""
        self._flush_queue(self._activity_queue, CollectionNames.ACTIVITIES.value)
        self._flush_queue(self._emotion_queue, CollectionNames.EMOTIONS.value)
        self._flush_queue(self._alert_queue, CollectionNames.ALERTS.value)
        self._flush_queue(self._movement_queue, CollectionNames.MOVEMENTS.value)
    
    def _flush_queue(self, q: queue.Queue, collection_name: str) -> None:
        """Flush a specific queue to MongoDB."""
        documents = []
        try:
            while not q.empty() and len(documents) < self.batch_size * 10:
                documents.append(q.get_nowait())
        except queue.Empty:
            pass
        
        if documents:
            try:
                self.db[collection_name].insert_many(documents, ordered=False)
            except Exception as e:
                print(f" Failed to write to {collection_name}: {e}")
    
    # ==================== Activity Methods ====================
    
    def store_activity(self, record: ActivityRecord) -> None:
        """Store activity record (async via queue)."""
        if not self.enabled:
            return
        record.session_id = self._current_session_id
        self._activity_queue.put(record.to_dict())
    
    def store_activity_immediate(self, record: ActivityRecord) -> str:
        """Store activity record immediately (sync)."""
        if not self.enabled:
            return ""
        record.session_id = self._current_session_id
        result = self.db[CollectionNames.ACTIVITIES.value].insert_one(record.to_dict())
        return str(result.inserted_id)
    
    def get_activities(
        self, 
        person_id: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get activity history."""
        if not self.enabled:
            return []
            
        query = {}
        if person_id is not None:
            query["person_id"] = person_id
        if start_time or end_time:
            query["timestamp"] = {}
            if start_time:
                query["timestamp"]["$gte"] = start_time
            if end_time:
                query["timestamp"]["$lte"] = end_time
        
        cursor = self.db[CollectionNames.ACTIVITIES.value].find(query).sort(
            "timestamp", DESCENDING
        ).limit(limit)
        
        return list(cursor)
    
    # ==================== Emotion Methods ====================
    
    def store_emotion(self, record: EmotionRecord) -> None:
        """Store emotion record (async via queue)."""
        if not self.enabled:
            return
        record.session_id = self._current_session_id
        self._emotion_queue.put(record.to_dict())
    
    def store_emotion_immediate(self, record: EmotionRecord) -> str:
        """Store emotion record immediately (sync)."""
        if not self.enabled:
            return ""
        record.session_id = self._current_session_id
        result = self.db[CollectionNames.EMOTIONS.value].insert_one(record.to_dict())
        return str(result.inserted_id)
    
    def get_emotions(
        self, 
        person_id: Optional[int] = None,
        emotion_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get emotion history."""
        if not self.enabled:
            return []
            
        query = {}
        if person_id is not None:
            query["person_id"] = person_id
        if emotion_type:
            query["emotion_type"] = emotion_type
        if start_time or end_time:
            query["timestamp"] = {}
            if start_time:
                query["timestamp"]["$gte"] = start_time
            if end_time:
                query["timestamp"]["$lte"] = end_time
        
        cursor = self.db[CollectionNames.EMOTIONS.value].find(query).sort(
            "timestamp", DESCENDING
        ).limit(limit)
        
        return list(cursor)
    
    def get_emotion_stats(
        self, 
        person_id: Optional[int] = None,
        hours: int = 24
    ) -> Dict:
        """Get emotion statistics for the last N hours."""
        if not self.enabled:
            return {}
            
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        pipeline = [
            {"$match": {
                "timestamp": {"$gte": start_time},
                **({"person_id": person_id} if person_id else {})
            }},
            {"$group": {
                "_id": "$emotion_type",
                "count": {"$sum": 1},
                "avg_confidence": {"$avg": "$confidence"},
                "avg_mood": {"$avg": "$mood_score"}
            }},
            {"$sort": {"count": -1}}
        ]
        
        result = list(self.db[CollectionNames.EMOTIONS.value].aggregate(pipeline))
        return {r["_id"]: r for r in result}
    
    # ==================== Alert Methods ====================
    
    def store_alert(self, record: AlertRecord) -> str:
        """Store alert record (immediate for critical alerts)."""
        if not self.enabled:
            return ""
        record.session_id = self._current_session_id
        result = self.db[CollectionNames.ALERTS.value].insert_one(record.to_dict())
        return str(result.inserted_id)
    
    def get_alerts(
        self, 
        status: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get alerts."""
        if not self.enabled:
            return []
            
        query = {}
        if status:
            query["status"] = status
        if severity:
            query["severity"] = severity
        
        cursor = self.db[CollectionNames.ALERTS.value].find(query).sort(
            "timestamp", DESCENDING
        ).limit(limit)
        
        return list(cursor)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if not self.enabled:
            return False
        try:
            result = self.db[CollectionNames.ALERTS.value].update_one(
                {"_id": ObjectId(alert_id)},
                {
                    "$set": {
                        "status": "acknowledged",
                        "acknowledged_at": datetime.utcnow()
                    }
                }
            )
            return result.modified_count > 0
        except Exception:
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if not self.enabled:
            return False
        try:
            result = self.db[CollectionNames.ALERTS.value].update_one(
                {"_id": ObjectId(alert_id)},
                {
                    "$set": {
                        "status": "resolved",
                        "resolved_at": datetime.utcnow()
                    }
                }
            )
            return result.modified_count > 0
        except Exception:
            return False
    
    # ==================== Movement Methods ====================
    
    def store_movement(self, record: MovementRecord) -> None:
        """Store movement record (async via queue for high-frequency data)."""
        if not self.enabled:
            return
        record.session_id = self._current_session_id
        self._movement_queue.put(record.to_dict())
    
    def get_movements(
        self,
        person_id: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> List[Dict]:
        """Get movement history."""
        if not self.enabled:
            return []
            
        query = {}
        if person_id is not None:
            query["person_id"] = person_id
        if start_time or end_time:
            query["timestamp"] = {}
            if start_time:
                query["timestamp"]["$gte"] = start_time
            if end_time:
                query["timestamp"]["$lte"] = end_time
        
        cursor = self.db[CollectionNames.MOVEMENTS.value].find(query).sort(
            "timestamp", DESCENDING
        ).limit(limit)
        
        return list(cursor)
    
    # ==================== Patient Methods ====================
    
    def store_patient(self, record: PatientRecord) -> str:
        """Store or update patient record."""
        if not self.enabled:
            return ""
        
        # Use person_id as unique identifier
        result = self.db[CollectionNames.PERSONS.value].update_one(
            {"person_id": record.person_id},
            {"$set": record.to_dict()},
            upsert=True
        )
        return str(result.upserted_id or "")
    
    def get_patients(self) -> List[Dict]:
        """Get all patient info."""
        if not self.enabled:
            return []
        cursor = self.db[CollectionNames.PERSONS.value].find().sort("name", ASCENDING)
        return list(cursor)
    
    def get_patient(self, person_id: int) -> Optional[Dict]:
        """Get patient by ID."""
        if not self.enabled:
            return None
        return self.db[CollectionNames.PERSONS.value].find_one({"person_id": person_id})
    
    # ==================== Video Methods ====================
    
    def store_video(self, record: 'VideoRecord') -> str:
        """Store video upload record."""
        if not self.enabled:
            return ""
        result = self.db[CollectionNames.VIDEOS.value].insert_one(record.to_dict())
        return str(result.inserted_id)
    
    def get_videos(self, limit: int = 50) -> List[Dict]:
        """Get all uploaded videos."""
        if not self.enabled:
            return []
        cursor = self.db[CollectionNames.VIDEOS.value].find().sort("upload_timestamp", DESCENDING).limit(limit)
        return list(cursor)
    
    def get_video(self, video_id: str) -> Optional[Dict]:
        """Get video by ID."""
        if not self.enabled:
            return None
        try:
            from bson import ObjectId
            return self.db[CollectionNames.VIDEOS.value].find_one({"_id": ObjectId(video_id)})
        except:
            return None
    
    def update_video_status(self, video_id: str, status: str, analysis_results: Optional[Dict] = None) -> bool:
        """Update video processing status."""
        if not self.enabled:
            return False
        try:
            from bson import ObjectId
            update_data = {"status": status}
            if analysis_results:
                update_data["analysis_results"] = analysis_results
            result = self.db[CollectionNames.VIDEOS.value].update_one(
                {"_id": ObjectId(video_id)},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except:
            return False

    def delete_video(self, video_id: str) -> bool:
        """Delete a video record from the database."""
        if not self.enabled:
            return False
        try:
            from bson import ObjectId
            result = self.db[CollectionNames.VIDEOS.value].delete_one({"_id": ObjectId(video_id)})
            return result.deleted_count > 0
        except:
            return False

    # ==================== Aggregation Methods ====================
    
    def get_activity_summary(self, hours: int = 24) -> Dict:
        """Get activity summary for the last N hours."""
        if not self.enabled:
            return {}
            
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        pipeline = [
            {"$match": {"timestamp": {"$gte": start_time}}},
            {"$group": {
                "_id": "$activity_type",
                "count": {"$sum": 1},
                "total_duration": {"$sum": "$duration_seconds"},
                "avg_movement_score": {"$avg": "$movement_score"}
            }},
            {"$sort": {"count": -1}}
        ]
        
        result = list(self.db[CollectionNames.ACTIVITIES.value].aggregate(pipeline))
        return {r["_id"]: r for r in result}
    
    def get_daily_report(self, date: Optional[datetime] = None) -> Dict:
        """Generate daily report."""
        if not self.enabled:
            return {}
            
        if date is None:
            date = datetime.utcnow()
        
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        # Activity summary
        activity_summary = self.get_activity_summary(24)
        
        # Emotion summary
        emotion_summary = self.get_emotion_stats(hours=24)
        
        # Alert count
        alerts = self.get_alerts(limit=1000)
        daily_alerts = [a for a in alerts if start_of_day <= a.get("timestamp", datetime.min) < end_of_day]
        
        return {
            "date": date.isoformat(),
            "activities": activity_summary,
            "emotions": emotion_summary,
            "alerts": {
                "total": len(daily_alerts),
                "by_severity": self._count_by_key(daily_alerts, "severity"),
                "by_type": self._count_by_key(daily_alerts, "alert_type")
            }
        }
    
    def _count_by_key(self, items: List[Dict], key: str) -> Dict[str, int]:
        """Count items by a specific key."""
        counts = {}
        for item in items:
            val = item.get(key, "unknown")
            counts[val] = counts.get(val, 0) + 1
        return counts
    
    # ==================== Cleanup ====================
    
    def close(self) -> None:
        """Close database connection."""
        self._running = False
        if hasattr(self, '_writer_thread'):
            self._writer_thread.join(timeout=5)
        
        # Flush remaining data
        if self.enabled:
            self._flush_queues()
            self.client.close()
            print(" MongoDB connection closed")
    
    def clear_all_data(self) -> None:
        """Clear all data (for testing purposes)."""
        if not self.enabled:
            return
        for collection in CollectionNames:
            self.db[collection.value].delete_many({})
        print(" All data cleared from MongoDB")


# Singleton instance
_db_instance: Optional[DatabaseService] = None


def get_database(config: Dict = None) -> DatabaseService:
    """Get or create database service singleton."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseService(config)
    return _db_instance


def close_database() -> None:
    """Close database connection."""
    global _db_instance
    if _db_instance:
        _db_instance.close()
        _db_instance = None

