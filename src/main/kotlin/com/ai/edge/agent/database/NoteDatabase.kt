/**
 * NoteDatabase - SQLite-based note persistence for the agent loop.
 * 
 * Replaces file-based append_note with robust SQLite storage.
 * Stores notes with titles, content, and timestamps.
 * Data survives app restarts.
 */
package com.ai.edge.agent.database

import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteOpenHelper
import android.util.Log
import java.io.File
import java.util.*

/**
 * Note schema stored in SQLite.
 */
data class Note(
    val id: String = UUID.randomUUID().toString(),
    val title: String = "",
    val content: String = "",
    val created: Long = System.currentTimeMillis(),
    val modified: Long = System.currentTimeMillis()
): Serializable {

    /** Format note for display */
    fun getDisplayString(): String {
        val dateFormatter = java.text.SimpleDateFormat(
            "MMM d, h:mm a",
            java.util.Locale.getDefault()
        )
        return "$dateFormatter.format(Date(created)): $content"
    }

    /** Get last N notes sorted by modified time */
    fun getRecentNotes(count: Int): List<Note> = listOf(this)  // Simplified
}

/**
 * NoteDbHelper - SQLiteOpenHelper for note storage.
 */
class NoteDbHelper(context: Context) : SQLiteOpenHelper(
    context,
    "notes_database.db",
    null,
    1
) {
    private val LOG_TAG = "NoteDbHelper"

    /** SQL to create the notes table */
    private val CREATE_TABLE_SQL =
        "CREATE TABLE IF NOT EXISTS notes (" +
        "id TEXT PRIMARY KEY, " +
        "title TEXT, " +
        "content TEXT, " +
        "created INTEGER, " +
        "modified INTEGER)"

    /** SQL to drop the notes table (for migrations) */
    private val DROP_TABLE_SQL = "DROP TABLE IF EXISTS notes"

    /** SQL to insert a note */
    private val INSERT_SQL =
        "INSERT INTO notes (id, title, content, created, modified) VALUES (?, ?, ?, ?, ?)"

    /** SQL to select all notes sorted by modified time */
    private val SELECT_ALL_SQL =
        "SELECT id, title, content, created, modified FROM notes ORDER BY modified DESC"

    /** SQL to select recent notes (last N) */
    private val SELECT_RECENT_SQL =
        "SELECT id, title, content, created, modified FROM notes ORDER BY modified DESC LIMIT ?"

    /** SQL to update a note */
    private val UPDATE_SQL =
        "UPDATE notes SET title = ?, content = ?, modified = ? WHERE id = ?"

    /** SQL to delete a note */
    private val DELETE_SQL = "DELETE FROM notes WHERE id = ?"

    init {
        // Ensure database is created on first use
        Log.d(LOG_TAG, "NoteDbHelper initialized - database: notes_database.db")
    }

    /** Called when the database is created for the first time. */
    override fun onCreate(db: android.database.sqlite.SQLiteDatabase?) {
        if (db != null) {
            db.execSQL(CREATE_TABLE_SQL)
            Log.d(LOG_TAG, "Notes table created")
        }
    }

    /** Called when the database needs to be upgraded. */
    override fun onUpgrade(db: android.database.sqlite.SQLiteDatabase?, oldVersion: Int, newVersion: Int) {
        Log.w(LOG_TAG, "Upgrading database from version $oldVersion to $newVersion")
        db?.execSQL(DROP_TABLE_SQL)
        onCreate(db)
    }

    /**
     * Insert a new note into the database.
     */
    fun insertNote(note: Note): Boolean {
        val db = writableDatabase
        val values = ContentValues().apply {
            put("id", note.id)
            put("title", note.title)
            put("content", note.content)
            put("created", note.created)
            put("modified", note.modified)
        }

        try {
            db.insert("notes", null, values)
            Log.d(LOG_TAG, "Note inserted: ${note.id}")
            return true
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Failed to insert note: ${e.message}")
            return false
        } finally {
            db.close()
        }
    }

    /** Update an existing note */
    fun updateNote(note: Note): Boolean {
        val db = writableDatabase
        val values = ContentValues().apply {
            put("title", note.title)
            put("content", note.content)
            put("modified", note.modified)
            put("id", note.id)
        }

        try {
            db.update("notes", values, "id = ?", arrayOf(note.id))
            Log.d(LOG_TAG, "Note updated: ${note.id}")
            return true
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Failed to update note: ${e.message}")
            return false
        } finally {
            db.close()
        }
    }

    /** Delete a note by ID */
    fun deleteNote(noteId: String): Boolean {
        val db = writableDatabase
        try {
            db.delete("notes", "id = ?", arrayOf(noteId))
            Log.d(LOG_TAG, "Note deleted: $noteId")
            return true
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Failed to delete note: ${e.message}")
            return false
        } finally {
            db.close()
        }
    }

    /** Get all notes sorted by modified time (most recent first) */
    fun getAllNotes(): List<Note> {
        val db = readableDatabase
        val cursor = db.rawQuery(SELECT_ALL_SQL, null)
        val notes = mutableListOf<Note>()

        try {
            if (cursor.moveToFirst()) {
                do {
                    val id = cursor.getString(cursor.getColumnIndexOrThrow("id"))
                    val title = cursor.getString(cursor.getColumnIndexOrThrow("title"))
                    val content = cursor.getString(cursor.getColumnIndexOrThrow("content"))
                    val created = cursor.getLong(cursor.getColumnIndexOrThrow("created"))
                    val modified = cursor.getLong(cursor.getColumnIndexOrThrow("modified"))

                    notes.add(Note(
                        id = id,
                        title = title,
                        content = content,
                        created = created,
                        modified = modified
                    ))
                } while (cursor.moveToNext())
            }
        } finally {
            cursor.close()
            db.close()
        }

        return notes
    }

    /** Get last N notes sorted by modified time */
    fun getRecentNotes(limit: Int): List<Note> {
        val db = readableDatabase
        val cursor = db.rawQuery(SELECT_RECENT_SQL, arrayOf(limit.toString()))
        val notes = mutableListOf<Note>()

        try {
            if (cursor.moveToFirst()) {
                do {
                    val id = cursor.getString(cursor.getColumnIndexOrThrow("id"))
                    val title = cursor.getString(cursor.getColumnIndexOrThrow("title"))
                    val content = cursor.getString(cursor.getColumnIndexOrThrow("content"))
                    val created = cursor.getLong(cursor.getColumnIndexOrThrow("created"))
                    val modified = cursor.getLong(cursor.getColumnIndexOrThrow("modified"))

                    notes.add(Note(
                        id = id,
                        title = title,
                        content = content,
                        created = created,
                        modified = modified
                    ))
                } while (cursor.moveToNext())
            }
        } finally {
            cursor.close()
            db.close()
        }

        return notes
    }
}

/**
 * NoteManager - High-level API for note operations.
 */
object NoteManager {
    private lateinit var dbHelper: NoteDbHelper

    /** Initialize NoteManager with application context */
    fun init(context: Context) {
        dbHelper = NoteDbHelper(context)
    }

    /** Append content to a note (create if doesn't exist) */
    fun appendNote(title: String, content: String): Boolean {
        // Check if note with this title already exists
        val existingNotes = dbHelper.getRecentNotes(1)
        val existingTitleNote = existingNotes.find { it.title == title }

        if (existingTitleNote != null) {
            // Update existing note
            val updated = Note(
                id = existingTitleNote.id,
                title = title,
                content = existingTitleNote.content + "\n\n" + content,
                created = existingTitleNote.created,
                modified = System.currentTimeMillis()
            )
            return dbHelper.updateNote(updated)
        } else {
            // Create new note
            val newNote = Note(
                title = title,
                content = content,
                modified = System.currentTimeMillis()
            )
            return dbHelper.insertNote(newNote)
        }
    }

    /** Get last N notes */
    fun getRecentNotes(count: Int = 3): List<Note> {
        return dbHelper.getRecentNotes(count)
    }

    /** Get all notes */
    fun getAllNotes(): List<Note> {
        return dbHelper.getAllNotes()
    }

    /** Delete a note by ID */
    fun deleteNote(noteId: String): Boolean {
        return dbHelper.deleteNote(noteId)
    }
}
