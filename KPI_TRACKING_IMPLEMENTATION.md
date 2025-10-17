# KPI Tracking System Implementation

## Overview
Implemented a comprehensive KPI (Key Performance Indicator) tracking system that monitors and stores detailed metrics for each call in JSON files, similar to how conversation transcripts and prompt logs are managed.

## Storage Location
KPI logs are stored in: `/home/novel/server/kpi_logs/`

Format: `kpi_YYYYMMDD_HHMMSS_sessionid.json`

## What Gets Tracked

### 1. **Call Metadata**
- Session ID and Connection ID
- Call start/end timestamps
- Total call duration (in seconds)

### 2. **Conversation Metrics**
- Total turns (combined user + AI)
- User turns (how many times candidate spoke)
- AI turns (how many times recruiter responded)
- Current conversation stage (greeting → background → details → location → role_discussion → closing)
- Stages completed during the call

### 3. **Information Collection Tracking**
Automatically detects and tracks:
- ✅ Candidate name (extracted and confirmed)
- ✅ Experience discussed
- ✅ Current employer mentioned
- ✅ CTC (current salary) discussed
- ✅ Expected CTC discussed
- ✅ Notice period discussed
- ✅ Location discussed
- ✅ Role interest confirmed

### 4. **Call Outcome**
- Current status: `in_progress`, `completed`, `disconnected`, `interview_scheduled`, `callback_needed`
- Interview scheduled flag
- Callback requested flag

### 5. **Technical Metrics**
- Audio chunks processed (total)
- Audio chunks rejected (failed validation)
- Transcription requests (Whisper calls)
- Ultravox requests (AI model calls)
- TTS generations (text-to-speech conversions)

### 6. **Performance Metrics**
- Average response time (in seconds)
- Individual response times for each AI request
- Errors encountered
- Detailed error log with timestamps

### 7. **Token Usage** (for future cost tracking)
- Total prompt tokens
- Total completion tokens
- Total tokens used

## KPI JSON Structure

```json
{
  "kpi_session_id": "unique-kpi-session-id",
  "connection_id": "websocket-connection-id",
  "linked_session_id": "conversation-session-id",
  "created_at": "20251013_120000",
  "last_updated": "2025-10-13 12:05:30",
  "kpis": {
    // Call metadata
    "call_start_time": "2025-10-13 12:00:00",
    "call_end_time": "2025-10-13 12:05:30",
    "call_duration_seconds": 330.45,
    
    // Conversation metrics
    "total_turns": 18,
    "user_turns": 9,
    "ai_turns": 9,
    "stage_reached": "details",
    "stages_completed": ["greeting", "background", "details"],
    
    // Information collected
    "information_collected": {
      "candidate_name": "Surya",
      "candidate_name_confirmed": true,
      "experience_discussed": true,
      "current_employer_mentioned": true,
      "ctc_discussed": true,
      "expected_ctc_discussed": true,
      "notice_period_discussed": false,
      "location_discussed": true,
      "role_interest_confirmed": true
    },
    
    // Call outcome
    "call_outcome": "interview_scheduled",
    "interview_scheduled": true,
    "callback_requested": false,
    
    // Technical metrics
    "audio_chunks_processed": 45,
    "audio_chunks_rejected": 12,
    "transcription_requests": 9,
    "ultravox_requests": 9,
    "tts_generations": 9,
    
    // Performance metrics
    "avg_response_time_seconds": 2.34,
    "response_times": [2.1, 2.5, 2.3, ...],
    "errors_encountered": 0,
    "error_log": []
  }
}
```

## How It Works

### 1. **Session Creation**
When a conversation starts, KPI tracking is automatically initialized:
```python
kpi_tracker.create_session(connection_id, session_id)
```

### 2. **Real-Time Tracking**
Throughout the conversation, KPIs are updated automatically:

- **User Turn**: Every time user speaks
- **AI Turn**: Every time AI responds
- **Audio Processing**: Tracks accepted/rejected audio chunks
- **Model Requests**: Counts Whisper, Ultravox, and TTS calls
- **Response Times**: Measures AI response latency
- **Stage Updates**: Tracks conversation flow progress
- **Information Extraction**: Automatically detects discussed topics

### 3. **Session End**
When the call ends, final metrics are calculated and saved:
```python
kpi_tracker.end_session(connection_id, outcome="completed")
```

## Integration Points

### Files Modified:

1. **`kpi_tracker.py`** (NEW)
   - Core KPI tracking module
   - 500+ lines of tracking logic
   - Automatic information extraction from conversations

2. **`websocket_handler.py`**
   - Integrated KPI session lifecycle
   - Tracks audio processing events
   - Tracks model requests and response times
   - Tracks errors

3. **`transcription_manager.py`**
   - Tracks user/AI conversation turns
   - Updates conversation stage in KPIs
   - Updates candidate name when detected

## Automatic Information Extraction

The system intelligently analyzes conversation content to detect discussed topics:

### User Input Analysis:
```python
# Detects experience mentions
if "year" or "experience" in user_input:
    kpis["information_collected"]["experience_discussed"] = True

# Detects CTC discussion
if "ctc" or "salary" or "lakh" in user_input:
    kpis["information_collected"]["ctc_discussed"] = True

# Detects location discussion
if "bangalore" or "marathahalli" in user_input:
    kpis["information_collected"]["location_discussed"] = True
```

### AI Response Analysis:
```python
# Detects interview scheduling
if "schedule" or "interview" in ai_response:
    kpis["interview_scheduled"] = True
    kpis["call_outcome"] = "interview_scheduled"

# Detects callback request
if "call back" or "get back" in ai_response:
    kpis["callback_requested"] = True
```

## Conversation Stages

The system tracks 6 conversation stages:

1. **Greeting** (0-2 turns)
   - Initial introduction
   - Name confirmation
   - Availability check

2. **Background** (3-6 turns)
   - Experience discussion
   - Current role information

3. **Details** (7-12 turns)
   - CTC collection
   - Notice period
   - Structured data gathering

4. **Location** (13-16 turns)
   - Location discussion
   - Commute feasibility

5. **Role Discussion** (17-20 turns)
   - BDM role explanation
   - Interest confirmation

6. **Closing** (21+ turns)
   - Interview scheduling
   - Next steps

## Use Cases

### 1. **Call Quality Analysis**
```bash
# Check average response time across calls
grep "avg_response_time_seconds" kpi_logs/*.json

# Find calls with high error rates
grep "errors_encountered" kpi_logs/*.json | grep -v ": 0"
```

### 2. **Recruitment Funnel Metrics**
```python
# Count interview conversions
interview_scheduled = count(kpis["interview_scheduled"] == True)

# Identify incomplete information collection
missing_ctc = count(kpis["information_collected"]["ctc_discussed"] == False)
```

### 3. **Performance Monitoring**
- Monitor average call duration
- Track response time trends
- Identify audio validation success rate
- Measure conversation completion rates

### 4. **Agent Performance**
- Stage progression rates
- Information collection completeness
- Call outcome distribution

## Example Queries

### Find all successful interview schedules:
```bash
grep -l '"interview_scheduled": true' kpi_logs/*.json
```

### Calculate average call duration:
```bash
grep "call_duration_seconds" kpi_logs/*.json | awk '{sum+=$2; count++} END {print sum/count}'
```

### Find calls where candidate name wasn't captured:
```bash
grep -l '"candidate_name": null' kpi_logs/*.json
```

### Check audio rejection rate:
```bash
# Total rejected vs processed
grep "audio_chunks_rejected" kpi_logs/*.json
grep "audio_chunks_processed" kpi_logs/*.json
```

## Benefits

### ✅ **Comprehensive Metrics**
Every call is fully tracked from start to finish

### ✅ **Automatic Collection**
No manual intervention required - KPIs are collected automatically

### ✅ **Real-Time Updates**
KPI file is updated throughout the conversation

### ✅ **Intelligent Extraction**
Automatically detects discussed topics and information

### ✅ **Performance Insights**
Track response times, error rates, and system performance

### ✅ **Business Intelligence**
Analyze recruitment funnel, conversion rates, and agent effectiveness

### ✅ **Quality Assurance**
Identify problematic calls, incomplete information, and technical issues

## Directory Structure

```
server/
├── kpi_logs/                          # KPI tracking directory (NEW)
│   ├── kpi_20251013_120000_abc123.json
│   ├── kpi_20251013_130000_def456.json
│   └── EXAMPLE_kpi_*.json            # Example/template
├── conversation_history/              # Conversation transcripts
│   └── conversation_*.json
├── prompt_logs/                       # Prompt logging
│   └── prompts_*.json
├── kpi_tracker.py                     # KPI tracking module (NEW)
├── transcription_manager.py           # Updated with KPI integration
└── websocket_handler.py               # Updated with KPI integration
```

## Future Enhancements

### Potential Additions:
1. **Token Cost Calculation**: Track actual token usage from API
2. **Sentiment Analysis**: Track candidate mood/sentiment
3. **Voice Quality Metrics**: Track audio quality scores
4. **Competitive Analysis**: Compare against other BDM calls
5. **ML Predictions**: Predict interview success probability
6. **Real-time Dashboard**: Live KPI visualization
7. **Automated Alerts**: Notify on anomalies or issues
8. **Export to Database**: Push KPIs to PostgreSQL/MongoDB
9. **Analytics API**: RESTful API for KPI queries
10. **Batch Reporting**: Daily/weekly KPI summary reports

## Monitoring and Debugging

### Check if KPI tracking is working:
```bash
# Check latest KPI file
ls -lt server/kpi_logs/ | head -5

# View a KPI file
cat server/kpi_logs/kpi_*.json | jq '.'

# Monitor KPI creation in real-time
watch -n 1 'ls -lt server/kpi_logs/ | head -5'
```

### Verify KPI completeness:
```bash
# Check if all sessions have KPIs
ls conversation_history/ | wc -l
ls kpi_logs/ | wc -l
```

## Example KPI File

See: `/home/novel/server/kpi_logs/EXAMPLE_kpi_20251013_120000_sample123.json`

This example shows a successful call where:
- Duration: 5 minutes 30 seconds
- 18 total conversation turns (9 each)
- Reached "details" stage
- Collected: name, experience, employer, CTC, location, interest
- Outcome: Interview scheduled
- Performance: 2.34s average response time
- Audio: 45 chunks processed, 12 rejected (73% acceptance)
- No errors encountered

## Status

✅ **Fully Implemented and Integrated**
- KPI tracking module created
- WebSocket handler integrated
- Transcription manager integrated
- Automatic session lifecycle management
- Real-time metric updates
- Comprehensive information extraction
- Example documentation provided

## Testing

To test KPI tracking:
1. Start the WebSocket server
2. Connect a client and have a conversation
3. Check `/home/novel/server/kpi_logs/` for the generated KPI file
4. Verify all metrics are being tracked correctly
5. Compare with conversation transcript for accuracy


