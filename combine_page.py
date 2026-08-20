import io
from datetime import datetime
import google.genai as genai
from google.genai import types
from gtts import gTTS
import pandas as pd
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import streamlit as st

# ==========================================
# 1. DATABASE CONFIGURATION & FUNCTIONS
# ==========================================

def get_db_connection():
    """Establish connection to Supabase PostgreSQL using secrets."""
    return psycopg2.connect(st.secrets["DB_URI"])


def get_subject_details(topic):
    """Fetch hourly_rate and default_tutor from subject_rates table."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                query = "SELECT hourly_rate, default_tutor FROM subject_rates WHERE topic = %s"
                cursor.execute(query, (topic,))
                row = cursor.fetchone()
                if row:
                    return row[0], row[1]
                return None, None
    except Exception:
        return None, None


def log_tutoring_session(
    session_date, topic, tutor, login_time, logout_time, total_time, fees
):
    """Insert logged session record into tutoring_sessions table."""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            query = """
                INSERT INTO tutoring_sessions 
                (sessiondate, topic, tutor, login_time, logout_time, totaltime, fees)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(
                query,
                (
                    session_date,
                    topic,
                    tutor,
                    login_time.strftime("%H:%M:%S"),
                    logout_time.strftime("%H:%M:%S"),
                    total_time,
                    fees,
                ),
            )
            conn.commit()


def log_to_chat_history(user_query, tutor_response):
    """Insert interaction into chathistory table."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                query = """
                    INSERT INTO chathistory (userquery, tutorresponse)
                    VALUES (%s, %s)
                """
                cursor.execute(query, (user_query, tutor_response))
                conn.commit()
    except Exception as e:
        st.error(f"Failed to record ChatHistory: {e}")


# ==========================================
# 2. INTENT ROUTER SETUP (SCIKIT-LEARN)
# ==========================================
@st.cache_resource
def train_intent_router():
    training_data = {
        "prompt": [
            "Hi",
            "Hello",
            "Good morning",
            "Hey tutor",
            "What is the hourly rate for Math?",
            "How much does Chemistry cost?",
            "Who is the default tutor for Physics?",
            "Show subject rates",
            "Explain Baudhayana Pythagoras theorem",
            "How do I solve quadrilaterals?",
            "What are perfect squares and cube numbers?",
            "How do exponents work?",
            "Help me expand algebraic expressions",
            "How to calculate area of a polygon?",
            "Solve maths matrix class 8 question",
        ],
        "intent": [
            "GREETING",
            "GREETING",
            "GREETING",
            "GREETING",
            "DATABASE_QUERY",
            "DATABASE_QUERY",
            "DATABASE_QUERY",
            "DATABASE_QUERY",
            "CLASS_8_MATH",
            "CLASS_8_MATH",
            "CLASS_8_MATH",
            "CLASS_8_MATH",
            "CLASS_8_MATH",
            "CLASS_8_MATH",
            "CLASS_8_MATH",
        ],
    }
    df = pd.DataFrame(training_data)
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(df["prompt"], df["intent"])
    return model


router_model = train_intent_router()

# Initialize Gemini API Client securely from Streamlit Secrets
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# Initialize session state counter for resetting input widgets
if "session_counter" not in st.session_state:
    st.session_state.session_counter = 0

# ==========================================
# 3. STREAMLIT APPLICATION LAYOUT & STYLING
# ==========================================
st.set_page_config(page_title="AI Tutor Platform", layout="wide")

# Custom CSS for soft green background, ultra-faint watermark, and tight top spacing
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    .stApp {
        background-color: #f0f7f2;
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="220" height="220" viewBox="0 0 220 220" opacity="0.03"><text x="20" y="40" font-size="28" font-family="sans-serif" fill="%232e7d32">π</text><text x="120" y="50" font-size="28" font-family="sans-serif" fill="%232e7d32">+</text><text x="60" y="110" font-size="26" font-family="sans-serif" fill="%232e7d32">√x</text><text x="160" y="130" font-size="30" font-family="sans-serif" fill="%232e7d32">÷</text><text x="20" y="180" font-size="26" font-family="sans-serif" fill="%232e7d32">∑</text><text x="140" y="190" font-size="28" font-family="sans-serif" fill="%232e7d32">∞</text></svg>');
        background-repeat: repeat;
    }

    div[data-testid="stColumn"], div[data-testid="stTabContent"] {
        background-color: #ffffff;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top Banner Header Layout
col_title, col_img = st.columns([8, 2], vertical_alignment="center")

with col_title:
    st.subheader("📚 Tutor for Grace Darnal")

with col_img:
    st.image("grace.jpeg", width=130)

tab1, tab2 = st.tabs(["💬 AI Student Chat", "📝 Log Session to Database"])

# ------------------------------------------
# TAB 1: AI STUDENT CHAT (INTENT ROUTER)
# ------------------------------------------
with tab1:
    header_col, reset_col = st.columns([7, 3], vertical_alignment="center")
    
    with header_col:
        st.markdown("#### Student AI Interaction")
    
    with reset_col:
        if st.button("🔄 Start New Chat / Clear", use_container_width=True):
            st.session_state.session_counter += 1
            st.rerun()

    selected_topic = st.selectbox(
        "Select Session Topic",
        ["Math", "Quadrilaterals", "Exponents", "Algebra", "General Query"],
        key=f"topic_{st.session_state.session_counter}"
    )

    st.subheader("1. Speak to AI")
    # Dynamic key forces Streamlit to render a fresh, unrecorded audio widget on reset
    audio_input = st.audio_input(
        "Click the mic to record your question", 
        key=f"audio_input_{st.session_state.session_counter}"
    )

    def process_query(user_text):
        intent = router_model.predict([user_text])[0]
        st.info(f"⚡ Router Decision: Classified as **{intent}**")

        if intent == "GREETING":
            answer = (
                "Hello Grace! I am your Class 8 Math tutor following MATHS"
                " MATRIX by Gourdas Saha. What concept are we working on"
                " today?"
            )

        elif intent == "DATABASE_QUERY":
            rate, tutor = get_subject_details(selected_topic)
            if rate:
                answer = (
                    f"The current hourly rate for {selected_topic} is ₹{rate}/hr"
                    f" with Tutor {tutor}."
                )
            else:
                answer = (
                    f"Topic '{selected_topic}' was not found in database"
                    " records."
                )

        elif intent == "CLASS_8_MATH":
            system_prompt = (
                "You are an expert Class 8 Mathematics tutor. Your guidance"
                " is aligned specifically with the textbook 'MATHS MATRIX –"
                " CLASS 8' by Gourdas Saha.\n"
                "Explain concepts step-by-step, clearly, and concisely in an"
                f" encouraging tone suitable for an 8th-grade student on topic:"
                f" {selected_topic}.\n"
                f"Question: {user_text}"
            )
            response = client.models.generate_content(
                model="gemini-3.6-flash", contents=system_prompt
            )
            answer = response.text

        # Automatically log interaction to Supabase ChatHistory
        log_to_chat_history(user_text, answer)
        return answer

    if audio_input:
        with st.spinner("Analyzing audio..."):
            try:
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=[
                        (
                            "System: You are an encouraging Class 8 Math tutor"
                            " following MATHS MATRIX by Gourdas Saha. Assist"
                            f" with {selected_topic}."
                        ),
                        types.Part.from_bytes(
                            data=audio_input.read(), mime_type="audio/wav"
                        ),
                    ],
                )
                st.markdown(f"**AI Tutor:** {response.text}")
                log_to_chat_history("[Voice Input]", response.text)

                speech_lang = st.radio(
                    "Voice Language",
                    ["en", "ne"],
                    format_func=lambda x: "English" if x == "en" else "Nepali",
                    key=f"lang_{st.session_state.session_counter}"
                )

                tts = gTTS(text=response.text, lang=speech_lang)
                sound_file = io.BytesIO()
                tts.write_to_fp(sound_file)
                sound_file.seek(0)
                st.audio(sound_file, format="audio/mp3", autoplay=True)
            except Exception as e:
                st.error(f"Error processing audio: {e}")

    st.divider()

    st.subheader("2. Or Type Your Question")
    if text_prompt := st.chat_input("Type your math query here...", key=f"chat_{st.session_state.session_counter}"):
        with st.chat_message("user"):
            st.markdown(text_prompt)

        with st.chat_message("assistant"):
            answer = process_query(text_prompt)
            st.markdown(answer)

# ------------------------------------------
# TAB 2: LOG SESSION TO DATABASE
# ------------------------------------------
with tab2:
    st.markdown("#### Tutoring Session Logger")

    col1, col2 = st.columns(2)
    with col1:
        session_date = st.date_input("Session Date", datetime.today())
        topic = st.text_input("Topic", "Math")

    with col2:
        login_time = st.time_input("Login Time")
        logout_time = st.time_input("Logout Time")

    if st.button("Log Session"):
        if logout_time <= login_time:
            st.error("Logout time must be later than Login time.")
        else:
            hourly_rate, default_tutor = get_subject_details(topic)

            if hourly_rate is None:
                st.error(
                    f"Topic '{topic}' was not found in subject_rates table."
                )
            else:
                dummy_date = datetime.today().date()
                dt_start = datetime.combine(dummy_date, login_time)
                dt_end = datetime.combine(dummy_date, logout_time)

                duration_seconds = (dt_end - dt_start).total_seconds()
                total_time = round(duration_seconds / 3600.0, 2)
                fees = int(round(total_time * hourly_rate))

                try:
                    log_tutoring_session(
                        session_date,
                        topic,
                        default_tutor,
                        login_time,
                        logout_time,
                        total_time,
                        fees,
                    )
                    st.success("Session logged successfully!")
                    st.json({
                        "Tutor Assigned": default_tutor,
                        "Duration (Hours)": total_time,
                        "Calculated Fees": fees,
                    })
                except Exception as e:
                    st.error(f"Database write failed: {e}")
