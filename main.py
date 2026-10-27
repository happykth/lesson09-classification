import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# 기본 설정 (브라우저 탭 제목과 아이콘)
# ---------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

DATA_URL = "https://raw.githubusercontent.com/happykth/data//main/stroke.csv"
SAVE_DIR = Path(__file__).parent / "내_기록"
SAVE_FILE = SAVE_DIR / "소개_화면_기록.json"


# ---------------------------------------------------------------
# 기록 불러오기 / 저장하기
# ---------------------------------------------------------------
def load_notes() -> dict:
    """내_기록 폴더의 json 파일에서 우리말 뜻과 출처를 불러온다."""
    if SAVE_FILE.exists():
        try:
            data = json.loads(SAVE_FILE.read_text(encoding="utf-8"))
            return {
                "meanings": dict(data.get("meanings", {})),
                "source": str(data.get("source", "")),
            }
        except (json.JSONDecodeError, OSError):
            pass
    return {"meanings": {}, "source": ""}


def save_notes(notes: dict) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_FILE.write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------------
@st.cache_data(show_spinner="데이터를 불러오는 중이에요...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def describe_values(series: pd.Series, total_rows: int) -> str:
    """열의 '값의 종류'를 짧은 글로 설명한다."""
    values = series.dropna()
    n_unique = values.nunique()
    if n_unique == total_rows:
        return f"겹치지 않는 번호 ({n_unique:,}개)"
    if n_unique <= 10:
        shown = ", ".join(str(v) for v in sorted(values.unique(), key=str))
        return f"{n_unique}가지: {shown}"
    if pd.api.types.is_numeric_dtype(series):
        return f"숫자 (최소 {values.min():g} ~ 최대 {values.max():g})"
    return f"{n_unique}가지 글자 값"


# ---------------------------------------------------------------
# 스타일 (따뜻한 색감의 큰 숫자 카드)
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    .stat-card {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        border: 1px solid #FFCC80;
        border-radius: 16px;
        padding: 22px 12px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(191, 120, 40, 0.15);
    }
    .stat-label { font-size: 1.0rem; color: #8D5A1F; margin-bottom: 6px; }
    .stat-value { font-size: 2.8rem; font-weight: 800; color: #BF360C; line-height: 1.1; }
    </style>
    """,
    unsafe_allow_html=True,
)


def stat_card(label: str, value: str) -> str:
    return (
        f'<div class="stat-card"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div></div>'
    )


# ---------------------------------------------------------------
# 화면 맨 위: 아이콘 + 제목
# ---------------------------------------------------------------
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("뇌졸중 데이터가 어떤 데이터인지 먼저 살펴보는 소개 화면이에요.")

try:
    df = load_data()
except Exception as error:  # 인터넷 연결 문제 등
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n({error})")
    st.stop()

notes = load_notes()

# ---------------------------------------------------------------
# 1) 큰 숫자 카드 네 개
# ---------------------------------------------------------------
total_people = len(df)
total_columns = df.shape[1]
stroke_count = int((df["stroke"] == 1).sum())
stroke_ratio = stroke_count / total_people * 100

c1, c2, c3, c4 = st.columns(4)
c1.markdown(stat_card("전체 사람 수", f"{total_people:,}명"), unsafe_allow_html=True)
c2.markdown(stat_card("열 개수", f"{total_columns}개"), unsafe_allow_html=True)
c3.markdown(
    stat_card("stroke가 1인 사람 수", f"{stroke_count:,}명"), unsafe_allow_html=True
)
c4.markdown(
    stat_card("stroke가 1인 비율", f"{stroke_ratio:.2f}%"), unsafe_allow_html=True
)

# ---------------------------------------------------------------
# 2) 열 설명 표 (우리말 뜻 칸은 직접 입력)
# ---------------------------------------------------------------
st.subheader("📋 열 설명 표")
st.caption("‘우리말 뜻’ 칸을 더블클릭해서 교재를 보고 직접 적어 보세요. 적은 내용은 자동으로 저장돼요.")

table = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [notes["meanings"].get(col, "") for col in df.columns],
        "값의 종류": [describe_values(df[col], total_people) for col in df.columns],
        "빈 값 개수": [int(df[col].isna().sum()) for col in df.columns],
    }
)

edited = st.data_editor(
    table,
    key="column_table",
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻", help="직접 적어 넣는 칸"),
    },
)

new_meanings = {
    row["열 이름"]: str(row["우리말 뜻"] or "") for _, row in edited.iterrows()
}
if new_meanings != notes["meanings"]:
    notes["meanings"] = new_meanings
    save_notes(notes)

# ---------------------------------------------------------------
# 3) 데이터 처음 다섯 줄
# ---------------------------------------------------------------
st.subheader("👀 데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# 4) 데이터 출처 (직접 입력)
# ---------------------------------------------------------------
st.subheader("📚 데이터 출처")
st.caption("교재에 나온 출처 글을 아래에 적어 주세요. 입력 칸 밖을 클릭하면 저장돼요.")

source_text = st.text_area(
    "데이터 출처",
    value=notes["source"],
    height=120,
    placeholder="여기에 교재에 적힌 데이터 출처를 적어 주세요.",
    label_visibility="collapsed",
    key="source_text",
)
if source_text != notes["source"]:
    notes["source"] = source_text
    save_notes(notes)
