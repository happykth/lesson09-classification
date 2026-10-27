import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"

st.set_page_config(page_title=f"탐색 · {APP_TITLE}", page_icon="🔍", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

WARM_PALETTE = ["#F4A259", "#BF360C"]  # 연한 주황, 진한 주황빛 빨강
GROUP_COLORS = {"겪지 않음": "#F4A259", "겪음": "#BF360C"}


@st.cache_data(show_spinner="데이터를 불러오는 중이에요...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def tidy(fig):
    """그림 모양을 조금 다듬는다."""
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), title_x=0.02)
    return fig


# ---------------------------------------------------------------
# 화면 맨 위
# ---------------------------------------------------------------
st.title("🔍 탐색")
st.caption("데이터를 여러 방향에서 살펴보는 화면이에요.")

try:
    df = load_data()
except Exception as error:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n({error})")
    st.stop()

df = df.copy()
df["뇌졸중 여부"] = df["stroke"].map({0: "겪지 않음", 1: "겪음"})

# ---------------------------------------------------------------
# 1) 나이와 평균 혈당의 분포
# ---------------------------------------------------------------
st.subheader("📊 나이와 평균 혈당의 분포")
left, right = st.columns(2)

fig_age = px.histogram(
    df,
    x="age",
    nbins=30,
    title="나이의 분포",
    labels={"age": "나이", "count": "사람 수"},
    color_discrete_sequence=[WARM_PALETTE[0]],
)
fig_age.update_layout(yaxis_title="사람 수", bargap=0.05)
left.plotly_chart(tidy(fig_age), use_container_width=True)

fig_glucose = px.histogram(
    df,
    x="avg_glucose_level",
    nbins=40,
    title="평균 혈당의 분포",
    labels={"avg_glucose_level": "평균 혈당"},
    color_discrete_sequence=[WARM_PALETTE[1]],
)
fig_glucose.update_layout(yaxis_title="사람 수", bargap=0.05)
right.plotly_chart(tidy(fig_glucose), use_container_width=True)

# ---------------------------------------------------------------
# 2) 뇌졸중 여부에 따른 나이와 평균 혈당 비교
# ---------------------------------------------------------------
st.subheader("📦 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")
order = {"뇌졸중 여부": ["겪지 않음", "겪음"]}
left, right = st.columns(2)

fig_box_age = px.box(
    df,
    x="뇌졸중 여부",
    y="age",
    color="뇌졸중 여부",
    category_orders=order,
    color_discrete_map=GROUP_COLORS,
    title="나이 비교",
    labels={"age": "나이"},
)
fig_box_age.update_layout(showlegend=False)
left.plotly_chart(tidy(fig_box_age), use_container_width=True)

fig_box_glucose = px.box(
    df,
    x="뇌졸중 여부",
    y="avg_glucose_level",
    color="뇌졸중 여부",
    category_orders=order,
    color_discrete_map=GROUP_COLORS,
    title="평균 혈당 비교",
    labels={"avg_glucose_level": "평균 혈당"},
)
fig_box_glucose.update_layout(showlegend=False)
right.plotly_chart(tidy(fig_box_glucose), use_container_width=True)

mean_table = (
    df.groupby("뇌졸중 여부")
    .agg(
        사람_수=("stroke", "size"),
        나이_평균=("age", "mean"),
        평균혈당_평균=("avg_glucose_level", "mean"),
    )
    .reindex(["겪지 않음", "겪음"])
    .round(2)
    .reset_index()
)
mean_table.columns = ["뇌졸중 여부", "사람 수", "나이 평균", "평균 혈당 평균"]
st.markdown("**두 그룹의 평균값**")
st.dataframe(mean_table, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------
# 3) 고혈압·심장병 여부에 따른 뇌졸중 비율
# ---------------------------------------------------------------
st.subheader("📈 고혈압·심장병 여부에 따른 뇌졸중 비율")


def stroke_rate_by(column: str, label: str):
    rate = (
        df.groupby(column)["stroke"]
        .mean()
        .mul(100)
        .round(2)
        .rename("뇌졸중 비율(%)")
        .reset_index()
    )
    rate[label] = rate[column].map({0: "없음", 1: "있음"})
    fig = px.bar(
        rate,
        x=label,
        y="뇌졸중 비율(%)",
        color=label,
        text="뇌졸중 비율(%)",
        category_orders={label: ["없음", "있음"]},
        color_discrete_map={"없음": "#F4A259", "있음": "#BF360C"},
        title=f"{label}에 따른 뇌졸중 비율",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="뇌졸중 비율(%)")
    fig.update_yaxes(range=[0, max(rate["뇌졸중 비율(%)"].max() * 1.25, 1)])
    return fig


left, right = st.columns(2)
left.plotly_chart(tidy(stroke_rate_by("hypertension", "고혈압")), use_container_width=True)
right.plotly_chart(tidy(stroke_rate_by("heart_disease", "심장병")), use_container_width=True)

# ---------------------------------------------------------------
# 4) bmi가 빈 사람들의 뇌졸중 비율
# ---------------------------------------------------------------
st.subheader("🧮 bmi가 비어 있는 사람들")
bmi_missing = df[df["bmi"].isna()]


def summary_row(name: str, part: pd.DataFrame) -> dict:
    count = len(part)
    stroke_count = int((part["stroke"] == 1).sum())
    return {
        "구분": name,
        "사람 수": count,
        "뇌졸중 사람 수": stroke_count,
        "뇌졸중 비율(%)": round(stroke_count / count * 100, 2) if count else 0.0,
    }


bmi_table = pd.DataFrame(
    [
        summary_row("bmi가 빈 사람", bmi_missing),
        summary_row("전체", df),
    ]
)
st.dataframe(bmi_table, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------
# 5) 흡연 상태별 사람 수
# ---------------------------------------------------------------
st.subheader("🚬 흡연 상태별 사람 수")
smoking_table = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking_table, hide_index=True, use_container_width=True)
