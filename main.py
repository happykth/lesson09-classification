import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/kobis_movies.csv"
)
START, END = "2025-09-01", "2026-07-31"
NUM_COLS = ["first_scrn", "first_show", "peak"]
THRESHOLDS = {"50만 명": 500_000, "100만 명": 1_000_000, "300만 명": 3_000_000}
COLOR_SUCCESS = "#E4572E"
COLOR_FAIL = "#4A6FA5"

st.set_page_config(page_title="흥행 분류기", page_icon="🎬", layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    opened = pd.to_datetime(df["openDt"].astype(str), format="%Y%m%d", errors="coerce")
    df = df[(opened >= START) & (opened <= END)].copy()
    df["genre"] = df["genre"].fillna("")
    df["nation"] = df["nation"].fillna("")
    return df.sort_values("movieCd").reset_index(drop=True)


def split_genres(text: str) -> list[str]:
    return [g.strip() for g in text.split("|") if g.strip()]


def make_features(d: pd.DataFrame, genres: list[str], nations: list[str]):
    """숫자 열, 장르(여러 개 가능), 국적을 학습용에서 정한 목록 기준으로 숫자로 바꾼다."""
    num = d[NUM_COLS].astype(float).reset_index(drop=True)
    genre_sets = d["genre"].apply(split_genres).reset_index(drop=True)
    gen = pd.DataFrame({f"장르_{g}": genre_sets.apply(lambda s, g=g: int(g in s)) for g in genres})
    nat = pd.DataFrame(
        {f"국적_{n}": (d["nation"].reset_index(drop=True) == n).astype(int) for n in nations}
    )
    return num, pd.concat([gen, nat], axis=1)


# ---------------------------------------------------------------- 화면 머리말
st.title("🎬 흥행 분류기")
st.warning(
    "이 앱의 정확도는 **이미 기록된 자료로 해 본 분류 연습**의 결과입니다. "
    "**개봉 전에 흥행을 맞히는 예측 성능이 아닙니다.**"
)

try:
    data = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

with st.sidebar:
    st.header("설정")
    label_name = st.radio("성공 기준 (총관객수)", list(THRESHOLDS.keys()), index=1)
    threshold = THRESHOLDS[label_name]
    st.caption(f"total_audi가 {label_name} 이상이면 '성공', 미만이면 '기준 미달'입니다.")
    st.markdown("**입력 변수**: first_scrn, first_show, peak, genre, nation")
    st.caption(
        "first_scrn·first_show는 개봉일이 아니라 **10위권에 처음 관측된 날**의 집계입니다. "
        "first_week_audi, total_audi, days_in_top10은 입력에서 제외했습니다."
    )

# ---------------------------------------------------------------- 자료 나누기
data["label"] = (data["total_audi"] >= threshold).astype(int)
data["actual"] = data["label"].map({1: "성공", 0: "기준 미달"})

# 영화 코드 순으로 정렬된 상태에서 열 편 중 앞 세 편을 시험용으로 고정
is_test = (data.index % 10) < 3
train, test = data[~is_test].copy(), data[is_test].copy()

st.subheader("자료 나누기")
c1, c2, c3 = st.columns(3)
c1.metric("분석 대상 영화", f"{len(data)}편")
c2.metric("학습용", f"{len(train)}편")
c3.metric("시험용", f"{len(test)}편")
st.caption(
    f"2025년 9월~2026년 7월 개봉작을 영화 코드 순으로 놓고, 열 편 중 앞 세 편을 시험용으로 고정했습니다. "
    f"성공 {int(data['label'].sum())}편 · 기준 미달 {int((1 - data['label']).sum())}편 (기준 {label_name})."
)

if train["label"].nunique() < 2:
    st.error("학습용 자료에 한쪽 범주만 있어 모델을 학습할 수 없습니다. 다른 기준을 골라 주세요.")
    st.stop()

# ---------------------------------------------------------------- 학습·표준화 (학습용으로만)
genres = sorted({g for s in train["genre"] for g in split_genres(s)})
nations = sorted(train["nation"].unique())

train_num, train_cat = make_features(train, genres, nations)
test_num, test_cat = make_features(test, genres, nations)

scaler = StandardScaler().fit(train_num)  # 평균·표준편차는 학습용에서만 계산
X_train = pd.concat([pd.DataFrame(scaler.transform(train_num), columns=NUM_COLS), train_cat], axis=1)
X_test = pd.concat([pd.DataFrame(scaler.transform(test_num), columns=NUM_COLS), test_cat], axis=1)
y_train, y_test = train["label"].to_numpy(), test["label"].to_numpy()

logit = LogisticRegression(max_iter=1000).fit(X_train, y_train)
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)

test["prob"] = logit.predict_proba(X_test)[:, 1]
test["pred"] = (test["prob"] >= 0.5).astype(int)
test["pred_name"] = test["pred"].map({1: "성공", 0: "기준 미달"})

acc_logit = float((logit.predict(X_test) == y_test).mean())
acc_tree = float((tree.predict(X_test) == y_test).mean())
majority = max(y_test.mean(), 1 - y_test.mean())

# ---------------------------------------------------------------- 정확도
st.subheader("시험용 자료 정확도")
a1, a2 = st.columns(2)
a1.metric("로지스틱 회귀", f"{acc_logit:.1%}")
a2.metric("결정 트리 (깊이 3)", f"{acc_tree:.1%}")
st.caption(
    f"두 모델 모두 같은 시험용 {len(test)}편으로 계산했습니다. "
    f"참고: 모두 '많은 쪽 범주'로만 답해도 {majority:.1%}가 나오므로, 정확도는 이 값과 비교해서 읽어야 합니다."
)

# ---------------------------------------------------------------- 영화 하나 보기
st.subheader("시험용 영화 하나 살펴보기")
pick = st.selectbox("시험용 영화", test["movieNm"].tolist())
row = test[test["movieNm"] == pick].iloc[0]

r1, r2, r3 = st.columns(3)
r1.metric("추정 확률 (성공)", f"{row['prob']:.3f}")
r2.metric("예측 범주 (확률 0.5 이상이면 성공)", row["pred_name"])
r3.metric("실제 라벨", row["actual"])
st.caption(
    f"{pick} · 장르 {row['genre']} · 국적 {row['nation']} · "
    f"first_scrn {int(row['first_scrn'])} · first_show {int(row['first_show'])} · peak {int(row['peak'])}"
)

# ---------------------------------------------------------------- 막대그래프
st.subheader("시험용 영화의 성공 추정 확률")
ordered = test.sort_values("prob", ascending=False)

fig = go.Figure()
for name, color in [("성공", COLOR_SUCCESS), ("기준 미달", COLOR_FAIL)]:
    part = ordered[ordered["actual"] == name]
    fig.add_bar(
        x=part["movieNm"],
        y=part["prob"],
        name=f"실제 {name}",
        marker_color=color,
        hovertemplate="%{x}<br>추정 확률 %{y:.3f}<extra></extra>",
    )
fig.add_hline(
    y=0.5,
    line_dash="dash",
    line_color="#444444",
    annotation_text="문턱값 0.5",
    annotation_position="top right",
)
fig.update_xaxes(
    categoryorder="array",
    categoryarray=ordered["movieNm"].tolist(),
    tickangle=-45,
    title_text="영화 (추정 확률이 높은 순)",
)
fig.update_yaxes(range=[0, 1.05], title_text="성공 추정 확률")
fig.update_layout(
    barmode="overlay",
    height=560,
    margin=dict(b=140),
    legend=dict(orientation="h", y=1.08, x=0),
)
st.plotly_chart(fig, width="stretch")
st.caption("막대 색은 실제 라벨입니다. 점선(0.5) 위쪽 막대는 성공으로 분류된 영화입니다.")
