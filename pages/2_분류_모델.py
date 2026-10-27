import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"

st.set_page_config(page_title=f"분류 모델 · {APP_TITLE}", page_icon="🤖", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
KOR = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
BINARY = {"hypertension", "heart_disease"}  # 0 또는 1 두 값뿐인 속성

NAME_LOGIT = "로지스틱 회귀(확률로 답하는 모델)"
NAME_TREE = "의사결정트리(질문으로 답하는 모델)"
NAME_BASE = "기준선 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

C_NO, C_YES = "#3F7CAC", "#D84315"  # 점 색: 아님 / 뇌졸중
R_NO, R_YES = "#E3F0F9", "#FFE0D6"  # 칸 색(옅은 색): 아님 / 뇌졸중
GRID_N = 150  # 그림을 칠할 격자 한 변의 칸 수


@st.cache_data(show_spinner="데이터를 불러오는 중이에요...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.markdown(
    """
    <style>
    .acc-card {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        border: 1px solid #FFCC80;
        border-radius: 16px;
        padding: 18px 14px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(191, 120, 40, 0.15);
        height: 100%;
    }
    .acc-name { font-size: 0.95rem; color: #8D5A1F; font-weight: 600; min-height: 3.2em; }
    .acc-cap { font-size: 0.85rem; color: #8D5A1F; margin-top: 4px; }
    .acc-big { font-size: 2.8rem; font-weight: 800; color: #BF360C; line-height: 1.1; }
    .acc-small { display: flex; justify-content: space-around; margin-top: 10px;
                 font-size: 0.85rem; color: #6D4C41; }
    </style>
    """,
    unsafe_allow_html=True,
)


def acc_card(name: str, train_acc: float, test_acc: float) -> str:
    return (
        f'<div class="acc-card">'
        f'<div class="acc-name">{name}</div>'
        f'<div class="acc-big">{test_acc * 100:.2f}%</div>'
        f'<div class="acc-cap">테스트 데이터 정확도</div>'
        f'<div class="acc-small">'
        f"<span>훈련 데이터 {train_acc * 100:.2f}%</span>"
        f"<span>테스트 데이터 {test_acc * 100:.2f}%</span>"
        f"</div></div>"
    )


# ---------------------------------------------------------------
# 화면 맨 위
# ---------------------------------------------------------------
st.title("🤖 분류 모델")
st.caption("뇌졸중이면 1, 아니면 0이에요. 뇌졸중을 양성(찾아내려는 쪽)으로 두고 두 모델을 비교해요.")

try:
    raw = load_data()
except Exception as error:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n({error})")
    st.stop()

# ---------------------------------------------------------------
# 1) 입력으로 사용할 속성 고르기
# ---------------------------------------------------------------
chosen = st.multiselect(
    "입력으로 사용할 속성",
    options=FEATURES,
    default=[f for f in FEATURES if f != "bmi"],
    format_func=lambda c: KOR[c],
)
selected = [f for f in FEATURES if f in chosen]  # 원래 순서로 정리

if len(selected) < 2:
    st.info("입력으로 사용할 속성을 두 개 이상 골라 주세요. 두 개 이상 고르면 모델이 만들어져요.")
    st.stop()

# ---------------------------------------------------------------
# 2) 테스트용 / 학습용 나누기 (번호 순으로 정렬 → 열 명씩 묶어 앞 세 명이 테스트용)
# ---------------------------------------------------------------
df = raw.sort_values("id").reset_index(drop=True)
is_test = pd.Series(df.index % 10 < 3, index=df.index)

X_all = df[selected].copy()
y_all = df["stroke"]

X_train, X_test = X_all[~is_test].copy(), X_all[is_test].copy()
y_train, y_test = y_all[~is_test], y_all[is_test]

fill_note = ""
if "bmi" in selected:
    bmi_median = X_train["bmi"].median()  # 훈련용의 중앙값으로만 채운다
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    fill_note = f" 체질량지수의 빈 값은 훈련용의 중앙값({bmi_median:.1f})으로 채웠어요."

st.write(
    f"번호 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했어요. "
    f"테스트용 **{len(X_test):,}명**, 학습용 **{len(X_train):,}명**이에요.{fill_note}"
)

# ---------------------------------------------------------------
# 3) 크기 맞추기와 모델 만들기 (모두 학습용으로만 배운다)
# ---------------------------------------------------------------
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

logit = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_s, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(
    X_train, y_train
)
baseline = DummyClassifier(strategy="most_frequent").fit(X_train_s, y_train)

st.caption(
    "크기 맞추기(표준화)는 학습용 사람들만으로 계산했어요. 로지스틱 회귀는 크기를 맞춘 값으로 배우고, "
    "의사결정트리는 질문이 읽기 쉽도록 원래 값 그대로 배워요(나무는 크기를 맞춰도 나누는 결과가 같아요). "
    "의사결정트리는 질문을 세 번까지만 던지고, 마지막 마디에 사람이 다섯 명 미만이 되게 나누지 않아요."
)

# ---------------------------------------------------------------
# 4) 정확도 카드
# ---------------------------------------------------------------
st.subheader("🎯 정확도 비교")
c1, c2, c3 = st.columns(3)
c1.markdown(
    acc_card(NAME_LOGIT, logit.score(X_train_s, y_train), logit.score(X_test_s, y_test)),
    unsafe_allow_html=True,
)
c2.markdown(
    acc_card(NAME_TREE, tree.score(X_train, y_train), tree.score(X_test, y_test)),
    unsafe_allow_html=True,
)
c3.markdown(
    acc_card(
        NAME_BASE,
        baseline.score(X_train_s, y_train),
        baseline.score(X_test_s, y_test),
    ),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# 5) 두 속성으로 그린 산점도와 두 모델의 경계
# ---------------------------------------------------------------
st.subheader("🗺️ 두 속성으로 본 테스트 데이터와 모델의 경계")

s1, s2 = st.columns(2)
x_col = s1.selectbox("가로축", selected, index=0, format_func=lambda c: KOR[c])
y_col = s2.selectbox("세로축", selected, index=1, format_func=lambda c: KOR[c])

if x_col == y_col:
    st.warning("가로축과 세로축은 서로 다른 속성으로 골라 주세요.")
else:

    def axis_range(col: str):
        if col in BINARY:
            return -0.4, 1.4
        lo, hi = float(X_test[col].min()), float(X_test[col].max())
        pad = (hi - lo) * 0.05 or 1.0
        return lo - pad, hi + pad

    x_lo, x_hi = axis_range(x_col)
    y_lo, y_hi = axis_range(y_col)
    xs = [x_lo + (x_hi - x_lo) * i / (GRID_N - 1) for i in range(GRID_N)]
    ys = [y_lo + (y_hi - y_lo) * i / (GRID_N - 1) for i in range(GRID_N)]

    # 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 둔다
    medians = X_test[selected].median()
    grid = pd.DataFrame({c: [float(medians[c])] * (GRID_N * GRID_N) for c in selected})
    grid[x_col] = xs * GRID_N
    grid[y_col] = [y for y in ys for _ in range(GRID_N)]

    proba_flat = logit.predict_proba(scaler.transform(grid))[:, 1].tolist()
    tree_flat = tree.predict(grid).tolist()
    proba_z = [proba_flat[i * GRID_N : (i + 1) * GRID_N] for i in range(GRID_N)]
    tree_z = [tree_flat[i * GRID_N : (i + 1) * GRID_N] for i in range(GRID_N)]

    fig = go.Figure()

    # 의사결정트리가 나눈 칸 (옅은 색)
    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys,
            z=tree_z,
            zmin=0,
            zmax=1,
            colorscale=[[0, R_NO], [1, R_YES]],
            showscale=False,
            hoverinfo="skip",
            zsmooth=False,
        )
    )

    # 로지스틱 회귀의 0.5 경계선
    p_min, p_max = min(proba_flat), max(proba_flat)
    line_inside = p_min < 0.5 < p_max
    if line_inside:
        fig.add_trace(
            go.Contour(
                x=xs,
                y=ys,
                z=proba_z,
                contours=dict(start=0.5, end=0.5, size=0.1, coloring="lines"),
                colorscale=[[0, "#212121"], [1, "#212121"]],
                line=dict(width=3),
                showscale=False,
                hoverinfo="skip",
            )
        )

    # 테스트 데이터 점 (실제 뇌졸중 여부로 색을 나눈다)
    jitter = random.Random(42)

    def spread(values, col):
        if col in BINARY:  # 0·1 두 값뿐이라 겹치지 않게 살짝 흩어서 찍는다
            return [v + jitter.uniform(-0.08, 0.08) for v in values]
        return list(values)

    for label, name, color, size in [
        (0, "실제 뇌졸중 아님", C_NO, 6),
        (1, "실제 뇌졸중", C_YES, 9),
    ]:
        part = X_test[(y_test == label).values]
        fig.add_trace(
            go.Scatter(
                x=spread(part[x_col].tolist(), x_col),
                y=spread(part[y_col].tolist(), y_col),
                mode="markers",
                name=name,
                marker=dict(
                    color=color,
                    size=size,
                    opacity=0.65,
                    line=dict(width=0.5, color="white"),
                ),
            )
        )

    # 범례용 안내 항목 (선과 칸 색)
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="로지스틱 회귀의 0.5 경계선",
            line=dict(color="#212121", width=3),
        )
    )
    for color, name in [
        (R_NO, "의사결정트리가 '아님'이라고 답하는 칸"),
        (R_YES, "의사결정트리가 '뇌졸중'이라고 답하는 칸"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=name,
                marker=dict(symbol="square", size=14, color=color, line=dict(width=1, color="#9E9E9E")),
            )
        )

    fig.update_layout(
        title=f"{KOR[x_col]} × {KOR[y_col]} (테스트 데이터)",
        xaxis=dict(title=KOR[x_col], range=[x_lo, x_hi]),
        yaxis=dict(title=KOR[y_col], range=[y_lo, y_hi]),
        legend=dict(orientation="h", yanchor="top", y=-0.18),
        margin=dict(l=10, r=10, t=50, b=10),
        height=620,
    )
    st.plotly_chart(fig, use_container_width=True)

    others = [c for c in selected if c not in (x_col, y_col)]
    if others:
        fixed = ", ".join(
            f"{KOR[c]} {int(round(medians[c]))}" if c in BINARY else f"{KOR[c]} {medians[c]:.1f}"
            for c in others
        )
        st.write(f"가로축·세로축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: **{fixed}**")
    else:
        st.write("고른 속성이 두 개뿐이라 따로 세워 둔 속성은 없어요.")

    if not line_inside:
        st.info(
            "로지스틱 회귀의 0.5 경계선은 이 그림 밖에 있어요. "
            f"그림 안에서 뇌졸중일 확률은 {p_min * 100:.1f}% ~ {p_max * 100:.1f}%라서 0.5에 닿지 않아요."
        )
    if x_col in BINARY or y_col in BINARY:
        st.caption("고혈압·심장병은 0과 1 두 값뿐이라, 점이 겹치지 않게 살짝 흩어서 찍었어요. 칸과 선 계산에는 영향이 없어요.")

# ---------------------------------------------------------------
# 6) 의사결정트리가 던진 질문 (가지 그림)
# ---------------------------------------------------------------
st.subheader("🌳 의사결정트리가 던진 질문")


def build_tree_dot(clf, X_fit: pd.DataFrame, y_fit: pd.Series, cols: list):
    t = clf.tree_
    path = pd.DataFrame(clf.decision_path(X_fit).toarray())
    n_per = path.sum().astype(int).tolist()  # 마디마다 그 자리에 온 훈련용 사람 수
    s_per = path[(y_fit == 1).values].sum().astype(int).tolist()  # 그중 뇌졸중인 사람 수

    lines = [
        "digraph Tree {",
        '  graph [ranksep=0.6, nodesep=0.3];',
        '  node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=12, margin="0.2,0.1", color="#9E9E9E"];',
        '  edge [fontname="sans-serif", fontsize=12];',
    ]
    leaf_answers = []
    used = set()

    for i in range(t.node_count):
        n, s = n_per[i], s_per[i]
        info = f"훈련용 {n}명\\n뇌졸중 {s}명 ({s / n * 100:.1f}%)"
        left, right = int(t.children_left[i]), int(t.children_right[i])

        if left == -1:  # 더 묻지 않고 답을 내는 마디
            answer = int(clf.classes_[t.value[i][0].argmax()])
            leaf_answers.append(answer)
            fill = R_YES if answer == 1 else R_NO
            border = C_YES if answer == 1 else C_NO
            word = "뇌졸중" if answer == 1 else "아님"
            lines.append(f'  n{i} [label="답: {word}\\n{info}", fillcolor="{fill}", color="{border}"];')
        else:
            col = cols[int(t.feature[i])]
            used.add(col)
            if col in BINARY:
                question = f"{KOR[col]}이 있나요?"
                yes_child, no_child = right, left
            else:
                question = f"{KOR[col]} ≤ {t.threshold[i]:.1f} ?"
                yes_child, no_child = left, right
            lines.append(f'  n{i} [label="{question}\\n{info}", fillcolor="#FAFAFA"];')
            lines.append(f'  n{i} -> n{yes_child} [label="예"];')
            lines.append(f'  n{i} -> n{no_child} [label="아니요"];')

    lines.append("}")
    return "\n".join(lines), leaf_answers, used


dot, leaf_answers, used_cols = build_tree_dot(tree, X_train, y_train, selected)
st.graphviz_chart(dot, use_container_width=True)

n_leaf = len(leaf_answers)
n_no = sum(1 for a in leaf_answers if a == 0)
st.markdown(f"답을 내는 마디는 모두 **{n_leaf}칸**이고, 그중 **{n_no}칸**이 '아님'이라고 답해요.")

asked = [KOR[c] for c in selected if c in used_cols]
if asked:
    st.markdown(f"고른 속성 {len(selected)}개 가운데 이 나무가 실제로 물은 것은 **{', '.join(asked)}**예요.")
else:
    st.markdown("이 나무는 아무것도 묻지 않고 바로 답을 내요.")
