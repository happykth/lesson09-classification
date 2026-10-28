import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"

st.set_page_config(page_title=f"성능 높이기 · {APP_TITLE}", page_icon="🚀", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

# pages 폴더 안의 파일이라, 프로젝트 맨 위 폴더(main.py가 있는 곳)의 내_기록 폴더를 쓴다
SAVE_DIR = Path(__file__).resolve().parent.parent / "내_기록"
SAVE_FILE = SAVE_DIR / "성능_높이기_기록.json"

FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
KOR = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
BINARY = {"hypertension", "heart_disease"}  # 0 또는 1 두 값뿐인 속성

# 화면에 모델 이름을 적을 때는 항상 이 이름을 그대로 쓴다
NAME_LOGIT = "로지스틱 회귀(확률로 답하는 모델)"
NAME_TREE = "의사결정트리(질문으로 답하는 모델)"

NOT_AVAILABLE = "계산할 수 없음"
CAP = 500  # 상담 정원
GOAL_CAP = f"정원 {CAP}명 안에서 찾아낸 환자"
GOALS = ["정확도", "재현율", "정밀도", "F1", GOAL_CAP]
BMI_DELETE = "지운다"
BMI_FILL = "훈련용 중앙값으로 채운다"
DEFAULT_DEPTH = 3
DEFAULT_THRESHOLD = 0.50

C_NO, C_YES = "#3F7CAC", "#D84315"  # 점 색: 아님 / 뇌졸중
C_BEFORE, C_AFTER = "#424242", "#2E7D32"  # 경계선·평면 색: 가중치 맞추기 전 / 후
GRID_N = 150


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
    .acc-name { font-size: 0.95rem; color: #8D5A1F; font-weight: 600; }
    .acc-cap { font-size: 0.85rem; color: #8D5A1F; margin-top: 4px; }
    .acc-big { font-size: 2.6rem; font-weight: 800; color: #BF360C; line-height: 1.15; }
    .acc-big.small { font-size: 1.5rem; padding: 0.55rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# 계산 도우미 (채점표 페이지와 같은 방식)
# ---------------------------------------------------------------
def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """뇌졸중(1)을 양성으로 두고 TP·FP·FN·TN 사람 수를 센다."""
    return {
        "TP": int(((y_true == 1) & (y_pred == 1)).sum()),
        "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
        "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
        "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


def metrics(cm: dict) -> dict:
    """네 칸에서 정확도·재현율·정밀도·F1을 구한다. 구할 수 없으면 None."""
    tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    total = tp + fp + fn + tn
    actual_pos = tp + fn  # 재현율의 분모: 실제 뇌졸중인 사람 수
    pred_pos = tp + fp  # 정밀도의 분모: 뇌졸중이라 예측한 사람 수
    accuracy = (tp + tn) / total if total else None
    recall = tp / actual_pos if actual_pos else None
    precision = tp / pred_pos if pred_pos else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"정확도": accuracy, "재현율": recall, "정밀도": precision, "F1": f1}


def f4(value) -> str:
    return NOT_AVAILABLE if value is None else f"{value:.4f}"


def make_logit(weight):
    return LogisticRegression(max_iter=1000, random_state=42, class_weight=weight)


def make_tree(depth: int, weight):
    return DecisionTreeClassifier(
        max_depth=depth, min_samples_leaf=5, random_state=42, class_weight=weight
    )


def to_list_with_none(array: np.ndarray) -> list:
    """NaN을 None으로 바꿔 그림 데이터로 넘긴다."""
    return [[None if np.isnan(v) else float(v) for v in row] for row in array]


# ---------------------------------------------------------------
# 최고 기록 저장·불러오기 (내_기록 폴더의 json 파일)
# ---------------------------------------------------------------
def load_records() -> list:
    if SAVE_FILE.exists():
        try:
            data = json.loads(SAVE_FILE.read_text(encoding="utf-8"))
            return list(data.get("기록", []))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def standings(records: list, goal: str) -> list:
    """같은 목표에서 팀마다 최고 기록 하나씩을 뽑아 순위 순으로 늘어놓는다.
    점수가 같으면 그 점수를 먼저 올린 팀(seq가 작은 쪽)이 앞선다."""
    best = {}
    for r in records:
        if r["goal"] != goal or not r["eligible"] or r["score"] is None:
            continue
        tid = (r["class"], r["team"])
        cur = best.get(tid)
        if cur is None or (r["score"], -r["seq"]) > (cur["score"], -cur["seq"]):
            best[tid] = r
    return sorted(best.values(), key=lambda r: (-r["score"], r["seq"]))


def save_records(records: list) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_per_goal = {}
    for g in GOALS:
        ranked = standings(records, g)
        best_per_goal[g] = ranked[0] if ranked else None  # 목표마다 최고 기록을 따로 저장
    payload = {"기록": records, "최고_기록": best_per_goal}
    fd, tmp_path = tempfile.mkstemp(dir=SAVE_DIR, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, SAVE_FILE)  # 쓰다 끊겨도 파일이 깨지지 않게 통째로 바꾼다


def setting_text(rec: dict) -> str:
    feats = ", ".join(KOR[c] for c in rec["features"])
    return (
        f"{rec['model']} · 속성 {feats} · 빈 값 {rec['bmi']} · "
        f"질문 {rec['depth']}번 · 기준 {rec['threshold']:.2f}"
    )


def score_text(goal: str, rec: dict) -> str:
    if goal == GOAL_CAP:
        return f"찾아낸 환자 {int(rec['score'])}명(안내 인원 {rec['guide']}명)"
    return f"{rec['score']:.4f}"


def stat_card(label: str, value: str, sub: str = "") -> str:
    small = " small" if len(value) > 7 else ""
    return (
        f'<div class="acc-card"><div class="acc-name">{label}</div>'
        f'<div class="acc-big{small}">{value}</div><div class="acc-cap">{sub}</div></div>'
    )


# ---------------------------------------------------------------
# 화면 맨 위: 제목과 입력 속성 고르기
# ---------------------------------------------------------------
st.title("🚀 성능 높이기")
st.caption(
    "뇌졸중을 겪은 사람이 아주 드물어요. 뇌졸중과 아님, 양쪽의 가중치를 같게 맞추면 결과가 어떻게 달라지는지 "
    "살펴보고, 설정을 바꿔 가며 챌린지에 도전해요."
)

try:
    raw = load_data()
except Exception as error:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n({error})")
    st.stop()

chosen = st.multiselect(
    "입력으로 사용할 속성",
    options=FEATURES,
    default=["age", "avg_glucose_level", "hypertension", "heart_disease"],
    format_func=lambda c: KOR[c],
)
selected = [f for f in FEATURES if f in chosen]

if len(selected) < 2:
    st.info("입력으로 사용할 속성을 두 개 이상 골라 주세요. 두 개 이상 고르면 모델이 만들어져요.")
    st.stop()

# ---------------------------------------------------------------
# 1) 전체 데이터를 '채점표' 페이지와 같은 방법으로 나누고, 가중치 전·후 모델 학습하기
#    번호 순으로 정렬 → 열 명씩 묶기 → 각 묶음의 앞 세 명이 테스트용, 나머지 일곱 명이 학습용
# ---------------------------------------------------------------
df = raw.sort_values("id").reset_index(drop=True)
is_test_all = (df.index % 10) < 3

train_df, test_df = df[~is_test_all], df[is_test_all]
X_train = train_df[selected].copy()
X_test = test_df[selected].copy()
y_train, y_test = train_df["stroke"], test_df["stroke"]

if "bmi" in selected:
    bmi_median_all = X_train["bmi"].median()  # 훈련용의 중앙값
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median_all)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median_all)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
y_test_arr = y_test.to_numpy()

VARIANTS = {"그대로": None, "맞춤": "balanced"}
top_models = {
    NAME_LOGIT: {k: make_logit(w).fit(X_train_s, y_train) for k, w in VARIANTS.items()},
    NAME_TREE: {k: make_tree(DEFAULT_DEPTH, w).fit(X_train_s, y_train) for k, w in VARIANTS.items()},
}

# ---------------------------------------------------------------
# 2) 가중치를 맞추기 전·후 비교 (테스트 데이터)
# ---------------------------------------------------------------
st.subheader("⚖️ 가중치를 맞추기 전과 맞춘 뒤")
st.write(
    f"테스트용 **{len(y_test):,}명** 가운데 뇌졸중인 사람은 **{int((y_test == 1).sum()):,}명**이에요. "
    "가중치를 맞추면 드문 쪽인 뇌졸중 한 명이 더 무겁게 반영돼요."
)

cols = st.columns(2)
for col, name in zip(cols, (NAME_LOGIT, NAME_TREE)):
    results = {}
    for key in ("그대로", "맞춤"):
        pred = top_models[name][key].predict(X_test_s)
        cm = confusion(y_test_arr, pred)
        results[key] = (cm, metrics(cm))
    (cm0, m0), (cm1, m1) = results["그대로"], results["맞춤"]

    rows = [
        ("찾아낸 환자 (TP)", f"{cm0['TP']:,}명", f"{cm1['TP']:,}명"),
        ("놓친 환자 (FN)", f"{cm0['FN']:,}명", f"{cm1['FN']:,}명"),
        ("헛짚은 사람 (FP)", f"{cm0['FP']:,}명", f"{cm1['FP']:,}명"),
        ("아님을 맞힘 (TN)", f"{cm0['TN']:,}명", f"{cm1['TN']:,}명"),
    ] + [(k, f4(m0[k]), f4(m1[k])) for k in ("정확도", "재현율", "정밀도", "F1")]

    with col:
        st.markdown(f"**{name}**")
        st.dataframe(
            pd.DataFrame(rows, columns=["항목", "가중치 그대로 (전)", "가중치 맞춤 (후)"]),
            hide_index=True,
            use_container_width=True,
        )
        st.write(
            f"찾아낸 환자 {cm0['TP']:,}명 → {cm1['TP']:,}명 · "
            f"놓친 환자 {cm0['FN']:,}명 → {cm1['FN']:,}명 · "
            f"헛짚은 사람 {cm0['FP']:,}명 → {cm1['FP']:,}명"
        )

# ---------------------------------------------------------------
# 3) 두 속성으로 본 0.5 경계선 (가중치 전·후, 확률로 답하는 모델)
# ---------------------------------------------------------------
st.subheader("🗺️ 두 속성으로 본 0.5 경계선")

s1, s2 = st.columns(2)
x_col = s1.selectbox("가로축", selected, index=0, format_func=lambda c: KOR[c], key="axis_x")
y_col = s2.selectbox("세로축", selected, index=1, format_func=lambda c: KOR[c], key="axis_y")

medians = X_test[selected].median()  # 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세운다


def axis_range(col: str):
    if col in BINARY:
        return -0.4, 1.4
    lo, hi = float(X_test[col].min()), float(X_test[col].max())
    pad = (hi - lo) * 0.05 or 1.0
    return lo - pad, hi + pad


def fixed_text(skip: list) -> str:
    parts = []
    for c in selected:
        if c in skip:
            continue
        value = float(medians[c])
        if c in BINARY:
            parts.append(f"{KOR[c]} {int(round(value))}({'있음' if round(value) == 1 else '없음'})")
        else:
            parts.append(f"{KOR[c]} {value:.1f}")
    return ", ".join(parts)


if x_col == y_col:
    st.warning("가로축과 세로축은 서로 다른 속성으로 골라 주세요.")
else:
    x_lo, x_hi = axis_range(x_col)
    y_lo, y_hi = axis_range(y_col)
    xs = [x_lo + (x_hi - x_lo) * i / (GRID_N - 1) for i in range(GRID_N)]
    ys = [y_lo + (y_hi - y_lo) * i / (GRID_N - 1) for i in range(GRID_N)]

    grid = pd.DataFrame({c: [float(medians[c])] * (GRID_N * GRID_N) for c in selected})
    grid[x_col] = xs * GRID_N  # 가로 방향이 먼저 바뀐다
    grid[y_col] = [y for y in ys for _ in range(GRID_N)]
    grid_s = scaler.transform(grid)

    fig = go.Figure()

    # 테스트 데이터 점 (실제 뇌졸중 여부로 색을 나눈다)
    hover = f"{KOR[x_col]}: %{{x}}<br>{KOR[y_col]}: %{{y}}<extra></extra>"
    for label, name, color, size, opacity in [
        (0, "실제 뇌졸중 아님", C_NO, 6, 0.5),
        (1, "실제 뇌졸중", C_YES, 9, 0.9),
    ]:
        part = X_test[(y_test == label).values]
        fig.add_trace(
            go.Scatter(
                x=part[x_col].tolist(),
                y=part[y_col].tolist(),
                mode="markers",
                name=name,
                hovertemplate=hover,
                marker=dict(color=color, size=size, opacity=opacity, line=dict(width=0.5, color="white")),
            )
        )

    notes = []
    for key, color, dash, label in [
        ("그대로", C_BEFORE, "dash", "가중치를 맞추기 전"),
        ("맞춤", C_AFTER, "solid", "가중치를 맞춘 뒤"),
    ]:
        model = top_models[NAME_LOGIT][key]
        proba = model.predict_proba(grid_s)[:, 1]
        p_min, p_max = float(proba.min()), float(proba.max())
        line_name = f"{label}의 {NAME_LOGIT} 0.5 경계선"
        if p_min < 0.5 < p_max:
            z = proba.reshape(GRID_N, GRID_N).tolist()
            fig.add_trace(
                go.Contour(
                    x=xs,
                    y=ys,
                    z=z,
                    contours=dict(start=0.5, end=0.5, size=0.1, coloring="lines"),
                    colorscale=[[0, color], [1, color]],
                    line=dict(width=3, dash=dash),
                    showscale=False,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        else:
            notes.append(
                f"{label}의 {NAME_LOGIT} 0.5 경계선은 이 그림 밖에 있어요. "
                f"그림 안에서 뇌졸중일 확률은 {p_min:.3f} ~ {p_max:.3f}라서 0.5에 닿지 않아요."
            )
        fig.add_trace(  # 범례용 안내 항목
            go.Scatter(
                x=[None], y=[None], mode="lines", name=line_name,
                line=dict(color=color, width=3, dash=dash),
            )
        )

    fig.update_layout(
        title=f"{KOR[x_col]} × {KOR[y_col]} (테스트 데이터 {len(X_test):,}명)",
        xaxis=dict(title=KOR[x_col], range=[x_lo, x_hi]),
        yaxis=dict(title=KOR[y_col], range=[y_lo, y_hi]),
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0),
        margin=dict(l=10, r=10, t=50, b=10),
        height=680,
    )
    st.plotly_chart(fig, use_container_width=True)

    others_text = fixed_text([x_col, y_col])
    if others_text:
        st.write(f"가로축과 세로축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {others_text}")
    for note in notes:
        st.write(note)

# ---------------------------------------------------------------
# 4) 세 속성으로 본 3차원 그림 (0.5 평면, 마우스로 돌려 볼 수 있다)
# ---------------------------------------------------------------
if len(selected) >= 3:
    st.subheader("🧊 세 속성으로 본 0.5 평면 (3차원)")
    t1, t2, t3 = st.columns(3)
    ax_x = t1.selectbox("가로축", selected, index=0, format_func=lambda c: KOR[c], key="axis3_x")
    ax_y = t2.selectbox("세로축", selected, index=1, format_func=lambda c: KOR[c], key="axis3_y")
    ax_z = t3.selectbox("높이축", selected, index=2, format_func=lambda c: KOR[c], key="axis3_z")
    axes3 = [ax_x, ax_y, ax_z]

    if len(set(axes3)) < 3:
        st.warning("세 축은 서로 다른 속성으로 골라 주세요.")
    else:
        ranges3 = {c: axis_range(c) for c in axes3}

        def plane_for(model, n: int = 40):
            """세 축을 가로지르는 0.5 평면. 로지스틱 회귀는 확률 0.5가 판별식 0인 평면이다."""
            base = pd.DataFrame([{c: float(medians[c]) for c in selected}])
            for a in axes3:
                base[a] = 0.0
            f0 = float(model.decision_function(scaler.transform(base))[0])
            coefs = []
            for a in axes3:
                row = base.copy()
                row[a] = 1.0
                coefs.append(float(model.decision_function(scaler.transform(row))[0]) - f0)
            k = int(np.argmax(np.abs(coefs)))
            if abs(coefs[k]) < 1e-12:
                return "none", None
            i, j = [m for m in range(3) if m != k]
            g1 = np.linspace(*ranges3[axes3[i]], n)
            g2 = np.linspace(*ranges3[axes3[j]], n)
            G1, G2 = np.meshgrid(g1, g2)
            Zk = -(f0 + coefs[i] * G1 + coefs[j] * G2) / coefs[k]
            lo, hi = ranges3[axes3[k]]
            Zk = np.where((Zk >= lo) & (Zk <= hi), Zk, np.nan)
            if np.isnan(Zk).all():
                return "outside", None
            arrays = [None, None, None]
            arrays[i], arrays[j], arrays[k] = G1, G2, Zk
            return "ok", arrays

        fig3 = go.Figure()
        for label, name, color, size, opacity in [
            (0, "실제 뇌졸중 아님", C_NO, 3, 0.35),
            (1, "실제 뇌졸중", C_YES, 5, 0.95),
        ]:
            part = X_test[(y_test == label).values]
            fig3.add_trace(
                go.Scatter3d(
                    x=part[ax_x].tolist(),
                    y=part[ax_y].tolist(),
                    z=part[ax_z].tolist(),
                    mode="markers",
                    name=name,
                    marker=dict(color=color, size=size, opacity=opacity),
                )
            )

        notes3 = []
        for key, color, label in [("그대로", C_BEFORE, "가중치를 맞추기 전"), ("맞춤", C_AFTER, "가중치를 맞춘 뒤")]:
            status, arrays = plane_for(top_models[NAME_LOGIT][key])
            plane_name = f"{label}의 {NAME_LOGIT} 0.5 평면"
            if status == "ok":
                fig3.add_trace(
                    go.Surface(
                        x=to_list_with_none(arrays[0]),
                        y=to_list_with_none(arrays[1]),
                        z=to_list_with_none(arrays[2]),
                        colorscale=[[0, color], [1, color]],
                        showscale=False,
                        opacity=0.45,
                        hoverinfo="skip",
                    )
                )
                visible = float(np.isfinite(arrays[0] * arrays[1] * arrays[2]).mean())
                if visible < 0.05:
                    notes3.append(
                        f"{label}의 {NAME_LOGIT} 0.5 평면은 거의 다 이 그림 밖에 있고, 아주 작은 조각만 안에 걸쳐 있어요."
                    )
            elif status == "outside":
                notes3.append(f"{label}의 {NAME_LOGIT} 0.5 평면은 이 그림 밖에 있어요.")
            else:
                notes3.append(f"{label}의 {NAME_LOGIT}은 고른 세 속성으로는 0.5 평면이 만들어지지 않아요.")
            fig3.add_trace(  # 범례용 안내 항목
                go.Scatter3d(
                    x=[None], y=[None], z=[None], mode="markers", name=plane_name,
                    marker=dict(size=7, color=color, symbol="square"),
                )
            )

        fig3.update_layout(
            title=f"{KOR[ax_x]} × {KOR[ax_y]} × {KOR[ax_z]} (테스트 데이터 {len(X_test):,}명)",
            scene=dict(
                xaxis=dict(title=KOR[ax_x], range=list(ranges3[ax_x])),
                yaxis=dict(title=KOR[ax_y], range=list(ranges3[ax_y])),
                zaxis=dict(title=KOR[ax_z], range=list(ranges3[ax_z])),
            ),
            legend=dict(orientation="h", yanchor="top", y=-0.02, x=0),
            margin=dict(l=0, r=0, t=50, b=0),
            height=720,
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("마우스로 끌면 돌려 볼 수 있고, 휠로 크게 작게 볼 수 있어요.")

        fixed3 = fixed_text(axes3)
        if fixed3:
            st.write(f"세 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {fixed3}")
        for note in notes3:
            st.write(note)

# ---------------------------------------------------------------
# 5) 챌린지
# ---------------------------------------------------------------
st.divider()
st.header("🏆 챌린지")
st.write(
    f"테스트 데이터로 채점해요. 두 모델 모두 양쪽의 가중치를 같게 맞춰 학습해요. "
    f"마지막 목표는 안내 인원이 상담 정원 {CAP}명 이하인 채로 찾아낸 환자를 가장 많이 만드는 거예요."
)

# 5-1) 목표 지표, 빈 값 처리, 설정 고르기
goal = st.selectbox("가장 높여 볼 지표", GOALS, index=0)

bmi_mode = st.radio(
    "체질량지수(bmi)가 비어 있는 사람은 어떻게 할까요?",
    [BMI_DELETE, BMI_FILL],
    index=0,
    horizontal=True,
)
bmi_na = df["bmi"].isna()
n_bmi_na = int(bmi_na.sum())
n_bmi_na_stroke = int(df.loc[bmi_na, "stroke"].sum())
delete_bmi = bmi_mode == BMI_DELETE

# 지우면 먼저 지운 뒤, 같은 방법(번호 순 정렬 → 열 명씩 묶어 앞 세 명이 테스트용)으로 나눈다
chal_df = df[~bmi_na].reset_index(drop=True) if delete_bmi else df
chal_is_test = (chal_df.index % 10) < 3
chal_train, chal_test = chal_df[~chal_is_test], chal_df[chal_is_test]
CX_train = chal_train[selected].copy()
CX_test = chal_test[selected].copy()
cy_train, cy_test = chal_train["stroke"], chal_test["stroke"]

if delete_bmi:
    st.write(f"체질량지수가 빈 사람 **{n_bmi_na:,}명**이 지워졌고, 그 안에 뇌졸중 환자가 **{n_bmi_na_stroke:,}명** 있었어요.")
else:
    chal_median = CX_train["bmi"].median() if "bmi" in selected else float(train_df["bmi"].median())
    if "bmi" in selected:
        CX_train["bmi"] = CX_train["bmi"].fillna(chal_median)
        CX_test["bmi"] = CX_test["bmi"].fillna(chal_median)
        st.write(f"체질량지수의 빈 값 {n_bmi_na:,}칸을 훈련용 중앙값 {chal_median:.1f}으로 채웠어요.")
    else:
        st.write(
            f"체질량지수가 빈 사람 {n_bmi_na:,}명은 그대로 두었어요. 체질량지수를 입력으로 고르지 않아 모델에는 영향이 없어요."
        )

p1, p2 = st.columns(2)
depth = p1.select_slider(
    "질문을 몇 번까지 던질까요? (의사결정트리)",
    options=list(range(1, 11)),
    value=DEFAULT_DEPTH,
    format_func=lambda v: f"{v}번",
)
threshold = p2.select_slider(
    "확률이 얼마 이상이면 뇌졸중이라고 답할까요? (두 모델 모두)",
    options=[round(0.05 * i, 2) for i in range(1, 20)],
    value=0.60,
    format_func=lambda v: f"{v:.2f}",
)

cscaler = StandardScaler().fit(CX_train)
CX_train_s = cscaler.transform(CX_train)
CX_test_s = cscaler.transform(CX_test)
cy_train_arr, cy_test_arr = cy_train.to_numpy(), cy_test.to_numpy()


def evaluate(model, X_s, y_arr, t: float):
    """확률이 t 이상이면 뇌졸중이라고 답한다. 의사결정트리의 확률은 도착한 칸의 뇌졸중 비율이다."""
    pred = (model.predict_proba(X_s)[:, 1] >= t).astype(int)
    cm = confusion(y_arr, pred)
    return cm, metrics(cm), pred


my_models = {
    NAME_LOGIT: make_logit("balanced").fit(CX_train_s, cy_train),
    NAME_TREE: make_tree(depth, "balanced").fit(CX_train_s, cy_train),
}
default_models = {
    NAME_LOGIT: make_logit(None).fit(CX_train_s, cy_train),
    NAME_TREE: make_tree(DEFAULT_DEPTH, None).fit(CX_train_s, cy_train),
}

# 5-2) 지금 설정의 성적 (테스트 데이터)
my_test = {name: evaluate(model, CX_test_s, cy_test_arr, threshold) for name, model in my_models.items()}


def rank_key(name: str):
    """두 모델 가운데 어느 쪽이 앞서는지 비교하는 값 (클수록 좋다)."""
    cm, m, _ = my_test[name]
    if goal == GOAL_CAP:
        guide = cm["TP"] + cm["FP"]
        return (1, cm["TP"]) if guide <= CAP else (0, -guide)
    value = m[goal]
    return (0, 0.0) if value is None else (1, value)


winner = max((NAME_LOGIT, NAME_TREE), key=rank_key)  # 같으면 앞에 놓인 모델
w_cm, w_m, _ = my_test[winner]
w_found = w_cm["TP"]
w_guide = w_cm["TP"] + w_cm["FP"]
w_in_cap = w_guide <= CAP

if goal == GOAL_CAP:
    my_score = float(w_found) if w_in_cap else None
    goal_value_text = f"{w_found:,}명"
    goal_label = "정원 안에서 찾아낸 환자"
else:
    my_score = None if w_m[goal] is None else round(w_m[goal], 4)
    goal_value_text = f4(w_m[goal])
    goal_label = f"고른 지표 · {goal}"

st.subheader("📈 지금 설정의 성적")
if goal == GOAL_CAP:
    compare_text = " · ".join(
        f"{n} 안내 {my_test[n][0]['TP'] + my_test[n][0]['FP']:,}명 / 찾아낸 환자 {my_test[n][0]['TP']:,}명"
        for n in (NAME_LOGIT, NAME_TREE)
    )
else:
    compare_text = " · ".join(f"{n} {f4(my_test[n][1][goal])}" for n in (NAME_LOGIT, NAME_TREE))
st.write(f"두 모델 가운데 **{winner}**이 앞서요. ({compare_text})")

k1, k2, k3, k4 = st.columns(4)
k1.markdown(stat_card(goal_label, goal_value_text, "테스트 데이터 기준"), unsafe_allow_html=True)
k2.markdown(
    stat_card("찾아낸 환자 수", f"{w_found:,}명", f"테스트용 뇌졸중 {int((cy_test == 1).sum()):,}명 중"),
    unsafe_allow_html=True,
)
k3.markdown(
    stat_card("안내 인원", f"{w_guide:,}명", "뇌졸중이라고 답한 사람 수"),
    unsafe_allow_html=True,
)
k4.markdown(
    stat_card("정원 안에 드는지", "예" if w_in_cap else "아니요", f"정원 {CAP}명 · 안내 인원 {w_guide:,}명"),
    unsafe_allow_html=True,
)

# 5-3) 최고 기록 한 줄 (아래 기록 올리기를 처리한 뒤 채운다)
best_slot = st.empty()

# 5-4) 기록 올리기
st.subheader("📮 기록 올리기")
q1, q2 = st.columns(2)
team_input = q1.text_input("팀명")
class_input = q2.text_input("반")

if st.button("고른 목표로 기록 올리기"):
    team, cls = team_input.strip(), class_input.strip()
    if not team or not cls:
        st.warning("팀명과 반을 모두 적어 주세요.")
    elif goal != GOAL_CAP and my_score is None:
        st.warning(f"지금 설정은 {goal}을(를) 계산할 수 없어서 올릴 수 없어요. 설정을 바꿔 보세요.")
    else:
        records = load_records()
        setting_key = json.dumps(
            {"features": selected, "bmi": bmi_mode, "depth": depth, "threshold": round(threshold, 2)},
            ensure_ascii=False,
            sort_keys=True,
        )
        if any(
            r["goal"] == goal and r["key"] == setting_key and r["team"] == team and r["class"] == cls
            for r in records
        ):
            st.info("이미 올린 설정이에요. 같은 목표에서 같은 설정은 한 번만 올릴 수 있어요.")
        else:
            new_rec = {
                "seq": max([r["seq"] for r in records], default=0) + 1,
                "time": datetime.now().isoformat(timespec="seconds"),
                "team": team,
                "class": cls,
                "goal": goal,
                "key": setting_key,
                "features": selected,
                "bmi": bmi_mode,
                "depth": depth,
                "threshold": round(threshold, 2),
                "model": winner,
                "score": my_score,
                "found": w_found,
                "guide": w_guide,
                "eligible": bool(w_in_cap) if goal == GOAL_CAP else True,
            }
            records.append(new_rec)
            save_records(records)

            if not new_rec["eligible"]:
                st.warning(
                    f"기록은 올렸지만, 안내 인원이 {w_guide:,}명으로 정원 {CAP}명을 넘어서 순위에는 들어가지 않아요."
                )
            else:
                ranked = standings(records, goal)
                tid = (cls, team)
                idx = next(i for i, r in enumerate(ranked) if (r["class"], r["team"]) == tid)
                n_teams = len(ranked)
                rank = idx + 1
                top_percent = rank / n_teams * 100
                class_ranked = [r for r in ranked if r["class"] == cls]
                class_rank = next(i for i, r in enumerate(class_ranked) if r["team"] == team) + 1

                if goal == GOAL_CAP:
                    head = f"정원 안에서 찾아낸 환자 {w_found:,}명 · 안내 인원 {w_guide:,}명!"
                else:
                    head = f"{goal} {my_score:.4f} · 찾아낸 환자 {w_found:,}명!"
                st.success(f"{head} 같은 목표 {n_teams}팀 가운데 {rank}위 · 상위 {top_percent:.1f}%")
                st.write(f"우리 반({cls}) {len(class_ranked)}팀 가운데 {class_rank}위예요.")
                team_best = ranked[idx]
                if team_best["seq"] != new_rec["seq"]:
                    st.caption(f"순위는 우리 팀의 최고 기록({score_text(goal, team_best)})으로 매겨요.")

# 최고 기록 한 줄 (방금 올린 기록까지 반영)
ranked_now = standings(load_records(), goal)
if ranked_now:
    best = ranked_now[0]
    best_slot.write(
        f"🏆 '{goal}' 최고 기록 **{score_text(goal, best)}** — {best['team']}({best['class']}) · "
        f"그때의 설정: {setting_text(best)}"
    )
else:
    best_slot.write(f"🏆 '{goal}' 목표에는 아직 올라온 기록이 없어요. 첫 기록을 올려 보세요!")

# 5-5) 기본 설정과 내 설정 비교표
st.subheader("🔬 기본 설정과 내 설정")
default_label = f"기본 설정(가중치 그대로 · 질문 {DEFAULT_DEPTH}번 · 기준 {DEFAULT_THRESHOLD:.2f})"
my_label = f"내 설정(가중치 맞춤 · 질문 {depth}번 · 기준 {threshold:.2f})"

table_rows = []
for name in (NAME_LOGIT, NAME_TREE):
    for setting_label, models, t in (
        (default_label, default_models, DEFAULT_THRESHOLD),
        (my_label, my_models, threshold),
    ):
        for data_label, X_s, y_arr in (
            ("훈련용", CX_train_s, cy_train_arr),
            ("테스트용", CX_test_s, cy_test_arr),
        ):
            cm, m, _ = evaluate(models[name], X_s, y_arr, t)
            table_rows.append(
                {
                    "모델": name,
                    "설정": setting_label,
                    "데이터": data_label,
                    "정확도": f4(m["정확도"]),
                    "재현율": f4(m["재현율"]),
                    "정밀도": f4(m["정밀도"]),
                    "F1": f4(m["F1"]),
                    "찾아낸 환자 수": f"{cm['TP']:,}명",
                    "헛짚은 사람 수": f"{cm['FP']:,}명",
                }
            )
st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

changes = ["가중치를 같게 맞췄어요"]
changes.append(
    f"질문 {DEFAULT_DEPTH}번 → {depth}번(의사결정트리)" if depth != DEFAULT_DEPTH else f"질문 {DEFAULT_DEPTH}번 그대로"
)
changes.append(
    f"기준 {DEFAULT_THRESHOLD:.2f} → {threshold:.2f}" if abs(threshold - DEFAULT_THRESHOLD) > 1e-9 else f"기준 {DEFAULT_THRESHOLD:.2f} 그대로"
)
st.write("기본 설정에서 달라진 것: " + " · ".join(changes))

# 5-6) 가중치를 맞춘 로지스틱 회귀가 놓친 환자 목록 (내 설정, 테스트 데이터)
logit_pred = my_test[NAME_LOGIT][2]
missed_mask = (cy_test_arr == 1) & (logit_pred == 0)
missed = chal_test.loc[missed_mask, ["id", "age", "hypertension", "heart_disease"]].copy()
missed["hypertension"] = missed["hypertension"].map({0: "없음", 1: "있음"})
missed["heart_disease"] = missed["heart_disease"].map({0: "없음", 1: "있음"})
missed = (
    missed.rename(
        columns={"id": "번호", "age": "나이", "hypertension": "고혈압 여부", "heart_disease": "심장병 여부"}
    )
    .sort_values(["나이", "번호"], ascending=[True, True])
    .reset_index(drop=True)
)

st.subheader(f"🔎 가중치를 맞춘 {NAME_LOGIT}이 놓친 환자 — 전체 {len(missed):,}명")
st.caption(f"내 설정(기준 {threshold:.2f})으로 테스트 데이터에서 찾은 결과이고, 나이가 적은 순서로 정렬했어요.")
st.dataframe(missed, hide_index=True, use_container_width=True)
