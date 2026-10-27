import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"

st.set_page_config(page_title=f"채점표 · {APP_TITLE}", page_icon="📝", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

# '분류 모델' 페이지에서 처음 골라져 있던 입력 속성(체질량지수를 뺀 넷)과 같다
FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]
KOR = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

# 화면에 모델 이름을 적을 때는 항상 이 이름을 그대로 쓴다
NAME_LOGIT = "로지스틱 회귀(확률로 답하는 모델)"
NAME_TREE = "의사결정트리(질문으로 답하는 모델)"
NAME_BASE = "훈련용에서 많은 쪽으로만 답하는 모델"

NOT_AVAILABLE = "계산할 수 없음"
MAX_ROWS = 20  # 목록에 보여 줄 최대 줄 수


@st.cache_data(show_spinner="데이터를 불러오는 중이에요...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.markdown(
    """
    <style>
    table.cm { border-collapse: separate; border-spacing: 6px; width: 100%; table-layout: fixed; }
    table.cm th { background: #FFF3E0; color: #8D5A1F; padding: 8px; border-radius: 8px;
                  font-size: 0.9rem; font-weight: 700; text-align: center; }
    table.cm td { text-align: center; padding: 12px 6px; border-radius: 12px; }
    table.cm td.ok { background: #FFE0B2; border: 1px solid #FFCC80; }
    table.cm td.bad { background: #FFE0D6; border: 1px solid #FFAB91; }
    table.cm td.sum { background: #F5F5F5; border: 1px solid #E0E0E0; color: #424242; }
    table.cm .abbr { font-size: 1.0rem; font-weight: 800; color: #BF360C; }
    table.cm .num { font-size: 1.8rem; font-weight: 800; color: #5D4037; line-height: 1.2; }
    table.cm .mean { font-size: 0.78rem; color: #8D5A1F; }
    table.cm .sumnum { font-size: 1.3rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# 계산 도우미
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


def pct(value) -> str:
    return NOT_AVAILABLE if value is None else f"{value * 100:.2f}%"


def cm_table_html(cm: dict) -> str:
    """행은 실제, 열은 예측인 2×2 혼동행렬(합계 포함)을 표로 만든다."""
    tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]

    def cell(abbr: str, meaning: str, n: int, css: str) -> str:
        return (
            f'<td class="{css}"><div class="abbr">{abbr}</div>'
            f'<div class="num">{n:,}명</div><div class="mean">{meaning}</div></td>'
        )

    def total_cell(n: int) -> str:
        return f'<td class="sum"><div class="sumnum">{n:,}명</div></td>'

    return (
        '<table class="cm">'
        "<tr><th></th><th>예측: 뇌졸중</th><th>예측: 아님</th><th>실제 합계</th></tr>"
        "<tr><th>실제: 뇌졸중</th>"
        + cell("TP", "뇌졸중을 맞힘", tp, "ok")
        + cell("FN", "뇌졸중을 놓침", fn, "bad")
        + total_cell(tp + fn)
        + "</tr>"
        "<tr><th>실제: 아님</th>"
        + cell("FP", "아닌데 뇌졸중이라 함", fp, "bad")
        + cell("TN", "아님을 맞힘", tn, "ok")
        + total_cell(fp + tn)
        + "</tr>"
        "<tr><th>예측 합계</th>"
        + total_cell(tp + fp)
        + total_cell(fn + tn)
        + total_cell(tp + fp + fn + tn)
        + "</tr></table>"
    )


# ---------------------------------------------------------------
# 화면 맨 위
# ---------------------------------------------------------------
st.title("📝 채점표")
st.caption(
    f"{NAME_LOGIT}과 {NAME_TREE}이 테스트 데이터에서 어떻게 맞히고 틀렸는지 채점해요. "
    "뇌졸중을 양성으로 둬요."
)

try:
    raw = load_data()
except Exception as error:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n({error})")
    st.stop()

# ---------------------------------------------------------------
# 1) '분류 모델' 페이지와 똑같은 방법으로 나누고 학습하기
#    번호 순으로 정렬 → 열 명씩 묶기 → 각 묶음의 앞 세 명이 테스트용, 나머지 일곱 명이 학습용
# ---------------------------------------------------------------
df = raw.sort_values("id").reset_index(drop=True)
is_test = (df.index % 10) < 3

X_train = df.loc[~is_test, FEATURES].copy()
X_test = df.loc[is_test, FEATURES].copy()
y_train = df.loc[~is_test, "stroke"]
y_test = df.loc[is_test, "stroke"]

scaler = StandardScaler().fit(X_train)  # 학습용 일곱 명으로만 크기 맞추기
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

logit = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_s, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(
    X_train_s, y_train
)
baseline = DummyClassifier(strategy="most_frequent").fit(X_train_s, y_train)

y_true = y_test.to_numpy()
preds = {
    NAME_LOGIT: logit.predict(X_test_s),
    NAME_TREE: tree.predict(X_test_s),
    NAME_BASE: baseline.predict(X_test_s),
}
cms = {name: confusion(y_true, pred) for name, pred in preds.items()}

n_test = len(y_test)
n_test_stroke = int((y_test == 1).sum())

st.write(
    f"입력으로 쓴 속성은 {', '.join(KOR[c] for c in FEATURES)}이고, "
    "'분류 모델' 페이지에서 처음 골라져 있던 것과 같아요."
)

# ---------------------------------------------------------------
# 2) 혼동행렬 (행은 실제, 열은 예측)
# ---------------------------------------------------------------
st.subheader("🧮 혼동행렬")

left, right = st.columns(2)
for col, name in zip((left, right), (NAME_LOGIT, NAME_TREE)):
    with col:
        st.markdown(f"**{name}**")
        st.markdown(cm_table_html(cms[name]), unsafe_allow_html=True)
st.caption("행은 실제, 열은 예측이에요. TP·FP·FN·TN은 뇌졸중을 양성으로 둔 약자예요.")

# ---------------------------------------------------------------
# 3) 네 지표 비교표
# ---------------------------------------------------------------
st.subheader("📊 네 가지 지표")
st.write(f"테스트용 **{n_test:,}명** 가운데 뇌졸중인 사람은 **{n_test_stroke:,}명**이에요.")

rows = []
for name in (NAME_LOGIT, NAME_TREE, NAME_BASE):
    m = metrics(cms[name])
    rows.append(
        {
            "모델": name,
            "정확도": pct(m["정확도"]),
            "재현율": pct(m["재현율"]),
            "정밀도": pct(m["정밀도"]),
            "F1": pct(m["F1"]),
        }
    )
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption(
    "정확도 = (TP+TN) ÷ 전체 사람 수 · 재현율 = TP ÷ (TP+FN), 실제 뇌졸중인 사람 수로 나눠요 · "
    "정밀도 = TP ÷ (TP+FP), 뇌졸중이라 예측한 사람 수로 나눠요 · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율)"
)

# ---------------------------------------------------------------
# 4) 놓친 환자와 헛짚은 사람 목록
# ---------------------------------------------------------------
st.subheader("🔎 틀린 사람 살펴보기")
chosen_name = st.selectbox("어느 모델의 목록을 볼까요?", [NAME_LOGIT, NAME_TREE])
chosen_pred = preds[chosen_name]
chosen_cm = cms[chosen_name]

people = df.loc[is_test, ["id", "age", "avg_glucose_level", "hypertension", "heart_disease"]].copy()
people["hypertension"] = people["hypertension"].map({0: "없음", 1: "있음"})
people["heart_disease"] = people["heart_disease"].map({0: "없음", 1: "있음"})
people = people.rename(
    columns={
        "id": "번호",
        "age": "나이",
        "avg_glucose_level": "평균 혈당",
        "hypertension": "고혈압 여부",
        "heart_disease": "심장병 여부",
    }
)

missed = people[(y_true == 1) & (chosen_pred == 0)]  # 놓친 환자
false_alarm = people[(y_true == 0) & (chosen_pred == 1)]  # 헛짚은 사람


def sorted_by_age(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["나이", "번호"], ascending=[False, True]).reset_index(drop=True)


def show_list(container, title: str, frame: pd.DataFrame):
    total = len(frame)
    suffix = f"앞 {MAX_ROWS}명만 보여요" if total > MAX_ROWS else "모두 보여요"
    container.markdown(f"**{title} — 전체 {total:,}명** ({suffix})")
    container.dataframe(
        sorted_by_age(frame).head(MAX_ROWS), hide_index=True, use_container_width=True
    )


left, right = st.columns(2)
show_list(left, "놓친 환자 (실제 뇌졸중인데 아니라고 예측한 사람)", missed)
show_list(right, "헛짚은 사람 (아닌데 뇌졸중이라고 예측한 사람)", false_alarm)
st.caption("나이가 많은 순서로 정렬했어요.")

same_fn = len(missed) == chosen_cm["FN"]
same_fp = len(false_alarm) == chosen_cm["FP"]
if same_fn and same_fp:
    st.write(
        f"{chosen_name}의 놓친 환자 {len(missed):,}명은 혼동행렬의 FN 칸({chosen_cm['FN']:,}명)과, "
        f"헛짚은 사람 {len(false_alarm):,}명은 FP 칸({chosen_cm['FP']:,}명)과 같아요."
    )
else:
    st.warning(
        f"목록의 사람 수(놓친 환자 {len(missed):,}명, 헛짚은 사람 {len(false_alarm):,}명)가 "
        f"혼동행렬의 FN({chosen_cm['FN']:,}명)·FP({chosen_cm['FP']:,}명)와 달라요."
    )
