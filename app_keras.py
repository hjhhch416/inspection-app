import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import streamlit as st

# ── 1. 페이지 설정 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="가죽 불량 검사 시스템",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 가죽 이상 탐지 시스템")
st.caption("AI 기반의 이미지 분석을 통해 가죽의 정상/불량 여부를 실시간으로 판별합니다.")

# ── 설정 및 상수 ────────────────────────────────────────────────────
MODEL_PATH = "./weights/leather_model.keras"
INPUT_IMG_SIZE = (224, 224)
CLASSES = ["정상", "불량"]


# ── 2. 모델 로드 (캐싱 적용) ──────────────────────────────────────────
@st.cache_resource
def load_model_cached():
    if not os.path.exists(MODEL_PATH):
        st.error(f"🚨 모델 파일을 찾을 수 없습니다: {MODEL_PATH}\n올바른 경로에 모델 파일이 있는지 확인해 주세요.")
        st.stop()  # 앱 실행 중단
    
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# ── 3. 전처리 및 추론 로직 ──────────────────────────────────────────
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = keras.applications.vgg16.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict(model, pil_img):
    arr = preprocess(pil_img)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = CLASSES[1 if prob > 0.5 else 0]
    return label, prob


# ── 메인 실행 루프 ──────────────────────────────────────────────────
def main():
    # 모델 불러오기 (최초 1회만 실행됨)
    model = load_model_cached()

    # 3. 이미지 입력 방식 선택 (사이드바 또는 메인 화면)
    st.subheader("📸 이미지 입력")
    input_mode = st.radio(
        "검사할 이미지의 입력 방식을 선택하세요.",
        ("파일 업로드", "카메라 촬영"),
        horizontal=True
    )

    uploaded_file = None
    pil_img = None

    if input_mode == "파일 업로드":
        uploaded_file = st.file_uploader(
            "가죽 이미지를 업로드하세요.", 
            type=["jpg", "jpeg", "png"]
        )
    else:
        uploaded_file = st.camera_input("웹캠으로 가죽을 촬영해 주세요.")

    # 사용자가 이미지를 제공했을 경우 PIL Image로 변환하여 미리보기
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        if input_mode == "파일 업로드":
            st.image(pil_img, caption="업로드된 이미지", use_column_width=True)

    st.write("---")

    # 4. 검사 실행 버튼
    if st.button("🔍 검사 시작", type="primary"):
        if pil_img is None:
            st.warning("⚠️ 검사할 이미지를 먼저 업로드하거나 촬영해 주세요.")
        else:
            with st.spinner("AI가 이미지를 분석하는 중입니다..."):
                label, prob = predict(model, pil_img)
            
            st.subheader("📊 검사 결과")
            
            # 5. 결과 표시 (정상/불량에 따른 조건부 UI)
            if label == "정상":
                st.success(f"### 완제품 판정: **{label}**")
            else:
                st.error(f"### 완제품 판정: **{label} (불량 감지)**")
            
            # 확률 메트릭 나란히 표시
            normal_prob = 1 - prob
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="정상 확률", value=f"{normal_prob:.1%}")
            with col2:
                st.metric(label="불량 확률", value=f"{prob:.1%}")
            
            # 불량 확률 프로그레스 바 표시
            st.write("**불량 위험도 위험 수치**")
            st.progress(prob)


if __name__ == "__main__":
    main()