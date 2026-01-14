import streamlit as st
import requests
import uuid
import json
import os

# --- 1. 기본 설정 ---
st.set_page_config(page_title="Multimodal RAG", layout="wide")

API_BASE_URL = "http://localhost:8000/api/v1"
HISTORY_FILE = "chat_history.json"

# --- 2. 데이터 저장/불러오기 함수 (새로고침 방지) ---
def save_state():
    """현재 세션 상태를 JSON 파일로 저장"""
    data = {
        "sessions": st.session_state.chat_sessions,
        "active_id": st.session_state.active_session_id,
        "counter": st.session_state.chat_counter
    }
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_state():
    """JSON 파일에서 이전 상태 불러오기"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None

# --- 3. 초기화 로직 (앱 실행 시 1회 수행) ---
if "chat_sessions" not in st.session_state:
    saved_data = load_state()
    
    if saved_data:
        # 📂 저장된 기록 복원
        st.session_state.chat_sessions = saved_data["sessions"]
        st.session_state.active_session_id = saved_data["active_id"]
        st.session_state.chat_counter = saved_data.get("counter", 2)
        print("✅ 이전 대화 기록을 복원했습니다.")
    else:
        # 🆕 신규 시작
        first_id = str(uuid.uuid4())
        st.session_state.chat_sessions = {
            first_id: {"title": "새로운 대화 1", "messages": []}
        }
        st.session_state.active_session_id = first_id
        st.session_state.chat_counter = 2  # 다음은 2번부터

# 현재 활성 세션 ID 가져오기 (안전장치 포함)
def get_active_session():
    active_id = st.session_state.active_session_id
    if active_id not in st.session_state.chat_sessions:
        active_id = list(st.session_state.chat_sessions.keys())[0]
        st.session_state.active_session_id = active_id
        save_state()
    return active_id

# --- 4. 사이드바 (채팅방 관리) ---
with st.sidebar:
    st.title("🗂️ 채팅방 목록")
    
    # [➕ 새 채팅방 만들기 버튼]
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        # 카운터 변수를 사용해 제목 생성
        new_title = f"새로운 대화 {st.session_state.chat_counter}"
        
        # 세션 추가
        st.session_state.chat_sessions[new_id] = {"title": new_title, "messages": []}
        st.session_state.active_session_id = new_id
        
        # 카운터 증가 및 저장
        st.session_state.chat_counter += 1
        save_state()
        st.rerun()

    st.divider()

    # [채팅방 목록 표시]
    session_ids = list(st.session_state.chat_sessions.keys())
    session_titles = [st.session_state.chat_sessions[s]["title"] for s in session_ids]
    
    # 현재 선택된 인덱스 찾기
    try:
        active_index = session_ids.index(st.session_state.active_session_id)
    except ValueError:
        active_index = 0

    selected_title = st.radio(
        "대화 목록",
        session_titles,
        index=active_index,
        label_visibility="collapsed"
    )

    # 선택된 타이틀로 ID 역추적해서 활성 세션 변경
    selected_id = session_ids[session_titles.index(selected_title)]
    if selected_id != st.session_state.active_session_id:
        st.session_state.active_session_id = selected_id
        save_state()
        st.rerun()

    st.divider()
    
    # [현재 방 정보]
    current_session_id = get_active_session()
    
    # 파일 업로더
    uploaded_file = st.file_uploader("📄 현재 방에 파일 추가", type=["png", "jpg", "jpeg", "pdf"])
    if uploaded_file:
        st.image(uploaded_file, caption="Preview", use_container_width=True)

    # [채팅방 삭제 버튼]
    if st.button("🗑️ 이 채팅방 삭제", type="primary"):
        if len(st.session_state.chat_sessions) > 1:
            # 1. 백엔드 메모리 삭제 요청
            try:
                requests.delete(f"{API_BASE_URL}/chat/session/{current_session_id}")
            except Exception as e:
                print(f"서버 삭제 실패 (무시): {e}")
            
            # 2. 프론트엔드 삭제
            del st.session_state.chat_sessions[current_session_id]
            st.session_state.active_session_id = list(st.session_state.chat_sessions.keys())[0]
            
            # 3. 저장 및 새로고침
            save_state()
            st.rerun()
        else:
            st.warning("최소 하나의 채팅방은 있어야 합니다.")

# --- 5. 메인 채팅 화면 ---
active_id = get_active_session()
current_chat = st.session_state.chat_sessions[active_id]

st.header(current_chat["title"])

# (1) 대화 기록 출력
for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# (2) 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 UI 표시 및 저장
    current_chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 백엔드 통신
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            try:
                files = None
                if uploaded_file:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

                payload = {"session_id": active_id, "query": prompt}
                
                # 파일 유무에 따라 요청 분기
                if files:
                    response = requests.post(f"{API_BASE_URL}/chat", data=payload, files=files)
                else:
                    response = requests.post(f"{API_BASE_URL}/chat", json=payload)

                if response.status_code == 200:
                    res_json = response.json()
                    answer = res_json.get("response", "응답 없음")
                    
                    # 제목 업데이트 (첫 질문일 경우 제목을 질문 내용으로 변경)
                    if len(current_chat["messages"]) == 1:
                        new_title = prompt[:15] + "..." if len(prompt) > 15 else prompt
                        current_chat["title"] = new_title
                        st.session_state.chat_sessions[active_id]["title"] = new_title # 확실하게 반영
                        # st.rerun() # 제목 변경 반영을 위해 리런

                    st.write(answer)
                    current_chat["messages"].append({"role": "assistant", "content": answer})
                    
                    # 대화 끝날 때마다 상태 저장
                    save_state()
                    
                else:
                    st.error(f"Error: {response.text}")
            
            except Exception as e:
                st.error(f"Connection Error: {e}")