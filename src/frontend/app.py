import streamlit as st
import requests
import uuid
import json
import os

#  1. 기본 설정 
st.set_page_config(page_title="Multimodal RAG", layout="wide")

# Docker Compose 네트워크 안에서는 'localhost' 대신 서비스 이름('backend')을 써야 함.
API_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8000/api/v1")

HISTORY_FILE = "chat_history.json"

# 2. 데이터 저장/불러오기 함수 (새로고침 방지) 
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

# 3. 초기화 로직 (앱 실행 시 1회 수행) 
if "chat_sessions" not in st.session_state:
    saved_data = load_state()
    
    if saved_data:
        st.session_state.chat_sessions = saved_data["sessions"]
        st.session_state.active_session_id = saved_data["active_id"]
        st.session_state.chat_counter = saved_data.get("counter", 2)
    else:
        first_id = str(uuid.uuid4())
        st.session_state.chat_sessions = {
            first_id: {"title": "새로운 대화 1", "messages": [], "file_info": None}
        }
        st.session_state.active_session_id = first_id
        st.session_state.chat_counter = 2

def get_active_session():
    active_id = st.session_state.active_session_id
    if active_id not in st.session_state.chat_sessions:
        active_id = list(st.session_state.chat_sessions.keys())[0]
        st.session_state.active_session_id = active_id
        save_state()
    return active_id

#  4. 사이드바 (채팅방 관리 및 업로드) 
with st.sidebar:
    st.title("🗂️ 채팅방 목록")
    
    # 새 채팅방 만들기]
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        new_title = f"새로운 대화 {st.session_state.chat_counter}"
        
        st.session_state.chat_sessions[new_id] = {
            "title": new_title, 
            "messages": [], 
            "file_info": None
        }
        st.session_state.active_session_id = new_id
        st.session_state.chat_counter += 1
        save_state()
        st.rerun()

    st.divider()

    # [채팅방 목록 선택]
    session_ids = list(st.session_state.chat_sessions.keys())
    session_titles = [st.session_state.chat_sessions[s]["title"] for s in session_ids]
    
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

    selected_id = session_ids[session_titles.index(selected_title)]
    if selected_id != st.session_state.active_session_id:
        st.session_state.active_session_id = selected_id
        save_state()
        st.rerun()

    st.divider()
    
    # [현재 방 파일 관리]
    current_session_id = get_active_session()
    current_chat_data = st.session_state.chat_sessions[current_session_id]
    
    st.subheader("📄 문서 분석")

    # 이미 파일이 등록된 경우
    if current_chat_data.get("file_info"):
        info = current_chat_data["file_info"]
        st.success(f"✅ 분석 완료")
        st.info(f"📁 파일: {info['filename']}\n🏷️ 유형: {info['label']}")
            
    # 파일이 없는 경우 -> 업로드 UI 노출
    else:
        uploaded_file = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg", "pdf"])
        
        if uploaded_file:
            st.image(uploaded_file, caption="Preview", use_container_width=True)
            
            if st.button("🚀 분석 시작", type="primary"):
                with st.spinner("AI가 문서를 분석 중입니다..."):
                    try:
                        # 1. API 호출 준비
                        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                        data = {"session_id": current_session_id}
                        
                        # 2. POST /upload 요청 (환경변수 적용된 URL 사용)
                        response = requests.post(f"{API_BASE_URL}/upload", files=files, data=data)
                        
                        if response.status_code == 200:
                            res_json = response.json()
                            # 3. 결과 저장
                            current_chat_data["file_info"] = {
                                "filename": res_json["filename"],
                                "label": res_json["label"]
                            }
                            save_state()
                            st.rerun()
                        else:
                            st.error(f"업로드 실패: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("🚨 서버 연결 실패! 백엔드가 켜져 있는지 확인하세요.")
                    except Exception as e:
                        st.error(f"에러 발생: {e}")

    st.divider()
    # [삭제 버튼]
    if st.button("🗑️ 이 채팅방 삭제"):
        if len(st.session_state.chat_sessions) > 1:
            del st.session_state.chat_sessions[current_session_id]
            st.session_state.active_session_id = list(st.session_state.chat_sessions.keys())[0]
            save_state()
            st.rerun()
        else:
            st.warning("최소 하나의 채팅방은 있어야 합니다.")

# 5. 메인 채팅 화면 
active_id = get_active_session()
current_chat = st.session_state.chat_sessions[active_id]

st.header(current_chat["title"])

# [대화 기록 출력]
for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# [사용자 입력 처리]
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 표시
    current_chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 백엔드 통신
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            try:
                payload = {
                    "session_id": active_id, 
                    "query": prompt
                }
                
                # API 호출 (환경변수 적용된 URL 사용)
                response = requests.post(f"{API_BASE_URL}/chat", json=payload)

                if response.status_code == 200:
                    res_json = response.json()
                    answer = res_json.get("response", "응답 없음")
                    doc_category = res_json.get("category", "General")
                    
                    # 제목 업데이트 (첫 질문일 때)
                    if len(current_chat["messages"]) == 1:
                        new_title = prompt[:15] + "..." if len(prompt) > 15 else prompt
                        current_chat["title"] = new_title
                        st.session_state.chat_sessions[active_id]["title"] = new_title

                    # 답변 출력
                    if doc_category:
                        st.caption(f"🧠 Context: {doc_category}")
                        
                    st.write(answer)
                    
                    current_chat["messages"].append({"role": "assistant", "content": answer})
                    save_state()
                    
                else:
                    st.error(f"Server Error: {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error(f"🚨 연결 실패: {API_BASE_URL}에 접속할 수 없습니다.")
            except Exception as e:
                st.error(f"Connection Error: {e}")