from __future__ import annotations

import os
import tempfile
import time
import traceback
from pathlib import Path

import streamlit as st

from v2a_inspect.observability import WorkflowTraceContext
from v2a_inspect.runner import (
    get_grouped_analysis,
    run_group_from_scene_analysis,
    run_scene_analysis_only,
)
from v2a_inspect.settings import settings
from v2a_inspect.ui.auth import require_authentication
from v2a_inspect.ui.render import (
    render_footer,
    render_page_header,
    render_results,
    render_scene_analysis_preview,
    render_sidebar,
)
from v2a_inspect.ui.session import (
    ensure_process_resources,
    get_analysis_semaphore,
    get_langfuse_session_id,
    get_user_last_analysis_times,
    initialize_session_state,
    reset_state,
)
from v2a_inspect.ui.video import (
    get_video_duration,
    save_uploaded_file,
    validate_video_file,
)
from v2a_inspect.workflows import InspectOptions

_MAX_FILE_SIZE_MB = 200
_MAX_DURATION_SECONDS = 30.0


def main() -> None:
    st.set_page_config(page_title="V2A Inspect", page_icon="🔍", layout="wide")

    authenticator = require_authentication()
    ensure_process_resources()
    initialize_session_state()

    render_page_header()
    options = render_sidebar(authenticator)
    render_upload_step(options)

    scene_analysis = st.session_state.get("scene_analysis")
    video_path = st.session_state.get("video_path") or ""
    clip_dir = st.session_state.get("clip_dir") or ""

    # Step 2: 씬 분석 결과 표시
    if scene_analysis is not None:
        render_scene_analysis_preview(
            scene_analysis, video_path=video_path, clip_dir=clip_dir
        )

        # Step 3: 그루핑 버튼 (아직 그루핑 안 된 경우)
        if st.session_state.get("grouped") is None:
            render_grouping_step(options)

    # Step 4: 그루핑 결과 표시
    grouped = st.session_state.get("grouped")
    if grouped is not None and scene_analysis is not None:
        render_results(
            grouped,
            scene_analysis,
            video_path=video_path,
            clip_dir=clip_dir,
            inspect_state=st.session_state.get("inspect_state"),
        )

    render_footer()


def render_upload_step(options: InspectOptions) -> None:
    st.header("Step 1: 영상 업로드")

    uploaded_file = st.file_uploader(
        "영상 파일 선택",
        type=["mp4", "mov", "avi", "mkv"],
        help=f"MP4, MOV, AVI, MKV | 최대 {_MAX_DURATION_SECONDS:.0f}초 / {_MAX_FILE_SIZE_MB}MB",
    )
    if uploaded_file is None:
        return

    is_new_video = (
        st.session_state.video_path is None
        or not Path(st.session_state.video_path).exists()
        or Path(st.session_state.video_path).name != uploaded_file.name
    )
    if is_new_video:
        reset_state()
        st.session_state.video_path = save_uploaded_file(uploaded_file)

        if not validate_video_file(st.session_state.video_path):
            st.error("유효한 영상 파일이 아닙니다.")
            reset_state()
            st.stop()

        file_size_mb = os.path.getsize(st.session_state.video_path) / (1024 * 1024)
        if file_size_mb > _MAX_FILE_SIZE_MB:
            st.error(
                f"파일 크기 {file_size_mb:.0f}MB 초과. 최대 {_MAX_FILE_SIZE_MB}MB."
            )
            reset_state()
            st.stop()

        duration = get_video_duration(st.session_state.video_path)
        if duration is not None and duration > _MAX_DURATION_SECONDS:
            st.error(
                f"영상 길이 {duration:.1f}초 초과. 최대 {_MAX_DURATION_SECONDS:.0f}초."
            )
            reset_state()
            st.stop()

    st.video(uploaded_file)

    phase1_done = st.session_state.get("scene_analysis") is not None
    btn_disabled = phase1_done or bool(st.session_state.get("is_analyzing"))
    if st.button(
        "🔍 1차 분석 (씬 분석)",
        type="primary",
        disabled=btn_disabled,
        help="Gemini로 씬/이벤트 분석. 결과 확인 후 그루핑을 따로 진행합니다.",
    ):
        run_phase1(st.session_state.video_path, options)

    if st.session_state.get("is_analyzing"):
        st.info("⏳ 분석 진행 중...")
    elif phase1_done:
        st.success("✅ 씬 분석 완료. 아래에서 결과를 확인하세요.")


def render_grouping_step(options: InspectOptions) -> None:
    st.divider()
    st.header("Step 3: 그루핑 / 검증 / 모델 선정")
    btn_disabled = bool(st.session_state.get("is_analyzing"))
    if st.button(
        "🔗 그루핑 시작",
        type="primary",
        disabled=btn_disabled,
        help="트랙 그루핑, VLM 검증, TTA/VTA 모델 선정 실행",
    ):
        run_phase2(options)
    if st.session_state.get("is_analyzing"):
        st.info("⏳ 그루핑 진행 중...")


def run_phase1(video_path: str, options: InspectOptions) -> None:
    _check_cooldown()
    st.session_state["is_analyzing"] = True

    clip_dir = tempfile.mkdtemp(prefix="v2a_inspect_clips_")
    st.session_state.clip_dir = clip_dir

    semaphore = get_analysis_semaphore()
    acquired = semaphore.acquire(timeout=settings.ui_analysis_acquire_timeout_seconds)
    if not acquired:
        st.session_state["is_analyzing"] = False
        st.error("서버가 바쁩니다. 잠시 후 다시 시도해주세요.")
        st.stop()

    _record_cooldown()
    try:
        with st.status("씬 분석 중...", expanded=True) as status:
            try:
                state = run_scene_analysis_only(
                    video_path,
                    options=options,
                    progress_callback=status.write,
                    warning_callback=lambda m: status.write(f"⚠️ {m}"),
                    trace_context=_build_ui_trace_context(options),
                )
                scene_analysis = state.get("scene_analysis")
                if scene_analysis is None:
                    raise ValueError("씬 분석 결과 없음.")

                st.session_state.scene_analysis = scene_analysis
                st.session_state.gemini_file = state.get("gemini_file")
                st.session_state.inspect_state = state

                n_scenes = len(scene_analysis.scenes)
                n_events = sum(len(s.audio_events) for s in scene_analysis.scenes)
                status.write(f"✅ {n_scenes}개 씬 / {n_events}개 이벤트 분석 완료")
                status.update(label="씬 분석 완료!", state="complete")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                status.update(label="분석 실패", state="error")
                st.error(f"오류: {exc}")
                st.code(traceback.format_exc())
    finally:
        semaphore.release()
        st.session_state["is_analyzing"] = False


def run_phase2(options: InspectOptions) -> None:
    _check_cooldown()
    scene_analysis = st.session_state.get("scene_analysis")
    if scene_analysis is None:
        st.error("씬 분석 결과가 없습니다. 1차 분석을 먼저 실행해주세요.")
        st.stop()

    st.session_state["is_analyzing"] = True

    semaphore = get_analysis_semaphore()
    acquired = semaphore.acquire(timeout=settings.ui_analysis_acquire_timeout_seconds)
    if not acquired:
        st.session_state["is_analyzing"] = False
        st.error("서버가 바쁩니다. 잠시 후 다시 시도해주세요.")
        st.stop()

    _record_cooldown()
    try:
        with st.status("그루핑 진행 중...", expanded=True) as status:
            try:
                state = run_group_from_scene_analysis(
                    scene_analysis,
                    options=options,
                    video_path=st.session_state.get("video_path") or "",
                    gemini_file=st.session_state.get("gemini_file"),
                    progress_callback=status.write,
                    warning_callback=lambda m: status.write(f"⚠️ {m}"),
                    trace_context=_build_ui_trace_context(options),
                )
                grouped = get_grouped_analysis(state)
                st.session_state.inspect_state = state
                st.session_state.grouped = grouped

                n_groups = len(grouped.groups)
                n_tracks = len(grouped.raw_tracks)
                status.write(f"✅ {n_tracks}개 트랙 → {n_groups}개 그룹")
                status.update(label="그루핑 완료!", state="complete")
                st.rerun()
            except TimeoutError:
                status.update(label="Timeout", state="error")
                st.error("처리 시간 초과. 더 짧은 영상을 사용해주세요.")
            except Exception as exc:  # noqa: BLE001
                status.update(label="그루핑 실패", state="error")
                st.error(f"오류: {exc}")
                st.code(traceback.format_exc())
    finally:
        semaphore.release()
        st.session_state["is_analyzing"] = False


def _check_cooldown() -> None:
    username = str(st.session_state.get("username") or "anonymous")
    cooldown = settings.ui_analysis_cooldown_seconds
    if cooldown > 0:
        elapsed = time.time() - get_user_last_analysis_times().get(username, 0.0)
        if elapsed < cooldown:
            st.error(
                f"요청이 너무 빠릅니다. {int(cooldown - elapsed)}초 후 다시 시도해주세요."
            )
            st.stop()


def _record_cooldown() -> None:
    username = str(st.session_state.get("username") or "anonymous")
    get_user_last_analysis_times()[username] = time.time()


def _build_ui_trace_context(options: InspectOptions) -> WorkflowTraceContext:
    username = st.session_state.get("username")
    tags: list[str] = []
    if options.enable_vlm_verify:
        tags.append("vlm-verify")
    if options.enable_model_select:
        tags.append("model-select")

    return WorkflowTraceContext(
        source="ui",
        operation="analyze",
        user_id=str(username) if username else None,
        session_id=get_langfuse_session_id(),
        tags=tuple(tags),
        metadata={
            "scene_analysis_mode": options.scene_analysis_mode,
            "fps": options.fps,
            "auth_mode": settings.auth_mode,
        },
    )


if __name__ == "__main__":
    main()
