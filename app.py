from __future__ import annotations

from html import escape

import streamlit as st

from screengrab_demo.frame_io import MissingImageDependencyError, load_frames_from_uploads
from screengrab_demo.inference_config import build_classifier
from screengrab_demo.models import Frame
from screengrab_demo.pipeline import run_pipeline


GALLERY_INDEX_KEY = "screenshot_gallery_index"
UPLOAD_SIGNATURE_KEY = "screenshot_upload_signature"


def upload_signature(uploads: list[object]) -> tuple[tuple[str, int | None], ...]:
    return tuple(
        (getattr(upload, "name", "uploaded"), getattr(upload, "size", None))
        for upload in uploads
    )


def render_gallery(frames: list[Frame]) -> None:
    index = min(max(st.session_state.get(GALLERY_INDEX_KEY, 0), 0), len(frames) - 1)
    st.session_state[GALLERY_INDEX_KEY] = index

    previous_frame = frames[index - 1] if index > 0 else None
    current_frame = frames[index]
    next_frame = frames[index + 1] if index < len(frames) - 1 else None

    gallery_cols = st.columns([2.4, 6, 2.4], vertical_alignment="center")
    with gallery_cols[0]:
        if previous_frame:
            st.image(previous_frame.image, width="stretch")
    with gallery_cols[1]:
        st.image(current_frame.image, width="stretch")
    with gallery_cols[2]:
        if next_frame:
            st.image(next_frame.image, width="stretch")

    nav_cols = st.columns([2, 5, 2])
    with nav_cols[0]:
        if previous_frame and st.button("<", width="stretch"):
            st.session_state[GALLERY_INDEX_KEY] = index - 1
            st.rerun()
    with nav_cols[2]:
        if next_frame and st.button(">", width="stretch"):
            st.session_state[GALLERY_INDEX_KEY] = index + 1
            st.rerun()


def render_result_value(label: str, value: str) -> None:
    st.markdown(
        (
            "<div style='background: #e9ecef; border: 1px solid #d1d5db; "
            "border-radius: 6px; padding: 0.8rem 0.9rem; min-height: 4.25rem;'>"
            f"<div style='color: #6b7280; font-size: 0.75rem; line-height: 1.2; "
            f"margin-bottom: 0.35rem;'>{escape(label)}</div>"
            f"<div style='color: #111827; font-size: 1rem; font-weight: 600; "
            f"line-height: 1.3;'>{escape(value)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_explanation(reason: str, visible_text: tuple[str, ...]) -> None:
    st.markdown(
        (
            "<div style='margin-top: 1.75rem; padding-top: 0.25rem;'>"
            f"<p style='margin: 0; line-height: 1.6;'>{escape(reason)}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if visible_text:
        visible_text_items = "".join(f"<li>{escape(text)}</li>" for text in visible_text)
        st.markdown(
            (
                "<div style='margin-top: 1rem; line-height: 1.6;'>"
                "<strong>Visible text:</strong>"
                f"<ul style='margin: 0.4rem 0 0; padding-left: 1.25rem;'>{visible_text_items}</ul>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_fallback_reason(fallback_reason: str) -> None:
    st.markdown(
        (
            "<p style='color: #6b7280; font-size: 0.875rem; margin: 1rem 0 0;'>"
            f"Fallback reason: {escape(fallback_reason)}"
            "</p>"
        ),
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Emulation Failure Classifier", layout="wide")

content_cols = st.columns([1, 2, 1])
with content_cols[1]:
    st.title("Emulation Failure Classifier")
    gallery_slot = st.container()

    st.subheader("Input")
    uploads = st.file_uploader(
        "Upload screenshots",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=True,
    )
    st.selectbox("Model", ["Qwen3-VL-32B"])

    if uploads:
        signature = upload_signature(uploads)
        if st.session_state.get(UPLOAD_SIGNATURE_KEY) != signature:
            st.session_state[UPLOAD_SIGNATURE_KEY] = signature
            st.session_state[GALLERY_INDEX_KEY] = 0

        try:
            frames = load_frames_from_uploads(uploads)
        except MissingImageDependencyError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Could not load uploaded frames: {exc}")
            st.stop()

        if not frames:
            st.error("No screenshots were found in the upload.")
            st.stop()

        with gallery_slot:
            render_gallery(frames)

        try:
            with st.spinner("Processing"):
                result = run_pipeline(frames, build_classifier())
        except Exception:
            st.error("Processing failed.")
            st.caption("Check that inference is configured and available.")
            st.stop()

        st.subheader("Classification")
        stat_cols = st.columns(3)
        with stat_cols[0]:
            render_result_value("Error type", result.classification.category)
        with stat_cols[1]:
            render_result_value("Confidence", f"{result.classification.confidence:.2f}")
        with stat_cols[2]:
            evidence_frames = (
                ", ".join(str(frame) for frame in result.classification.evidence_frames)
                or "None"
            )
            render_result_value("Evidence frames", evidence_frames)

        if result.fallback_reason:
            render_fallback_reason(result.fallback_reason)
        render_explanation(result.classification.reason, result.classification.visible_text)
    else:
        st.info("Upload screenshots to classify them.")
