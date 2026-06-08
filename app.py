"""
app.py — Gradio web interface for the UST Unofficial Housing Guide
Run:  python app.py
Then open http://localhost:7860 in your browser.
"""

import gradio as gr
from query import ask


# ── Query handler ─────────────────────────────────────────────────────────────

def handle_query(question: str) -> tuple[str, str, str]:
    """
    Call the RAG pipeline and format output for the three Gradio output boxes.
    Returns: (answer, sources_text, debug_text)
    """
    question = question.strip()
    if not question:
        return "Please enter a question.", "", ""

    result = ask(question)

    # Format sources as a bulleted list
    sources_text = "\n".join(f"• {s}" for s in result["sources"]) if result["sources"] else "No sources retrieved."

    # Format debug info (top chunks with distances)
    debug_lines = []
    for i, chunk in enumerate(result["chunks"], 1):
        debug_lines.append(
            f"[{i}] {chunk['source']} (distance: {chunk['distance']})\n"
            f"    {chunk['text'][:200].replace(chr(10), ' ')}..."
        )
    debug_text = "\n\n".join(debug_lines) if debug_lines else ""

    return result["answer"], sources_text, debug_text


# ── Gradio UI ─────────────────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    "What do students say about parking near UST?",
    "Are there mold problems in apartments near campus?",
    "Which landlords or property managers are responsive?",
    "Is it safe to walk at night in Mac-Groveland?",
    "How much should I expect to pay for a 1-bedroom near UST?",
    "What are the best bus routes for students living off campus?",
    "What should I look for on an apartment tour?",
    "How do I get my security deposit back in Minnesota?",
]

with gr.Blocks(
    title="UST Unofficial Housing Guide",
    theme=gr.themes.Soft(),
    css="""
    .main-header { text-align: center; margin-bottom: 1rem; }
    .disclaimer { font-size: 0.85em; color: #666; margin-top: 0.5rem; }
    """,
) as demo:

    gr.Markdown(
        """
        # 🏠 UST Unofficial Housing Guide
        ### Student-powered answers about off-campus housing near the University of St. Thomas (St. Paul)

        Ask anything about apartments, landlords, neighborhoods, parking, transit, leases, or safety.
        All answers are grounded in real student-written reviews and forum posts — no made-up advice.
        """,
        elem_classes="main-header",
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Your question",
                placeholder='e.g. "Is parking easy near Summit Ave?" or "Which buildings have mold issues?"',
                lines=2,
            )
            ask_btn = gr.Button("Ask", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("**Example questions:**")
            for eq in EXAMPLE_QUESTIONS[:4]:
                gr.Button(eq, size="sm").click(
                    fn=lambda q=eq: q,
                    outputs=question_input,
                )

    with gr.Row():
        with gr.Column(scale=2):
            answer_output = gr.Textbox(
                label="Answer",
                lines=10,
                interactive=False,
            )
        with gr.Column(scale=1):
            sources_output = gr.Textbox(
                label="Retrieved from",
                lines=6,
                interactive=False,
            )

    with gr.Accordion("Retrieved chunks (debug view)", open=False):
        debug_output = gr.Textbox(
            label="Top retrieved chunks with similarity scores",
            lines=15,
            interactive=False,
        )

    gr.Markdown(
        """
        <div class="disclaimer">
        ⚠️ This guide reflects student experiences and opinions — not official UST information.
        Reviews may be outdated. Always verify details with landlords and official sources before signing.
        </div>
        """,
    )

    # Wire up
    ask_btn.click(
        fn=handle_query,
        inputs=question_input,
        outputs=[answer_output, sources_output, debug_output],
    )
    question_input.submit(
        fn=handle_query,
        inputs=question_input,
        outputs=[answer_output, sources_output, debug_output],
    )


if __name__ == "__main__":
    demo.launch(show_error=True)
