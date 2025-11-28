# app.py
"""
Gradio app for the workshop.
Two modes:
 - Single Review -> instant sentiment
 - Upload CSV (with 'review' column) -> run batch predictions and download CSV
"""

import gradio as gr
import pandas as pd
from model import analyze_text, analyze_batch
from data_loader import load_data
from utils import add_predictions_to_df
from io import StringIO, BytesIO

# Single prediction function
def predict_single(review: str):
    pred = analyze_text(review)
    label = pred.get('label')
    score = pred.get('score')
    return label, float(score)

# Batch prediction function for uploaded CSV
def predict_file(file_obj):
    """
    Accepts an uploaded CSV file (file-like). Returns a downloadable CSV with predictions.
    """
    try:
        df = pd.read_csv(file_obj.name) if hasattr(file_obj, "name") else pd.read_csv(file_obj)
    except Exception as e:
        return "Error reading CSV: " + str(e), None

    # find review-like column
    cols = df.columns
    text_cols = [c for c in cols if 'review' in c.lower() or 'text' in c.lower()]
    if not text_cols:
        return "Uploaded CSV must contain a text column named like 'review' or 'text'.", None
    text_col = text_cols[0]

    texts = df[text_col].fillna("").astype(str).tolist()
    preds = analyze_batch(texts, batch_size=32)
    out_df = add_predictions_to_df(df, preds)

    # prepare downloadable CSV in memory
    buffer = StringIO()
    out_df.to_csv(buffer, index=False)
    buffer.seek(0)
    return "Success: Predictions added", ("predictions.csv", buffer.getvalue(), "text/csv")

# Optional demo: load a few rows from local imdb.csv (if present)
def demo_sample():
    try:
        df = load_data()
        sample = df.head(5).to_dict(orient='records')
        # show text samples in the UI
        texts = [r['review'] for r in sample]
        preds = analyze_batch(texts, batch_size=8)
        return {f"Review {i+1}": (texts[i], preds[i]['label'], preds[i]['score']) for i in range(len(texts))}
    except Exception as e:
        return {"error": str(e)}

with gr.Blocks() as demo:
    gr.Markdown("# Movie Review Sentiment — Workshop App")
    gr.Markdown("**Single prediction** — Type a review and get sentiment.")
    with gr.Row():
        txt = gr.Textbox(lines=4, label="Enter movie review here")
        out_label = gr.Textbox(label="Predicted label")
        out_score = gr.Number(label="Confidence score")
    btn = gr.Button("Analyze")
    btn.click(fn=lambda t: predict_single(t), inputs=[txt], outputs=[out_label, out_score])

    gr.Markdown("----")
    gr.Markdown("**Batch prediction** — Upload a CSV with a `review` (or `text`) column.")
    csv_in = gr.File(label="Upload CSV")
    status = gr.Textbox(label="Status")
    download_button = gr.File(label="Download predictions (after running)")
    run_btn = gr.Button("Run batch predictions")
    def run_and_return(file):
        msg, download = predict_file(file)
        # gr.File requires a filename/path: return tuple (filename, content, mime)
        return msg, download
    run_btn.click(fn=run_and_return, inputs=[csv_in], outputs=[status, download_button])

    gr.Markdown("----")
    gr.Markdown("**Demo sample (if `imdb.csv` exists locally)**")
    sample_btn = gr.Button("Load demo sample & predict")
    demo_output = gr.JSON()
    sample_btn.click(fn=demo_sample, inputs=None, outputs=[demo_output])

if __name__ == "__main__":
    # When running locally for the workshop
    demo.launch(server_name="0.0.0.0", server_port=7860)
