# Sentiment Omikuji (感情おみくじ)

An interactive Japanese sentiment analysis web application that combines deep learning (BERT) with generative AI (Small LLM) to provide traditional-style fortunes based on the user's current mood.

## Project Vision
To demonstrate the power of high-performance NLP inference in a Ruby/Rails environment. The app uses an ONNX-exported BERT model to "hear" the user's emotion and a tiny generative model to "speak" a fortune.

## Tech Stack
- **Framework:** Ruby on Rails 7.x
- **Inference Engine:** ONNX Runtime (via the `onnxruntime` gem)
- **Sentiment Model:** BERT (Japanese)
- **Generative Model:** Small LLM (e.g., GPT-2 Japanese or Phi-2)
- **Frontend:** Tailwind CSS + Hotwire (Turbo/Stimulus)
- **Data Source:** Japanese review dataset for model calibration

## Getting Started
(Development in progress)
- Refer to `spec/design_overview.md` for the technical architecture.
- Refer to `ruby/FFI_APPROACH_NOTES.md` for historical context on the inference implementation.