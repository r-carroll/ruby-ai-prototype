# Project Roadmap: Sentiment Omikuji

This roadmap tracks the evolution of the Sentiment Omikuji application.

## Phase 1: Foundation & Environment (COMPLETED)
- [x] Initialize Rails 8 application with Modern Defaults.
- [x] Configure `Gemfile` with `onnxruntime`, `tokenizers`, `natto` (MeCab), and `turbo-rails`.
- [x] Set up directory structure for models (`/models/bert_model`).
- [x] Create a `ModelLoader` singleton to handle ONNX sessions efficiently.

## Phase 2: The "Ear" (Sentiment Analysis Service) (COMPLETED)
- [x] Implement `SentimentAnalysisService`.
- [x] Port Japanese tokenization logic using `natto` and `tokenizers`.
- [x] Load BERT ONNX model and implement `predict` method.
- [x] **Validation:** Verified sentiment scores via RSpec.

## Phase 3: The "Voice" (Fortune Generation) (IN PROGRESS)
- [x] Implement custom `MarkovService` for high-speed generation (Current Engine).
- [ ] Integrate `rinna/japanese-gpt2-small` ONNX model (Future Engine).
- [x] Design prompt templates for various sentiment ranks.
- [x] **Validation:** Verified coherent Japanese output.

## Phase 4: Core Web Workflow & Async Architecture (COMPLETED)
- [x] Generate `Fortune` model and database schema.
- [x] Implement `FortunesController` and views.
- [x] Migrate to **Solid Queue** for asynchronous ML inference.
- [x] Set up **Solid Cable** for real-time result broadcasting.
- [x] Implement `FortuneGenerationJob` to decouple web requests from ML processing.

## Phase 5: Thematic UI & UX (COMPLETED)
- [x] Design the "Omikuji Slip" with Tailwind CSS.
- [x] Add Stimulus.js controller for the shaking box animation.
- [x] Implement Turbo Stream updates for real-time reveal.
- [x] Mobile-friendly responsive design.

## Phase 6: Modern Infrastructure & Deployment (COMPLETED)
- [x] Configure **Multi-Database SQLite** for production (primary, queue, cache, cable).
- [x] Set up **Kamal** for zero-downtime containerized deployment.
- [x] Integrate **Thruster** for asset acceleration and compression.
- [x] Deploy to Fly.io (`nrt` region).

## Future Polish (Planned)
- [ ] Add "History" view for users to see their past fortunes.
- [ ] Implement rate limiting to protect ML resources.
- [ ] Fine-tune GPT-2 specifically on omikuji-style poetic Japanese.
- [ ] Explore WebAssembly (Wasm) for client-side BERT inference to further reduce server load.
