# Project Roadmap: Sentiment Omikuji

This roadmap breaks down the implementation into six logical phases.

## Phase 1: Foundation & Environment
- [ ] Initialize Rails 7 application (`rails new omikuji_app --css tailwind --database postgresql`).
- [ ] Configure `Gemfile` with `onnxruntime`, `tokenizers`, and `turbo-rails`.
- [ ] Set up directory structure for models (`/vendor/models`).
- [ ] Create a `ModelLoader` initializer to handle singleton ONNX sessions.

## Phase 2: The "Ear" (Sentiment Analysis Service)
- [ ] Implement `SentimentAnalysisService`.
- [ ] Port the Japanese WordPiece tokenizer logic using the `tokenizers` gem.
- [ ] Load the BERT ONNX model and implement the `predict` method.
- [ ] **Validation:** Create RSpec tests to verify sentiment scores for known inputs (e.g., "嬉しい" -> Positive).

## Phase 3: The "Voice" (Fortune Generation)
- [ ] Select and download a tiny Japanese LLM (e.g., GPT-2 Japanese Small).
- [ ] Implement `FortuneGeneratorService`.
- [ ] Design prompt templates for `Great Blessing`, `Small Blessing`, and `Curse` vibes.
- [ ] **Validation:** Verify that the service returns coherent Japanese sentences based on input sentiment.

## Phase 4: Core Web Workflow
- [ ] Generate `Fortune` model (fields: `input_text`, `sentiment_score`, `fortune_text`, `rank`).
- [ ] Build `FortunesController` and basic routes.
- [ ] Implement the "Input Form" with `remote: true`.
- [ ] Set up Turbo Stream responses to update the page without refresh.

## Phase 5: Thematic UI & UX
- [ ] Design the "Omikuji Slip" component using Tailwind CSS.
- [ ] Add Stimulus.js controller for the "Box Shaking" animation.
- [ ] Implement "Result Reveal" animations (e.g., fading in the parchment, falling blossoms).
- [ ] Ensure mobile-friendly layout for "On-the-go" fortunes.

## Phase 6: Deployment & Polish
- [ ] Optimize model loading to prevent memory bloat in production.
- [ ] Pre-populate a "Demo" mode with existing review data.
- [ ] Deploy to a hosting provider (e.g., Fly.io or Render).
- [ ] **Final Review:** Ensure all Student Code of Conduct requirements (from the original repo) are met.
