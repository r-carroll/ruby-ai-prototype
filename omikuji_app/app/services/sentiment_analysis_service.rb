# omikuji_app/app/services/sentiment_analysis_service.rb

class SentimentAnalysisService
  # Sentiment labels based on our BERT training (0: Negative, 1: Positive)
  LABELS = {
    0 => :negative,
    1 => :positive
  }.freeze

  def initialize(text)
    @text = text
    @loader = ModelLoader.instance
    @tokenizer = @loader.bert_tokenizer
    @bert_session = @loader.bert_session
  end

  def predict
    return { error: "Model or Tokenizer not loaded" } unless @bert_session && @tokenizer

    # BERT Japanese models expect text to be pre-segmented by MeCab
    segmented_text = @loader.mecab.parse(@text).split("\n").map { |line| line.split("\t").first }.join(" ")

    # Configure padding/truncation to match training (128 tokens)
    @tokenizer.enable_padding(length: 128)
    @tokenizer.enable_truncation(128)

    # 1. Tokenize the input
    encoding = @tokenizer.encode(segmented_text)
    input_ids = encoding.ids
    attention_mask = encoding.attention_mask

    # 2. Prepare tensors for ONNX Runtime
    inputs = {
      "input_ids" => [input_ids],
      "attention_mask" => [attention_mask]
    }

    # 3. Run Inference
    results = @bert_session.predict(inputs)
    logits = results["output"] # results is a hash with output names as keys

    # 4. Process Results (Softmax/Argmax)
    # logits is an array of arrays: [[neg_score, pos_score]]
    scores = logits.first
    prediction_idx = scores.index(scores.max)
    
    {
      label: LABELS[prediction_idx],
      score: scores[prediction_idx],
      all_scores: scores,
      text: @text
    }
  end
end
