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

    # 4. Process Results (Softmax)
    # logits is an array: [neg_logit, pos_logit]
    logits_array = results["output"].first
    
    # Standard Softmax implementation
    max_logit = logits_array.max
    exp_logits = logits_array.map { |l| Math.exp(l - max_logit) }
    sum_exp = exp_logits.sum
    probabilities = exp_logits.map { |e| e / sum_exp }
    
    # Higher probability wins
    prediction_idx = probabilities.index(probabilities.max)
    
    {
      label: LABELS[prediction_idx],
      score: probabilities[prediction_idx], # This will now be between 0.0 and 1.0
      all_scores: probabilities,
      text: @text
    }
  end
end
