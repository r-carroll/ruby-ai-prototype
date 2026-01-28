require 'onnxruntime'
require_relative 'japanese_bert_tokenizer'

class JapaneseSentimentAnalyzer
  attr_reader :max_length

  # Labels: 0 = negative, 1 = positive
  LABELS = { 0 => :negative, 1 => :positive }

  def initialize(model_path: File.join(__dir__, "../models/model.onnx"), max_length: 128)
    @max_length = max_length
    @tokenizer = JapaneseBertTokenizer.new
    @model = OnnxRuntime::Model.new(model_path)
  end

  def predict(text)
    # 1. Tokenize
    encoded = @tokenizer.encode(text, max_length: @max_length)
    
    # 2. Prepare Inputs
    # The model expects batched inputs: [1, max_length]
    # We wrap the flat arrays in another array to create the batch dimension
    input_ids = [encoded[:input_ids]]
    attention_mask = [encoded[:attention_mask]]

    input_feed = {
      "input_ids" => input_ids,
      "attention_mask" => attention_mask
    }

    # 3. Run Inference
    # Returns a Hash, e.g. {"logits" => [[val1, val2]]}
    output = @model.predict(input_feed)
    
    # Extract logits for the first (and only) item in batch
    # Output structure depends on model export names, usually "logits" or "last_hidden_state"
    # Assuming standard classification head output named "logits" or the first output
    raw_logits = output.values.first.first

    # 4. Softmax & Decoding
    probs = softmax(raw_logits)
    
    # Get max probability label
    max_score_index = probs.each_with_index.max[1]
    label = LABELS[max_score_index]
    score = probs[max_score_index]

    { label: label, score: score, probabilities: probs }
  end

  private

  def softmax(logits)
    # Simple softmax implementation for 1D array
    max_logit = logits.max
    exps = logits.map { |x| Math.exp(x - max_logit) }
    sum_exps = exps.sum
    exps.map { |x| x / sum_exps }
  end
end
