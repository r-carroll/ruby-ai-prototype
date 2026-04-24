# omikuji_app/app/services/fortune_generator_service.rb

class FortuneGeneratorService
  RANKS = {
    positive: ["大吉 (Daikichi)", "中吉 (Chukichi)", "小吉 (Shokichi)"],
    negative: ["末吉 (Suekichi)", "凶 (Kyo)"]
  }.freeze

  PROMPTS = {
    positive: "運勢は大吉です。アドバイス：",
    negative: "運勢は末吉です。アドバイス："
  }.freeze

  def initialize(sentiment_label)
    @sentiment = sentiment_label
    @loader = ModelLoader.instance
    @session = @loader.gpt_session
    @tokenizer = @loader.gpt_tokenizer
  end

  def generate
    return "AI fortune teller is resting... (Model not loaded)" unless @session && @tokenizer

    # 1. Pick a rank and a starting prompt
    rank = RANKS[@sentiment].sample
    prompt = "#{PROMPTS[@sentiment]}「"
    
    # 2. Tokenize the prompt
    tokens = @tokenizer.encode(prompt).ids
    
    # 3. Simple Greedy Generation (generate 20 tokens)
    20.times do
      # GPT-2 expects [batch_size, seq_len]
      # Some models also expect position_ids
      position_ids = (0...tokens.size).to_a
      attention_mask = [1] * tokens.size
      inputs = { 
        "input_ids" => [tokens],
        "position_ids" => [position_ids],
        "attention_mask" => [attention_mask]
      }
      
      # The ONNX model returns logits
      outputs = @session.predict(inputs)
      logits = outputs["logits"] # Shape: [1, seq_len, vocab_size]
      
      # Get the logits for the LAST token in the sequence
      last_token_logits = logits[0].last
      
      # Greedy choice: pick the highest logit
      next_token_id = last_token_logits.index(last_token_logits.max)
      
      tokens << next_token_id
      
      # Stop if we hit an End-of-Text token (usually 0 or 2 in many JP models)
      break if next_token_id == 2 
    end

    # 4. Decode the result
    full_text = @tokenizer.decode(tokens)
    
    # Clean up the output (remove the prompt and add a closing bracket)
    generated_content = full_text.sub(prompt, "").split("」").first
    
    {
      rank: rank,
      fortune: "「#{generated_content}」"
    }
  end
end
