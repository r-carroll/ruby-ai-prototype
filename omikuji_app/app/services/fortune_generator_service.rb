# omikuji_app/app/services/fortune_generator_service.rb

class FortuneGeneratorService
  RANKS = {
    positive: ["大吉 (Daikichi)", "中吉 (Chukichi)", "小吉 (Shokichi)"],
    negative: ["末吉 (Suekichi)", "凶 (Kyo)"]
  }.freeze

  PROMPTS = {
    positive: "吉兆の兆しあり。お告げ：「",
    negative: "慎重に歩むべし。お告げ：「"
  }.freeze

  def initialize(sentiment_label)
    @sentiment = sentiment_label
    @loader = ModelLoader.instance
    @session = @loader.gpt_session
    @tokenizer = @loader.gpt_tokenizer
    @temperature = 0.8
    @top_k = 40
    @repetition_penalty = 1.2
  end

  def generate
    return "AI fortune teller is resting... (Model not loaded)" unless @session && @tokenizer

    # 1. Pick a rank and a starting prompt
    rank = RANKS[@sentiment].sample
    prompt = PROMPTS[@sentiment]
    
    # 2. Tokenize the prompt
    tokens = @tokenizer.encode(prompt).ids
    
    # 3. Sampling Generation (generate up to 40 tokens)
    40.times do
      position_ids = (0...tokens.size).to_a
      attention_mask = [1] * tokens.size
      inputs = { 
        "input_ids" => [tokens],
        "position_ids" => [position_ids],
        "attention_mask" => [attention_mask]
      }
      
      outputs = @session.predict(inputs)
      logits = outputs["logits"][0].last # Get logits for the last token
      
      # Apply Repetition Penalty
      tokens.uniq.each do |token_id|
        if logits[token_id] > 0
          logits[token_id] /= @repetition_penalty
        else
          logits[token_id] *= @repetition_penalty
        end
      end

      # Apply Temperature
      logits = logits.map { |l| l / @temperature }

      # Top-K Sampling
      # 1. Map to indices and sort by value
      indexed_logits = logits.each_with_index.to_a.sort_by { |val, idx| -val }
      # 2. Keep only top K
      top_k_logits = indexed_logits.take(@top_k)
      
      # 3. Softmax and Sample
      values = top_k_logits.map(&:first)
      max_val = values.max
      exp_values = values.map { |v| Math.exp(v - max_val) }
      sum_exp = exp_values.sum
      probs = exp_values.map { |e| e / sum_exp }

      # Random sampling from the probability distribution
      r = rand
      cumulative = 0
      next_token_id = top_k_logits.last.last # Default to last if something goes wrong
      
      probs.each_with_index do |p, i|
        cumulative += p
        if r < cumulative
          next_token_id = top_k_logits[i].last
          break
        end
      end

      tokens << next_token_id
      
      # Stop if we hit an End-of-Text or a closing bracket
      break if next_token_id == 2 || @tokenizer.decode([next_token_id]).include?("」")
    end

    # 4. Decode and Cleanup
    full_text = @tokenizer.decode(tokens)
    generated_content = full_text.sub(prompt, "").split("」").first
    
    {
      rank: rank,
      fortune: "「#{generated_content}」"
    }
  end
end
