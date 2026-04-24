# omikuji_app/app/services/fortune_generator_service.rb

class FortuneGeneratorService
  RANKS = {
    positive: ["大吉 (Daikichi)", "中吉 (Chukichi)", "小吉 (Shokichi)"],
    negative: ["末吉 (Suekichi)", "凶 (Kyo)"]
  }.freeze

  PROMPTS = {
    positive: "【神託】素晴らしい運気です。和歌のごとく、心穏やかに過ごせば幸運が舞い込みます。助言：",
    negative: "【神託】今は嵐の前の静けさ。焦らず、自分を見つめ直す時です。助言："
  }.freeze

  def initialize(sentiment_label)
    @sentiment = sentiment_label
    @loader = ModelLoader.instance
    @session = @loader.gpt_session
    @tokenizer = @loader.gpt_tokenizer
    @temperature = 0.8
    @top_p = 0.95
    @repetition_penalty = 1.2
  end

  def generate
    return "AI fortune teller is resting... (Model not loaded)" unless @session && @tokenizer

    rank = RANKS[@sentiment].sample
    prompt = PROMPTS[@sentiment]
    
    # We want the model to generate content following the prompt
    # We'll pre-seed it with an opening bracket to encourage generation
    prompt_tokens = @tokenizer.encode(prompt).ids
    bracket_id = @tokenizer.encode("「").ids.last
    
    tokens = prompt_tokens + [bracket_id]
    
    # 40 tokens max for the wisdom
    40.times do
      position_ids = (0...tokens.size).to_a
      attention_mask = [1] * tokens.size
      inputs = { 
        "input_ids" => [tokens],
        "position_ids" => [position_ids],
        "attention_mask" => [attention_mask]
      }
      
      outputs = @session.predict(inputs)
      logits = outputs["logits"][0].last

      # 1. Repetition Penalty
      tokens.uniq.each { |tid| logits[tid] /= @repetition_penalty if logits[tid] > 0 }

      # 2. Temperature
      logits = logits.map { |l| l / @temperature }

      # 3. Top-P (Nucleus) Sampling
      indexed_logits = logits.each_with_index.to_a.sort_by { |v, i| -v }
      
      max_v = indexed_logits[0][0]
      exp_v = indexed_logits.map { |v, i| Math.exp(v - max_v) }
      sum_exp = exp_v.sum
      probs = exp_v.map { |e| e / sum_exp }
      
      cumulative_prob = 0
      cutoff_index = 0
      probs.each_with_index do |p, i|
        cumulative_prob += p
        cutoff_index = i
        break if cumulative_prob > @top_p
      end
      
      top_p_logits = indexed_logits.take(cutoff_index + 1)
      subset_exp = top_p_logits.map { |v, i| Math.exp(v - max_v) }
      subset_sum = subset_exp.sum
      subset_probs = subset_exp.map { |e| e / subset_sum }

      r = rand
      cum = 0
      next_token_id = top_p_logits.last.last
      subset_probs.each_with_index do |p, i|
        cum += p
        if r < cum
          next_token_id = top_p_logits[i].last
          break
        end
      end

      tokens << next_token_id
      
      # Stop if we hit EOS or a closing bracket or period
      decoded_char = @tokenizer.decode([next_token_id])
      break if next_token_id == 2 || decoded_char.include?("」") || decoded_char.include?("。")
    end

    # SLICE generated tokens only (everything after prompt_tokens + the bracket we forced)
    generated_tokens = tokens[(prompt_tokens.size + 1)..-1] || []
    generated_content = @tokenizer.decode(generated_tokens)
    
    # Cleanup: remove any trailing closing brackets or periods that survived
    generated_content = generated_content.gsub(/[」。」]/, '').strip
    
    # If the model still returned nothing, give a fallback "Mystic silence"
    generated_content = "今は語らぬが吉。心の静寂を大切にせよ" if generated_content.blank?
    
    {
      rank: rank,
      fortune: "「#{generated_content}」"
    }
  end
end
