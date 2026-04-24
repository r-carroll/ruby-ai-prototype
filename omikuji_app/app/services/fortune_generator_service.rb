# omikuji_app/app/services/fortune_generator_service.rb

class FortuneGeneratorService
  RANKS = {
    positive: ["大吉 (Daikichi)", "中吉 (Chukichi)", "小吉 (Shokichi)"],
    negative: ["末吉 (Suekichi)", "凶 (Kyo)"]
  }.freeze

  PROMPTS = {
    positive: "【神託】素晴らしい運気です。和歌のごとく、心穏やかに過ごせば幸運が舞い込みます。助言：「",
    negative: "【神託】今は嵐の前の静けさ。焦らず、自分を見つめ直す時です。助言：「"
  }.freeze

  def initialize(sentiment_label)
    @sentiment = sentiment_label
    @loader = ModelLoader.instance
    @session = @loader.gpt_session
    @tokenizer = @loader.gpt_tokenizer
    @temperature = 0.7
    @top_p = 0.9
    @repetition_penalty = 1.5
  end

  def generate
    return "AI fortune teller is resting... (Model not loaded)" unless @session && @tokenizer

    rank = RANKS[@sentiment].sample
    prompt = PROMPTS[@sentiment]
    tokens = @tokenizer.encode(prompt).ids
    
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

      # 2. Ban non-Japanese characters (roughly) to prevent "ove-one" English slips
      # We allow common Japanese punctuation and all characters above ID 500 
      # (Most JP models put Latin/English in the 0-200 range)
      (33..126).each { |id| logits[id] = -100 } # Banning most ASCII printable chars

      # 3. Temperature
      logits = logits.map { |l| l / @temperature }

      # 4. Top-P (Nucleus) Sampling
      indexed_logits = logits.each_with_index.to_a.sort_by { |v, i| -v }
      
      # Convert to probabilities for Top-P calculation
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
      
      # Re-calculate probabilities for the selected subset
      subset_exp = top_p_logits.map { |v, i| Math.exp(v - max_v) }
      subset_sum = subset_exp.sum
      subset_probs = subset_exp.map { |e| e / subset_sum }

      # Sample
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
      
      # Stop if we hit EOS or a closing bracket
      decoded_char = @tokenizer.decode([next_token_id])
      break if next_token_id == 2 || decoded_char.include?("」")
    end

    full_text = @tokenizer.decode(tokens)
    generated_content = full_text.sub(prompt, "").split("」").first
    
    {
      rank: rank,
      fortune: "「#{generated_content}」"
    }
  end
end
