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
    @temperature = 0.6 # Lowered from 0.8 to reduce "wild" hallucinations
    @top_p = 0.9 # Lowered from 0.95 to cut off more noise
    @repetition_penalty = 1.4
  end

  def generate
    return "AI fortune teller is resting... (Model not loaded)" unless @session && @tokenizer

    rank = RANKS[@sentiment].sample
    prompt = PROMPTS[@sentiment]
    
    # Pre-seed with the opening bracket
    prompt_tokens = @tokenizer.encode(prompt).ids
    bracket_id = @tokenizer.encode("「").ids.last
    tokens = prompt_tokens + [bracket_id]
    
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

      # Repetition Penalty
      tokens.uniq.each { |tid| logits[tid] /= @repetition_penalty if logits[tid] > 0 }

      # Ban weird symbols and excessive ASCII to keep it "clean"
      [33, 34, 35, 36, 37, 38, 42, 60, 62, 124].each { |id| logits[id] = -100 }

      # Temperature & Top-P Sampling
      logits = logits.map { |l| l / @temperature }
      indexed_logits = logits.each_with_index.to_a.sort_by { |v, i| -v }
      
      max_v = indexed_logits[0][0]
      exp_v = indexed_logits.map { |v, i| Math.exp(v - max_v) }
      sum_exp = exp_v.sum
      probs = exp_v.map { |e| e / sum_exp }
      
      cumulative_prob = 0
      cutoff_index = 0
      probs.each_with_index { |p, i| cumulative_prob += p; cutoff_index = i; break if cumulative_prob > @top_p }
      
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
      decoded_char = @tokenizer.decode([next_token_id])
      break if next_token_id == 2 || decoded_char.include?("」") || decoded_char.include?("。")
    end

    generated_tokens = tokens[(prompt_tokens.size + 1)..-1] || []
    generated_content = @tokenizer.decode(generated_tokens)
    
    # STRIP WEB NOISE & HALLUCINATIONS
    # 1. Remove URLs and domain fragments
    generated_content = generated_content.gsub(/https?:\/\/[\S]+/, '')
    generated_content = generated_content.gsub(/[a-zA-Z0-9\-\.]+\.(jp|com|net|org|info)\/\S*/, '')
    
    # 2. Remove web markers like [続きを読む], (12月10日), etc.
    generated_content = generated_content.gsub(/\[[^\]]+\]/, '')
    generated_content = generated_content.gsub(/（[^）]+）|\([^\)]+\)/, '')
    
    # 3. Strip web symbols and dates
    generated_content = generated_content.gsub(/[♪☆★■◆●○]|[0-9]+月[0-9]+日/, '')

    # 4. Standard cleanup
    generated_content = generated_content.gsub(/[」。」]/, '').strip
    
    # 5. Fallback
    generated_content = "穏やかな心で過ごせば、自ずと道は開けるでしょう" if generated_content.length < 3
    
    {
      rank: rank,
      fortune: "「#{generated_content}」"
    }
  end
end
