# omikuji_app/app/services/fortune_generator_service.rb

class FortuneGeneratorService
  RANKS = {
    positive: ["大吉 (Daikichi)", "中吉 (Chukichi)", "小吉 (Shokichi)"],
    negative: ["末吉 (Suekichi)", "凶 (Kyo)"],
    neutral: ["吉 (Kichi)", "平 (Heira)"]
  }.freeze

  PROMPTS = {
    positive: "【神託】素晴らしい運気です。助言：",
    negative: "【神託】今は嵐の前の静けさ。助言：",
    neutral: "【神託】穏やかな一日となるでしょう。助言："
  }.freeze

  def initialize(sentiment_label)
    @sentiment = (sentiment_label || :neutral).to_sym
    @loader = ModelLoader.instance
    @markov_service = @loader.markov_service
  end

  def generate
    unless @markov_service
      return {
        rank: "平 (Heira)",
        fortune: "「おみくじは現在準備中です」"
      }
    end

    rank = (RANKS[@sentiment] || RANKS[:neutral]).sample
    prefix = PROMPTS[@sentiment] || PROMPTS[:neutral]
    
    # Generate a unique sentence using our custom service
    generated = @markov_service.generate_sentence
    
    # Clean up MeCab spaces
    clean_fortune = generated.gsub(" ", "")
    
    # Ensure it ends with a period if it doesn't have one
    clean_fortune += "。" unless clean_fortune.end_with?("。", "！", "？")

    {
      rank: rank,
      fortune: "#{prefix}「#{clean_fortune}」"
    }
  end
end
