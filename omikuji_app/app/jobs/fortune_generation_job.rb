class FortuneGenerationJob < ApplicationJob
  queue_as :default

  def perform(fortune_id)
    fortune = Fortune.find(fortune_id)
    return if fortune.status == "completed"

    # 1. Run Sentiment Analysis (The "Ear")
    sentiment_result = SentimentAnalysisService.new(fortune.input_text).predict
    fortune.sentiment_label = sentiment_result[:label].to_s
    fortune.score = sentiment_result[:score]

    # 2. Run Fortune Generation (The "Voice")
    generation_result = FortuneGeneratorService.new(sentiment_result[:label]).generate
    fortune.fortune_text = generation_result[:fortune]
    fortune.rank = generation_result[:rank]
    
    fortune.status = "completed"
    fortune.save!
  end
end
