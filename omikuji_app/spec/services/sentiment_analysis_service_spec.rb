# omikuji_app/spec/services/sentiment_analysis_service_spec.rb

require 'rails_helper'

RSpec.describe SentimentAnalysisService do
  describe "#predict" do
    it "correctly predicts positive sentiment for a happy Japanese sentence" do
      # "I am very happy today!"
      text = "今日はとても嬉しいです！"
      service = SentimentAnalysisService.new(text)
      result = service.predict

      expect(result).to have_key(:label)
      expect([:positive, :negative]).to include(result[:label])
      # Since BERT is trained on reviews, "Happy" should likely be positive
      expect(result[:label]).to eq(:positive)
    end

    it "correctly predicts negative sentiment for an unhappy Japanese sentence" do
      # "The food was bad and the service was slow."
      text = "料理は不味いし、対応も遅かったです。"
      service = SentimentAnalysisService.new(text)
      result = service.predict

      expect(result[:label]).to eq(:negative)
    end
  end
end
