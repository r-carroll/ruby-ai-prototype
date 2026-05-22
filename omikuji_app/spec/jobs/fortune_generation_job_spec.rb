require 'rails_helper'

RSpec.describe FortuneGenerationJob, type: :job do
  include ActiveJob::TestHelper

  let(:fortune) { Fortune.create!(input_text: "今日は楽しい！", status: "pending") }

  it "enqueues the job" do
    expect {
      FortuneGenerationJob.perform_later(fortune.id)
    }.to have_enqueued_job(FortuneGenerationJob).with(fortune.id)
  end

  it "updates the fortune status and fields" do
    # We might want to mock the services here to avoid loading the model in tests,
    # but since this is a small project and we want to verify integration,
    # let's assume the test environment handles it or we mock the heavy parts.
    
    # Mocking the heavy ML services to avoid loading the 400MB model in CI/tests
    sentiment_mock = instance_double(SentimentAnalysisService)
    allow(SentimentAnalysisService).to receive(:new).and_return(sentiment_mock)
    allow(sentiment_mock).to receive(:predict).and_return({ label: :positive, score: 0.95 })

    generator_mock = instance_double(FortuneGeneratorService)
    allow(FortuneGeneratorService).to receive(:new).and_return(generator_mock)
    allow(generator_mock).to receive(:generate).and_return({ rank: "大吉 (Daikichi)", fortune: "素晴らしい一日になります。" })

    perform_enqueued_jobs do
      FortuneGenerationJob.perform_later(fortune.id)
    end

    fortune.reload
    expect(fortune.status).to eq("completed")
    expect(fortune.sentiment_label).to eq("positive")
    expect(fortune.rank).to eq("大吉 (Daikichi)")
    expect(fortune.fortune_text).to eq("素晴らしい一日になります。")
  end
end
