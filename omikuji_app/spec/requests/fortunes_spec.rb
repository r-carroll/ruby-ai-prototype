require 'rails_helper'

RSpec.describe "Fortunes", type: :request do
  describe "GET /index" do
    it "returns http success" do
      get fortunes_path
      expect(response).to have_http_status(:success)
    end
  end

  describe "POST /create" do
    it "redirects after creating a fortune and enqueues the job" do
      expect {
        post fortunes_path, params: { fortune: { input_text: "今日はとても良い日です。" } }
      }.to have_enqueued_job(FortuneGenerationJob)
      
      expect(response).to have_http_status(:redirect)
      expect(Fortune.last.status).to eq("pending")
    end

    it "responds with turbo_stream" do
      post fortunes_path, params: { fortune: { input_text: "今日はとても良い日です。" } }, as: :turbo_stream
      expect(response).to have_http_status(:success)
      expect(response.content_type).to include("text/vnd.turbo-stream.html")
    end
  end

  describe "GET /show" do
    it "returns http success for completed fortune" do
      fortune = Fortune.create!(
        input_text: "テスト",
        sentiment_label: "positive",
        score: 0.9,
        rank: "大吉",
        fortune_text: "良いことがあります。",
        status: "completed"
      )
      get fortune_path(fortune)
      expect(response).to have_http_status(:success)
      expect(response.body).to include("大吉")
    end

    it "returns http success for pending fortune" do
      fortune = Fortune.create!(
        input_text: "テスト",
        status: "pending"
      )
      get fortune_path(fortune)
      expect(response).to have_http_status(:success)
      expect(response.body).to include("神託を待つ")
    end
  end
end
