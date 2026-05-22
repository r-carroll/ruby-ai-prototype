require 'rails_helper'

RSpec.describe Fortune, type: :model do
  it "is valid with valid attributes" do
    fortune = Fortune.new(
      input_text: "今日はとても良い日です。",
      sentiment_label: "positive",
      score: 0.9,
      rank: "大吉",
      fortune_text: "素晴らしい一日になるでしょう。"
    )
    expect(fortune).to be_valid
  end

  it "is invalid without input_text" do
    fortune = Fortune.new(input_text: nil)
    expect(fortune).not_to be_valid
  end
end
